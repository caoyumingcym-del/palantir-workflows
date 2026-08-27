"""
Cache the expensive outputs of the transcriptome stage.

``PipelineConfig.use_checkpoints`` and ``checkpoint_dir`` existed from v1.1.0
but nothing read or wrote them -- the directory was created and left empty. So
every re-run repeated HVG selection, regress_out, scale, PCA, harmony, UMAP and
Leiden even when nothing about the input or the thresholds had changed. On a
187k-cell object that is the bulk of the wall clock.

What is cached is only what is expensive and stable: the PCA and UMAP
coordinates, the cluster labels, the HVG list, and the small obs columns the
stage adds (phase, scores, doublet calls). ``X_log`` is deliberately NOT
cached -- it is a normalise + log1p away from the counts, which is seconds, and
it is the largest array in the stage.

Invalidation is by content key, not by timestamp alone. The key covers
everything that would change the answer: the input file (path, size, mtime),
the cell barcodes that survived QC, and the embedding settings. Change any of
them and the cache misses rather than silently returning a stale embedding --
which matters, because a stale embedding is exactly the failure mode that hid a
broken doublet step for three versions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CACHE_VERSION = 3

# obs columns the transcriptome stage produces and that downstream stages read.
#
# The cluster column is named ``cluster``, not ``leiden`` -- getting this wrong
# meant the cached path returned an obs frame with no clusters at all and the
# report blew up on ``res.obs["cluster"]``. Anything the stage writes onto obs
# and a later stage reads has to be listed here.
CACHED_OBS_COLUMNS = (
    "cluster", "phase", "S_score", "G2M_score",
    "doublet_score", "predicted_doublet", "total_counts",
)

# Columns that must be present after a cache hit, or the hit is unusable. A
# miss that recomputes is always better than a hit that crashes the report.
REQUIRED_OBS_COLUMNS = ("cluster",)


def embedding_key(
    h5ad_path: Path | None,
    obs_names: list[str],
    cfg: Any,
) -> str:
    """Content hash of everything that would change the embedding."""
    ecfg = cfg.embedding
    parts: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "n_cells": len(obs_names),
        # The barcode set, not just its size: two different QC threshold sets
        # can retain the same number of cells.
        "obs_digest": hashlib.sha1(
            "\n".join(map(str, obs_names)).encode()
        ).hexdigest(),
    }
    if h5ad_path is not None:
        p = Path(h5ad_path)
        try:
            st = p.stat()
            parts["input"] = [str(p), st.st_size, int(st.st_mtime)]
        except OSError:
            parts["input"] = [str(p)]
    for name in (
        "target_sum", "n_top_genes", "exclude_mito_from_hvg",
        "exclude_ribo_from_hvg", "hvg_flavor", "regress_out",
        "scale_max_value", "n_pcs", "n_neighbors", "leiden_resolution",
        "batch_correct", "score_cell_cycle", "detect_doublets", "random_state",
    ):
        if hasattr(ecfg, name):
            v = getattr(ecfg, name)
            parts[name] = list(v) if isinstance(v, (tuple, list)) else v
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _paths(cfg: Any, key: str) -> tuple[Path, Path]:
    d = Path(cfg.checkpoint_dir)
    return d / f"embedding_{key}.npz", d / f"embedding_{key}.json"


def save(result: Any, cfg: Any, key: str) -> Path | None:
    """Write the cacheable parts of an EmbeddingResult. Never raises."""
    npz_path, meta_path = _paths(cfg, key)
    try:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        if result.pca is not None:
            arrays["pca"] = np.asarray(result.pca, dtype=np.float32)
        if result.umap is not None:
            arrays["umap"] = np.asarray(result.umap, dtype=np.float32)
        arrays["obs_names"] = np.asarray(
            [str(x) for x in result.obs.index], dtype=object
        )
        obs_keep = [c for c in CACHED_OBS_COLUMNS if c in result.obs.columns]
        missing = [c for c in REQUIRED_OBS_COLUMNS if c not in obs_keep]
        if missing:
            # Writing a checkpoint that cannot be loaded back is worse than
            # writing none: the next run would hit it and then fail.
            return None
        for c in obs_keep:
            col = result.obs[c]
            if isinstance(col.dtype, pd.CategoricalDtype):
                col = col.astype(str)
            arrays[f"obs__{c}"] = col.to_numpy()
        np.savez_compressed(npz_path, **arrays)
        meta_path.write_text(json.dumps({
            "key": key,
            "hvg": list(result.hvg),
            "var_names_digest": hashlib.sha1(
                "\n".join(map(str, result.var_names)).encode()).hexdigest(),
            "n_cells": int(len(result.obs)),
            "obs_columns": obs_keep,
            "backend": result.backend,
            "batch_corrected": result.batch_corrected,
        }, indent=1))
        return npz_path
    except Exception:
        for p in (npz_path, meta_path):
            try:
                p.unlink()
            except OSError:
                pass
        return None


def load(cfg: Any, key: str, obs_names: list[str], var_names: list[str]):
    """``(payload, reason)``. payload is None on a miss; reason says why."""
    npz_path, meta_path = _paths(cfg, key)
    if not (npz_path.exists() and meta_path.exists()):
        return None, "no cached embedding for this input and configuration"
    try:
        meta = json.loads(meta_path.read_text())
        with np.load(npz_path, allow_pickle=True) as z:
            cached_obs = [str(x) for x in z["obs_names"]]
            if cached_obs != [str(x) for x in obs_names]:
                return None, "cached embedding covers different cells"
            digest = hashlib.sha1(
                "\n".join(map(str, var_names)).encode()).hexdigest()
            if meta.get("var_names_digest") != digest:
                return None, "cached embedding was built on different genes"
            payload = {
                "pca": np.asarray(z["pca"]) if "pca" in z else None,
                "umap": np.asarray(z["umap"]) if "umap" in z else None,
                "hvg": list(meta.get("hvg", [])),
                "backend": meta.get("backend", "scanpy"),
                "batch_corrected": meta.get("batch_corrected", "none"),
                "obs": {
                    c: np.asarray(z[f"obs__{c}"])
                    for c in meta.get("obs_columns", [])
                    if f"obs__{c}" in z
                },
            }
        absent = [c for c in REQUIRED_OBS_COLUMNS if c not in payload["obs"]]
        if absent:
            return None, (
                "cached embedding is missing "
                + ", ".join(absent)
                + " (written by an older version)"
            )
        return payload, "hit"
    except Exception as exc:
        return None, f"cached embedding could not be read ({exc})"


def apply(payload: dict, obs: pd.DataFrame) -> pd.DataFrame:
    """Put the cached obs columns back onto a fresh obs frame."""
    out = obs.copy()
    for c, values in payload["obs"].items():
        if c == "cluster":
            # Restored as a Categorical, matching what the live stage produces,
            # so plotting and grouping behave identically either way.
            out[c] = pd.Categorical([str(v) for v in values])
        elif c == "predicted_doublet":
            out[c] = np.asarray(values).astype(bool)
        else:
            out[c] = values
    return out
