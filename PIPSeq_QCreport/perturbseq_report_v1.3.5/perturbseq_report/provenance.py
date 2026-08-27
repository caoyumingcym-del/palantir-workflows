"""
Detect what has already been done to an .h5ad, so the pipeline does not redo it.

A Perturb-seq h5ad arrives in one of two very different states:

* **unanalysed** -- raw counts in ``X``, no QC columns, no embedding. Everything
  must be computed.
* **already analysed** -- ``X`` normalised and log-transformed, QC metrics in
  ``obs``, ``X_pca``/``X_umap`` in ``obsm``, Leiden labels in ``obs``. Almost
  nothing needs computing, and recomputing is actively harmful.

Redoing the work on an already-analysed object is not merely wasteful. Running
``normalize_total`` + ``log1p`` over values that are already log-transformed
produces log(log(x)) -- silently, with no error -- and every downstream number
built on it is wrong. Separately, recomputing an embedding for 300,000 cells
costs tens of gigabytes and hours, which is how a run that had nothing to do
ends up being killed for memory.

This module inspects the object and reports what is already present. The stages
then reuse it. ``--force-recompute`` overrides everything, for when you want the
pipeline's own processing regardless.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import ModalityConfig
from .stats import is_sparse

# X state labels
RAW_COUNTS = "raw counts"
NORMALISED = "normalised (not log-transformed)"
LOG1P = "normalised and log-transformed"
UNKNOWN = "indeterminate"


@dataclass
class PriorAnalysis:
    """What the input object already contains."""

    x_state: str = UNKNOWN
    x_evidence: list[str] = field(default_factory=list)
    counts_layer: str | None = None          # layer holding raw counts, if any

    qc_columns: dict[str, str] = field(default_factory=dict)  # canonical -> obs col
    pca_key: str | None = None
    umap_key: str | None = None
    cluster_column: str | None = None
    hvg_column: str | None = None
    doublet_score_column: str | None = None
    doublet_call_column: str | None = None

    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- verdicts
    @property
    def x_is_raw_counts(self) -> bool:
        return self.x_state == RAW_COUNTS

    @property
    def has_counts(self) -> bool:
        """Are raw counts available anywhere (X or a layer)?"""
        return self.x_is_raw_counts or self.counts_layer is not None

    @property
    def has_qc_metrics(self) -> bool:
        return len(self.qc_columns) >= 2

    @property
    def has_embedding(self) -> bool:
        return self.pca_key is not None

    @property
    def can_skip_embedding(self) -> bool:
        """Enough present to skip normalisation, HVG, PCA, UMAP and clustering."""
        return (
            self.pca_key is not None
            and self.umap_key is not None
            and self.cluster_column is not None
        )

    def summary(self) -> str:
        bits = [f"X holds {self.x_state}"]
        if self.counts_layer:
            bits.append(f"raw counts in layers['{self.counts_layer}']")
        if self.has_qc_metrics:
            bits.append(f"{len(self.qc_columns)} QC metric(s)")
        if self.pca_key:
            bits.append(f"PCA in obsm['{self.pca_key}']")
        if self.umap_key:
            bits.append(f"UMAP in obsm['{self.umap_key}']")
        if self.cluster_column:
            bits.append(f"clusters in obs['{self.cluster_column}']")
        if self.doublet_call_column:
            bits.append(f"doublet calls in obs['{self.doublet_call_column}']")
        return "; ".join(bits)


# ===========================================================================
# X state
# ===========================================================================
def _sample_values(X, n_rows: int = 500, n_values: int = 20000) -> np.ndarray:
    """A cheap sample of non-zero values from the top of the matrix."""
    try:
        block = X[:n_rows]
    except Exception:
        return np.array([])
    if is_sparse(block):
        data = np.asarray(block.data, dtype=np.float64)
    else:
        arr = np.asarray(block, dtype=np.float64)
        data = arr.ravel()
    data = data[np.isfinite(data)]
    data = data[data != 0]
    if data.size > n_values:
        data = data[:n_values]
    return data


def detect_x_state(adata: Any) -> tuple[str, list[str]]:
    """Decide whether X holds raw counts, normalised values, or log values.

    Integrality is checked FIRST and is authoritative, not ``uns['log1p']``.

    v1.3.3's real MDL-1856 run exposed why: that h5ad's ``X`` sampled as 20,000
    non-zero values, min 1, max 94, ALL EXACT INTEGERS -- unambiguously raw (or
    otherwise untransformed) count data, since real ``log1p`` output on scRNA
    data is continuous and essentially never lands on an exact integer at this
    sample size. But ``uns['log1p']`` was also present (stale -- almost
    certainly left over from an earlier processing pass whose output was later
    replaced with raw counts, an upstream data-hygiene problem this pipeline
    cannot fix, only detect). The previous version of this function checked
    the flag before re-confirming it against integrality, so it returned
    ``LOG1P`` outright on seeing the flag, integrality evidence notwithstanding.
    Every downstream consumer of ``X_log`` (percent_knockdown,
    differential_expression, perturbation_score) un-logs with ``expm1()``,
    which explodes raw integer counts (e.g. ``expm1(94)`` ~ 10^40) into the
    same absurd mean-expression values (1e13-1e23) that motivated the earlier
    NORMALISED-vs-LOG1P fix in ``gex.py``'s ``_reuse_embedding`` -- a different
    root cause, same visible symptom.

    A ``uns`` flag is metadata that can go stale the moment ``X`` is
    overwritten without updating it. Integrality is a hard mathematical fact
    about the data actually present right now. When they disagree, the data
    wins, and the disagreement itself is reported rather than silently
    resolved in the flag's favour.

    Three signals, in the order they are actually trusted:

    1. Integrality -- raw UMI counts are whole numbers, and log1p output is
       essentially never an exact integer. Decisive on its own.
    2. ``uns['log1p']`` -- scanpy writes this key when it applies ``log1p``.
       Used to confirm/classify non-integer data, not to override integrality.
    3. Magnitude -- ``log1p`` of normalised expression rarely exceeds ~12,
       whereas normalised-but-unlogged values routinely reach the hundreds.
       Used as a fallback when the flag is absent.
    """
    evidence: list[str] = []

    uns = getattr(adata, "uns", {}) or {}
    logged_flag = "log1p" in uns

    data = _sample_values(getattr(adata, "X", None))
    if data.size == 0:
        return UNKNOWN, ["could not sample any non-zero values from X"]

    all_int = bool(np.allclose(data, np.round(data), atol=1e-8))
    vmax = float(np.max(data))
    vmin = float(np.min(data))

    evidence.append(
        f"sampled {data.size:,} non-zero values: min {vmin:.4g}, max {vmax:.4g}, "
        f"{'all integers' if all_int else 'non-integer values present'}"
    )
    if logged_flag:
        evidence.append("uns['log1p'] is present, which scanpy writes after log1p")

    if all_int:
        if logged_flag:
            evidence.append(
                "CONTRADICTION: uns['log1p'] claims X was log1p-transformed, "
                f"but all {data.size:,} sampled non-zero values are exact "
                "integers, which real log1p output essentially never produces "
                "at this sample size. Trusting the data over the (evidently "
                "stale) uns flag -- treating X as raw counts."
            )
        return RAW_COUNTS, evidence
    if logged_flag:
        return LOG1P, evidence
    if vmax <= 12.0:
        evidence.append(
            f"maximum value {vmax:.3g} is in the range typical of log1p data"
        )
        return LOG1P, evidence
    evidence.append(
        f"non-integer values reaching {vmax:.3g} look normalised but not "
        f"log-transformed"
    )
    return NORMALISED, evidence


def find_counts_layer(adata: Any, preferred: str | None = None) -> str | None:
    """Locate a layer holding raw integer counts."""
    layers = getattr(adata, "layers", None)
    if not layers:
        return None
    names = list(layers.keys())
    order = ([preferred] if preferred else []) + [
        n for n in ("counts", "raw_counts", "umi", "X_counts", "spliced") if n in names
    ] + [n for n in names if n not in ("counts", "raw_counts", "umi", "X_counts")]

    for name in order:
        if not name or name not in names:
            continue
        data = _sample_values(layers[name])
        if data.size and np.allclose(data, np.round(data), atol=1e-8):
            return name
    return None


# ===========================================================================
# Prior results
# ===========================================================================
_QC_ALIASES = {
    "total_counts": ("total_counts", "nCount_RNA", "n_counts", "total_umis"),
    "n_genes_by_counts": (
        "n_genes_by_counts", "nFeature_RNA", "n_genes", "genes_detected",
    ),
    "pct_counts_mt": (
        "pct_counts_mt", "percent_mt", "percent.mt", "pct_mito", "pct_counts_mito",
    ),
}

_PCA_KEYS = ("X_pca_harmony", "X_pca", "X_PCA", "X_pca_hvg")
_UMAP_KEYS = ("X_umap", "X_UMAP", "X_umap_harmony")
_HVG_COLS = ("highly_variable", "vst.variable", "is_hvg")


def detect(
    adata: Any,
    cfg: ModalityConfig,
    counts_layer_hint: str | None = None,
) -> PriorAnalysis:
    """Inspect an AnnData and report everything already computed on it."""
    p = PriorAnalysis()
    obs = getattr(adata, "obs", pd.DataFrame())
    obsm = getattr(adata, "obsm", {}) or {}
    var = getattr(adata, "var", pd.DataFrame())

    p.x_state, p.x_evidence = detect_x_state(adata)
    p.counts_layer = find_counts_layer(adata, counts_layer_hint)

    for canonical, aliases in _QC_ALIASES.items():
        for name in aliases:
            if name in obs.columns and pd.api.types.is_numeric_dtype(obs[name]):
                p.qc_columns[canonical] = name
                break

    for key in _PCA_KEYS:
        if key in obsm:
            p.pca_key = key
            break
    for key in _UMAP_KEYS:
        if key in obsm:
            p.umap_key = key
            break

    for name in cfg.cluster_col_candidates:
        if name in obs.columns and obs[name].nunique() >= 2:
            p.cluster_column = name
            break

    for name in _HVG_COLS:
        if name in var.columns:
            p.hvg_column = name
            break

    for name in cfg.doublet_score_col_candidates:
        if name in obs.columns:
            p.doublet_score_column = name
            break
    for name in cfg.doublet_call_col_candidates:
        if name in obs.columns:
            p.doublet_call_column = name
            break

    # ---- notes -----------------------------------------------------------
    if p.x_state == RAW_COUNTS:
        p.notes.append("X holds raw counts, so the full pipeline applies.")
    elif p.x_state in (LOG1P, NORMALISED):
        if p.counts_layer:
            p.notes.append(
                f"X holds {p.x_state}, and raw counts were found in "
                f"layers['{p.counts_layer}']. Counts-based steps (QC metrics, "
                f"highly-variable-gene selection) will read that layer; nothing "
                f"is normalised twice."
            )
        else:
            p.notes.append(
                f"X holds {p.x_state} and NO raw-counts layer was found. "
                f"Normalisation and log1p will be SKIPPED to avoid transforming "
                f"the data twice, and QC count thresholds are being applied to "
                f"{p.x_state}, so they are not UMI counts. Supply raw counts, or "
                f"point --counts-layer at them, for QC to mean what it usually "
                f"means."
            )
    else:
        p.notes.append(
            "Could not determine whether X holds raw counts or transformed "
            "values. Proceeding as if raw; check the QC distributions carefully."
        )

    if p.can_skip_embedding:
        p.notes.append(
            f"A complete embedding is already present (obsm['{p.pca_key}'], "
            f"obsm['{p.umap_key}'], obs['{p.cluster_column}']) and will be REUSED "
            f"rather than recomputed. This is both faster and avoids producing a "
            f"second, different clustering of the same cells. Pass "
            f"--force-recompute to recompute it anyway."
        )
    elif p.pca_key or p.umap_key or p.cluster_column:
        present = [
            x for x in (
                f"obsm['{p.pca_key}']" if p.pca_key else None,
                f"obsm['{p.umap_key}']" if p.umap_key else None,
                f"obs['{p.cluster_column}']" if p.cluster_column else None,
            ) if x
        ]
        p.notes.append(
            f"Partial prior results found ({', '.join(present)}) but not a "
            f"complete embedding, so the embedding is being recomputed. The "
            f"existing columns are left untouched."
        )

    if p.has_qc_metrics:
        p.notes.append(
            f"QC metrics already in obs and will be reused: "
            f"{', '.join(f'{k} <- {v}' for k, v in p.qc_columns.items())}."
        )

    if p.doublet_call_column:
        p.notes.append(
            f"Existing doublet calls in obs['{p.doublet_call_column}'] will be "
            f"used instead of recomputing them. As always they are annotated, "
            f"never removed."
        )
    return p


def apply_force_recompute(p: PriorAnalysis) -> PriorAnalysis:
    """Discard detected prior results, keeping only the X-state finding.

    The X state is never discarded: whether the matrix is already
    log-transformed is a fact about the data, not a cached result, and ignoring
    it would double-transform.
    """
    forced = PriorAnalysis(
        x_state=p.x_state,
        x_evidence=list(p.x_evidence),
        counts_layer=p.counts_layer,
    )
    forced.notes.append(
        "--force-recompute: ignoring any pre-existing QC metrics, embedding, "
        "clustering and doublet calls, and computing everything from scratch. "
        "The detected state of X is still respected, so the data is not "
        "transformed twice."
    )
    return forced
