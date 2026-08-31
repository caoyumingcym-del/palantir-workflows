"""
Stage orchestration.

Replaces ``run_pipeline.py``, which executed the analysis by parsing the
notebook as JSON, substring-matching for a cell containing
``"MIN_GENES = "``, injecting a synthetic cell after it, truncating the cell
list, writing the result to a temp directory and shelling out to
``jupyter nbconvert --execute``.  That approach had no way to fail cleanly: a
renamed variable, a commented-out occurrence, a Windows path ending in a
backslash, or a cell whose source lacked a trailing newline all produced either
a silent no-op or a syntax error inside a generated notebook that was then
deleted.

Here the stages are Python function calls.  Parameters are passed as arguments.
There is nothing to inject.
"""
from __future__ import annotations

import gc
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import gex as GEX
from . import guide as GUIDE
from . import hto as HTO
from . import hto_dragen as HTO_DRAGEN
from . import perturb as PERT
from . import plotting as P
from . import pseudobulk as PB
from . import qc as QC
from . import report as REPORT
from . import sanity as SANITY
from . import seqmetrics as SEQ
from . import text as T
from .artifacts import Registry
from .config import PipelineConfig, THRESHOLD_KEYS
from .manifest import Manifest, ManifestError, read_manifest
from .whitelists import (
    cross_check_families, load_guide_whitelist, load_hashtag_whitelist,
)
from .modalities import Modality, SplitResult, resolve_column, split_modalities
from . import provenance as PROV
from .version import __version__


class PipelineError(Exception):
    """A stage could not proceed. The CLI turns this into a clean exit code."""


@dataclass
class RunResult:
    registry: Registry
    report_path: Path | None
    thresholds: Any
    n_cells_before: int
    n_cells_after: int
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


def current_rss_gb() -> float | None:
    """Resident memory of this process, in GB, or None if unavailable."""
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        import os

        return pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return None


def _log(cfg: PipelineConfig, message: str, mem: bool = False) -> None:
    """Print a progress line, flushed immediately.

    Flushing matters more than it looks: when stdout is a pipe (``| tee``),
    Python block-buffers it, so anything not yet flushed is lost if the process
    is SIGKILLed by the out-of-memory killer. That is how a crash produces a
    completely empty log.
    """
    if not cfg.verbose:
        return
    prefix = "[perturbseq]"
    if mem:
        rss = current_rss_gb()
        if rss is not None:
            prefix = f"[perturbseq {rss:6.2f}GB]"
    print(f"{prefix} {message}", flush=True)


_STEP_CLOCK: dict[str, float] = {}


def _step(cfg: PipelineConfig, label: str) -> None:
    """Mark the start of a potentially expensive operation, with memory.

    Every heavy step announces itself *before* running, so if the process is
    killed the last line in the log names the operation that was in flight.

    It also reports how long the PREVIOUS step took. Announcing before running
    is what makes a crash diagnosable, but it means no line can carry its own
    duration -- so up to v1.2.5 the log said which step was slow only in the
    sense that you could subtract wall-clock timestamps that were not there
    either. "Which step is longest?" was unanswerable from a log, including for
    me. Now each line closes out the one before it.
    """
    now = time.time()
    prev_label = _STEP_CLOCK.get("label")
    prev_start = _STEP_CLOCK.get("start")
    if prev_label is not None and prev_start is not None:
        elapsed = now - float(prev_start)
        # Only worth the noise once a step is slow enough to care about.
        if elapsed >= 1.0:
            _log(cfg, f"     ({_fmt_duration(elapsed)} for: {prev_label})")
    _STEP_CLOCK["label"] = label
    _STEP_CLOCK["start"] = now
    _log(cfg, f"  -> {label}", mem=True)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _is_unreadable_uns_error(exc: Exception) -> bool:
    """Is this the "anndata can't decode a value in uns" version-skew error?

    ``IORegistryError`` for ``IOSpec(encoding_type='null', ...)`` (or similar)
    means the file was WRITTEN by a newer anndata than the one installed here,
    and it wrote a value -- almost always a ``None`` somewhere in ``uns``, e.g.
    an upstream tool's (Solo doublet detection, scVI, ...) per-group metadata
    dict -- using an on-disk encoding this installation predates support for.
    Detected by message content because ``IORegistryError`` is only importable
    from newer anndata versions and this has to work against the OLD one that
    is failing.
    """
    msg = str(exc)
    return "No read method registered for IOSpec" in msg or "IORegistryError" in type(exc).__name__


def _load_h5ad_skipping_uns(path: Path, ad) -> tuple[Any, int, list[str]]:
    """Reconstruct an AnnData, tolerating unreadable values WITHIN ``uns``.

    Originally this dropped ``uns`` wholesale on the theory that the pipeline
    never reads it from a real input file. That is false: ``modalities.py``
    reads ``uns['gRNA_features']`` / ``uns['HTO_features']`` (and their
    aliases) to name guide and hashtag columns pulled from ``obsm`` -- exactly
    the resolution path MDL-1856 takes, since it has no ``var`` feature-type
    column. Dropping all of ``uns`` to dodge one bad key (typically a ``None``
    from Solo doublet detection, encoded in a way this anndata predates
    support for) silently took the guide/hashtag name vectors down with it:
    every guide fell back to placeholder names (``guide_0``, ``guide_1``, ...),
    which then match nothing in the gRNA whitelist and nothing in
    ``GuideConfig.guide_id_regexes``, corrupting NTC detection and
    perturbation quantification without raising anything.

    Fixed to read ``uns`` KEY BY KEY: each top-level entry is attempted
    through anndata's own ``read_elem``, and only the individual keys that
    actually fail to decode are dropped. Every other top-level group is
    unchanged from before -- this does not reimplement anndata's parsing, it
    just narrows the blast radius of one bad value from "all of uns" to "the
    one key that is actually unreadable".
    """
    import h5py
    try:
        from anndata.experimental import read_elem
    except ImportError:  # older anndata layout
        from anndata._io.specs.registry import read_elem  # type: ignore

    kwargs: dict[str, Any] = {}
    skipped = 0
    uns_skipped_keys: list[str] = []
    with h5py.File(path, "r") as f:
        for key in ("X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers"):
            if key not in f:
                continue
            try:
                kwargs[key] = read_elem(f[key])
            except Exception:
                skipped += 1
        if "raw" in f:
            try:
                kwargs["raw"] = read_elem(f["raw"])
            except Exception:
                skipped += 1
        if "uns" in f:
            uns: dict[str, Any] = {}
            for uns_key in f["uns"].keys():
                try:
                    uns[uns_key] = read_elem(f["uns"][uns_key])
                except Exception:
                    uns_skipped_keys.append(uns_key)
                    skipped += 1
            kwargs["uns"] = uns
    return ad.AnnData(**kwargs), skipped, uns_skipped_keys


def _load_h5ad(path: Path) -> tuple[Any, list[str]]:
    try:
        import anndata as ad
    except ImportError as exc:
        raise PipelineError(
            "anndata is required to read .h5ad files but is not installed. "
            "Install the analysis dependencies with:\n"
            "    pip install anndata scanpy scikit-learn scipy\n"
            f"(underlying error: {exc})"
        ) from exc
    if not path.exists():
        raise PipelineError(
            f"h5ad file not found: {path}\n"
            f"This path came from the manifest's 'h5ad_path' column and is "
            f"resolved relative to the manifest's own directory."
        )

    notes: list[str] = []
    try:
        return ad.read_h5ad(path), notes
    except Exception as exc:
        if not _is_unreadable_uns_error(exc):
            raise
        try:
            adata, n_skipped, uns_skipped_keys = _load_h5ad_skipping_uns(path, ad)
        except Exception:
            # The fallback itself failed -- surface the ORIGINAL error, since
            # it is the more informative one and the fallback adds no signal.
            raise PipelineError(
                f"Could not read {path}: anndata raised "
                f"{type(exc).__name__} while decoding a value in 'uns' "
                f"({exc}), and the uns-skipping fallback also failed. This is "
                f"a version-skew issue: the file was written by a newer "
                f"anndata than the one installed here. Upgrade anndata "
                f"(`pip install -U anndata`) in this environment."
            ) from exc
        if uns_skipped_keys:
            keys_note = (
                f"uns key(s) {uns_skipped_keys} could not be decoded and were "
                f"dropped; every other uns key was kept. If guide or hashtag "
                f"feature names normally live under one of these keys "
                f"(GuideConfig/ModalityConfig's *_feature_uns_keys), naming "
                f"will fall back to placeholders for that modality -- check "
                f"the 'Modality detection' notes below."
            )
        else:
            keys_note = "every uns key was read successfully despite the error below."
        notes.append(
            f"A value in this h5ad's 'uns' could not be decoded by the "
            f"installed anndata ({type(exc).__name__}: {exc}) -- the file was "
            f"written by a newer anndata than the one installed here, and that "
            f"value (commonly a None from an upstream tool such as Solo "
            f"doublet detection's per-celltype threshold dict) uses an "
            f"on-disk encoding this installation predates support for. X, obs, "
            f"var, obsm, varm, obsp, varp, layers and raw were read normally "
            f"through anndata's own reader. 'uns' was read KEY BY KEY rather "
            f"than skipped wholesale, because this pipeline DOES read "
            f"'uns' for guide/hashtag feature names (modalities.py) even "
            f"though it never reads it for anything else: {keys_note} "
            f"({n_skipped} top-level item(s) not attempted in total). The "
            f"durable fix is upgrading anndata in this environment "
            f"(`pip install -U anndata`)."
        )
        return adata, notes


def resolve_group_columns(
    obs: pd.DataFrame,
    manifest: Manifest | None,
    cfg: PipelineConfig,
) -> dict[str, pd.Series]:
    """Which columns to use as comparison axes, in priority order.

    Explicit config wins; otherwise the manifest's varying condition columns;
    and if the experiment has only one condition (as in a single-sample,
    multi-lane run) it falls back to sample and lane, which is still the useful
    comparison. The original required at least one condition column to vary and
    otherwise produced no per-group panels at all.
    """
    out: dict[str, pd.Series] = {}

    # Nominated columns come FIRST and are never displaced; autodetected ones
    # fill whatever slots remain. "Prioritise" rather than "restrict" -- naming
    # the comparison you care about should not throw away the others, and it
    # must be impossible for the named one to be the one that gets dropped.
    # (v1.2.1 shipped with the restrictive reading, which meant declaring
    # `condition = fixation` silently discarded gRNA_method, acoh and buffer.)
    nominated: list[str] = list(cfg.condition_columns)
    if not nominated and manifest is not None:
        try:
            nominated = manifest.declared_condition_columns()
        except Exception:
            nominated = []
    declared = bool(nominated)

    auto: list[str] = []
    if manifest is not None:
        try:
            auto = manifest.condition_columns(cfg.max_compare_levels)
        except Exception:
            auto = []
    candidates = nominated + [c for c in auto if c not in nominated]

    for col in candidates:
        if col in obs.columns and obs[col].astype(str).nunique() >= 2:
            out[col] = obs[col].astype(str)
        elif col in nominated:
            # Nominated explicitly but unusable -- say so rather than drop it.
            _log(
                cfg,
                f"  [warn] condition column {col!r} was nominated in the "
                f"manifest but has fewer than 2 distinct values in the data; "
                f"no panels will be produced for it",
            )

    if not out:
        for fallback in ("sample", "prefix", "batch", "library"):
            if fallback in obs.columns and obs[fallback].astype(str).nunique() >= 2:
                out[fallback] = obs[fallback].astype(str)
                break

    # Cap the number of axes: each one multiplies the figure count, and past
    # three the report becomes unreadable rather than more informative.
    #
    # When the columns were declared, keep the FIRST three in the order given
    # -- the manifest's order is the author's priority. Only fall back to the
    # fewest-levels heuristic when the axes were guessed, and say so, because
    # that rule can otherwise silently discard the comparison the experiment
    # was designed around.
    cap = max(1, int(getattr(cfg, "max_condition_axes", 3)))
    if len(out) > cap:
        if declared:
            keep = list(out)[:cap]
            dropped = list(out)[cap:]
        else:
            sizes = {k: v.nunique() for k, v in out.items()}
            keep = sorted(sizes, key=lambda k: sizes[k])[:cap]
            dropped = [k for k in out if k not in keep]
        _log(
            cfg,
            f"  [warn] {len(out)} condition columns available but the cap is "
            f"{cap}; using {keep} and dropping {dropped}. Name the ones you "
            f"want in a manifest column called 'condition_columns' (or "
            f"'condition'), pass --conditions, or raise max_condition_axes.",
        )
        out = {k: out[k] for k in keep}
    return out


def attach_manifest_metadata(
    obs: pd.DataFrame, manifest: Manifest, cfg: PipelineConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Join per-sample manifest metadata onto obs, reporting any mismatch."""
    warnings: list[str] = []
    sample_col = resolve_column(obs, cfg.modality.sample_col_candidates)
    if sample_col is None:
        warnings.append(
            f"No sample column found in obs (looked for "
            f"{list(cfg.modality.sample_col_candidates)}); manifest metadata "
            f"could not be attached, so per-condition panels will fall back to "
            f"whatever grouping is available."
        )
        return obs, warnings

    meta = manifest.sample_metadata_frame()
    obs_samples = set(obs[sample_col].astype(str).unique())
    man_samples = set(meta.index.astype(str))

    missing_in_manifest = sorted(obs_samples - man_samples)
    missing_in_data = sorted(man_samples - obs_samples)
    if missing_in_manifest:
        warnings.append(
            f"{len(missing_in_manifest)} sample(s) in the data are absent from the "
            f"manifest ({', '.join(missing_in_manifest[:5])}); their metadata "
            f"columns will be blank. Add them to the manifest so they are included "
            f"in per-condition comparisons."
        )
    if missing_in_data:
        warnings.append(
            f"{len(missing_in_data)} sample(s) in the manifest are absent from the "
            f"data ({', '.join(missing_in_data[:5])}). This is expected if the "
            f"h5ad is a subset; otherwise a sample failed upstream."
        )

    joined = obs.copy()
    keyed = joined[sample_col].astype(str)
    for col in meta.columns:
        if col in joined.columns:
            continue
        joined[col] = keyed.map(meta[col]).to_numpy()
    return joined, warnings


def run(cfg: PipelineConfig, manifest: Manifest) -> RunResult:
    """Execute the pipeline end to end, reading the h5ad named by the manifest."""
    cfg.output_path = cfg.output_path or manifest.output_path
    cfg.h5ad_path = cfg.h5ad_path or manifest.h5ad_path

    _log(cfg, f"reading {cfg.h5ad_path}")
    step = time.time()
    adata, load_notes = _load_h5ad(Path(cfg.h5ad_path))
    load_seconds = time.time() - step
    _log(cfg, f"loaded {adata.n_obs:,} cells x {adata.n_vars:,} features", mem=True)
    for note in load_notes:
        _log(cfg, f"NOTE: {note}")

    return run_with_adata(cfg, manifest, adata, load_seconds=load_seconds,
                          load_notes=load_notes)


def run_with_adata(
    cfg: PipelineConfig,
    manifest: Manifest,
    adata: Any,
    load_seconds: float = 0.0,
    load_notes: list[str] | None = None,
) -> RunResult:
    """Run every stage against an already-loaded AnnData.

    Separated from ``run`` so that callers who already have the object in memory
    (a notebook session, a batch driver, the test suite) do not have to write it
    to disk and read it back. It also keeps the only h5ad-format dependency in
    one function.
    """
    t0 = time.time() - load_seconds
    timings: dict[str, float] = {"load_h5ad": load_seconds}
    warnings: list[str] = list(manifest.warnings)

    cfg.output_path = cfg.output_path or manifest.output_path
    cfg.ensure_dirs()
    P.apply_style(cfg.figures)
    reg = Registry(cfg.analysis_dir)

    for i, note in enumerate(load_notes or []):
        reg.note("appendix", f"h5ad_load_{i}", "Reading the input .h5ad", note,
                 level="warn", order=2 + i)

    for i, w in enumerate(manifest.warnings):
        reg.note("appendix", f"manifest_warn_{i}", "Manifest warning", w,
                 level="warn", order=10 + i)

    _report_matrix_size(adata, cfg, reg, warnings)

    # ---- optional subsample ------------------------------------------------
    if cfg.subsample_cells and adata.n_obs > cfg.subsample_cells:
        rng = np.random.default_rng(cfg.subsample_random_state)
        keep_idx = np.sort(rng.choice(adata.n_obs, cfg.subsample_cells,
                                      replace=False))
        adata = adata[keep_idx].copy()
        gc.collect()
        msg = (
            f"Randomly subsampled to {cfg.subsample_cells:,} cells "
            f"(--subsample-cells). Every rate and distribution below is still "
            f"representative, but per-perturbation and per-hashtag CELL COUNTS "
            f"are reduced proportionally, so anything limited by cell number -- "
            f"weak perturbations especially -- is under-powered relative to the "
            f"full experiment."
        )
        warnings.append(msg)
        reg.note("summary", "subsampled", "This run used a subsample", msg,
                 level="warn", order=0)
        _log(cfg, f"subsampled to {adata.n_obs:,} cells", mem=True)

    # ---- what has already been done to this object? -----------------------
    _step(cfg, "detecting prior analysis in the input")
    prior = PROV.detect(adata, cfg.modality, cfg.counts_layer)
    if cfg.force_recompute:
        prior = PROV.apply_force_recompute(prior)
    _log(cfg, f"input state: {prior.summary()}")
    for i, note in enumerate(prior.notes):
        level = "warn" if ("NO raw-counts layer" in note
                           or "Could not determine" in note) else "info"
        reg.note("appendix", f"prior_{i}", "Input object", note, level=level,
                 order=4 + i)
    reg.note(
        "appendix", "prior_evidence", "How the input state was determined",
        "; ".join(prior.x_evidence), order=3,
    )

    # ----------------------------------------------------------- modalities
    step = time.time()
    # Guide and hashtag matrices must come from RAW COUNTS. If X has already
    # been transformed, read them from the counts layer instead.
    counts_matrix = None
    if not prior.x_is_raw_counts and prior.counts_layer:
        try:
            counts_matrix = adata.layers[prior.counts_layer]
            _log(cfg, f"guide/hashtag counts will be read from "
                      f"layers['{prior.counts_layer}']")
        except Exception:
            counts_matrix = None
    if not prior.x_is_raw_counts and counts_matrix is None:
        warnings.append(
            "X is not raw counts and no counts layer was found, so guide and "
            "hashtag depth thresholds (--guide-min-reads, --hto-quantile) are "
            "being applied to transformed values. Assignment and singlet rates "
            "below are therefore not trustworthy; supply raw counts or point "
            "--counts-layer at them."
        )
        reg.note(
            "appendix", "guide_counts_transformed",
            "Guide/hashtag counts are not raw",
            warnings[-1], level="poor", order=5,
        )
    split: SplitResult = split_modalities(
        adata, cfg.modality, cfg.guide.guide_id_regexes, counts=counts_matrix
    )
    timings["split_modalities"] = time.time() - step
    for i, note in enumerate(split.notes):
        level = "warn" if note.startswith("WARNING") else "info"
        reg.note("appendix", f"modality_{i}", "Modality detection", note,
                 level=level, order=30 + i)

    # ---- hashtag fallback: DRAGEN cellhashing.tsv -------------------------
    #
    # All four in-h5ad lookups (var feature-types, obsm, uns, flattened obs
    # columns -- modalities.split_modalities) failed. Before giving up, try
    # the DRAGEN per-run output named in the manifest: some h5ads are built
    # from a GEX-only matrix and never carried the hashtag counts through at
    # all, in which case the only surviving copy is
    # ``<dragen_path>/<prefix>.scRNA.cellhashing.tsv``.
    #
    # Skipped outright when the manifest explicitly says this experiment did
    # not use hashtags (``declares_hto() is False``) -- there is nothing to
    # recover, and searching would just produce a confusing "found unexpected
    # hashtag data" note for an experiment that never had any.
    if not split.hto.present and manifest.declares_hto() is not False:
        dragen_runs_for_hto = manifest.dragen_runs(root_override=cfg.dragen_root)
        if dragen_runs_for_hto:
            hto_target_obs_names = [str(x) for x in split.gex.obs.index]
            # A sample with more than one dragen run needs to know whether
            # those runs are multiple library preps of the SAME cells (sum
            # hashtag counts across them) or independent pools (keep only
            # the strongest-matching run) -- not guessable from the files
            # themselves; see manifest.dragen_runs_share_cells and
            # hto_dragen.py's module docstring. Resolved per sample here so
            # a mixed manifest (some samples single-run, some multi-run
            # shared, some multi-run independent) is handled correctly
            # rather than applying one global assumption.
            hto_sample_col = resolve_column(
                split.gex.obs, cfg.modality.sample_col_candidates
            )
            hto_sample_of_cell = (
                [str(x) for x in split.gex.obs[hto_sample_col]]
                if hto_sample_col else None
            )
            hto_share_cells_by_sample = {
                s: manifest.dragen_runs_share_cells(s) for s in manifest.samples
            }
            recovered, dragen_notes = HTO_DRAGEN.build_hto_modality_multi_sample(
                dragen_runs_for_hto, hto_target_obs_names, hto_sample_of_cell,
                hto_share_cells_by_sample, cfg.modality,
            )
            for i, note in enumerate(dragen_notes):
                level = "warn" if recovered is None else "info"
                reg.note(
                    "appendix", f"hto_dragen_{i}",
                    "Hashtag recovery from DRAGEN output", note,
                    level=level, order=35 + i,
                )
            if recovered is not None and recovered.present:
                split.hto = recovered
                _log(
                    cfg,
                    f"recovered {recovered.n_features} hashtags for "
                    f"{recovered.n_cells:,} cells from DRAGEN cellhashing "
                    f"output (not found in the h5ad itself)",
                )
        else:
            reg.note(
                "appendix", "hto_dragen_no_runs",
                "Hashtag recovery from DRAGEN output",
                "No hashtag matrix was found in the h5ad, and the manifest has "
                "no usable 'prefix'/'dragen_path' columns to look for a "
                "DRAGEN cellhashing.tsv fallback.",
                level="warn", order=35,
            )

    # split.gex is an independent copy, so the original is now dead weight --
    # a second full matrix held for no reason. On a 300k-cell experiment that is
    # several GB, and the pipeline was carrying it all the way through QC.
    adata = None
    gc.collect()
    _log(
        cfg,
        f"modalities: {split.gex.n_vars:,} genes, {split.guide.n_features} guides, "
        f"{split.hto.n_features} hashtags",
        mem=True,
    )

    # ------------------------------------------------- input plausibility
    # Runs on the GEX matrix as the pipeline will actually read it -- after
    # guides and hashtags have been removed -- so it tests the thing that
    # feeds HVG selection, clustering and every gene-level statistic.
    if cfg.check_input_matrix:
        _step(cfg, "checking input matrix plausibility")
        chk = SANITY.check_expression_matrix(
            split.gex.X,
            [str(v) for v in split.gex.var.index],
            var=split.gex.var,
            min_detection=cfg.housekeeping_min_detection,
        )
        for i, note in enumerate(chk.notes):
            reg.note("appendix", f"input_check_{i}", "Input matrix check",
                     note, order=10 + i)
        if not chk.ok:
            reg.note(
                "appendix", "input_check_failed",
                "Input matrix failed plausibility checks",
                " ".join(chk.failures), level="poor", order=1,
            )
            raise PipelineError(SANITY.format_failure(chk))
        _log(cfg, f"  input matrix: {chk.verdict} "
                  f"({100 * chk.median_detection:.1f}% median housekeeping "
                  f"detection over {chk.checked} probes)")
    else:
        warnings.append(
            "Input matrix plausibility checks were skipped "
            "(--skip-input-check). Gene-level results are unverified."
        )
        reg.note("appendix", "input_check_skipped",
                 "Input matrix checks skipped", warnings[-1],
                 level="warn", order=1)

    # ------------------------------------------------------------ whitelists
    # Loaded after the split so the hashtag whitelist can be validated against
    # the hashtag names actually present -- a typo like "hash.A" for
    # "prot:hash.A" should stop the run with the real names listed, not
    # silently classify every cell as Ambiguous.
    guide_wl = hto_wl = None
    wl_notes: list[str] = []
    gp = manifest.grna_whitelist_path
    hp = manifest.hashtag_whitelist_path
    if gp is not None:
        guide_wl = load_guide_whitelist(gp)
        wl_notes.extend(guide_wl.warnings)
        missing, extra = guide_wl.coverage(split.guide.names)
        _log(cfg, f"gRNA whitelist: {len(guide_wl.df)} guides, "
                  f"{len(guide_wl.families)} families "
                  f"({len(missing)} in data but unlisted, {len(extra)} listed "
                  f"but absent from data)")
        declared_desc = (
            ", ".join(guide_wl.declared_columns())
            or "none -- every annotation derived from the guide IDs"
        )
        reg.note(
            "guides", "whitelist", "gRNA whitelist",
            (
                f"Families and target annotation read from {gp.name}: "
                f"{len(guide_wl.df)} guides across families "
                f"{', '.join(guide_wl.families)}. Columns declared explicitly: "
                f"{declared_desc}."
            ),
            order=4,
        )
    if hp is not None:
        hto_wl = load_hashtag_whitelist(
            hp, known_hashtags=split.hto.names if split.hto.present else None
        )
        wl_notes.extend(hto_wl.warnings)
        _log(cfg, f"hashtag whitelist: {len(hto_wl.df)} combinations, "
                  f"{len(hto_wl.demux_ids)} samples")
    wl_notes.extend(cross_check_families(guide_wl, hto_wl))
    warnings.extend(wl_notes)
    for i, w in enumerate(wl_notes):
        reg.note("appendix", f"whitelist_{i}", "Whitelist validation", w,
                 level="warn", order=20 + i)

    # -------------------------------------------------------------- metadata
    obs, meta_warnings = attach_manifest_metadata(split.gex.obs, manifest, cfg)
    split.gex.obs = obs
    warnings.extend(meta_warnings)
    for i, w in enumerate(meta_warnings):
        reg.note("appendix", f"meta_warn_{i}", "Sample metadata", w, level="warn",
                 order=50 + i)

    sample_col = resolve_column(obs, cfg.modality.sample_col_candidates)
    group_columns = resolve_group_columns(obs, manifest, cfg)
    _log(
        cfg,
        "comparison axes: "
        + (", ".join(f"{k} ({v.nunique()} levels)" for k, v in group_columns.items())
           or "none"),
    )
    if not group_columns:
        reg.note(
            "appendix", "no_axes", "No comparison axes",
            "No manifest column varies between samples and no sample/lane column "
            "was found in the data, so per-condition panels are omitted. This is "
            "expected for a single-condition experiment.",
            order=60,
        )

    # ------------------------------------------------------------- seq QC
    step = time.time()
    runs = manifest.dragen_runs(root_override=cfg.dragen_root)
    sm = None            # bound unconditionally: the downsampling check below
                         # runs with or without upstream metrics
    if runs:
        sm = SEQ.load_sequencing_metrics(runs)
        _emit_seq_qc(sm, cfg, reg)
    else:
        reg.skipped(
            "seq_qc", "all", "Sequencing QC",
            "The manifest has no usable 'dragen_path'/'prefix' columns, so no "
            "upstream metrics files could be located. Transcriptome, guide and "
            "hashtag analysis are unaffected.",
        )
    timings["seq_qc"] = time.time() - step

    # ------------------------------------------------------------- cell QC
    _step(cfg, "stage: per-cell QC")
    step = time.time()
    counts_for_qc = None
    if not prior.x_is_raw_counts and prior.counts_layer:
        try:
            counts_for_qc = split.gex.layers[prior.counts_layer]
        except Exception:
            counts_for_qc = None
    qc_table, filt = QC.run_qc_stage(
        split.gex, split.guide, split.hto, cfg, reg, group_columns,
        cell_input=manifest.total_cell_input(),
        prior=prior, counts=counts_for_qc,
    )
    timings["cell_qc"] = time.time() - step
    _log(cfg, f"QC: kept {filt.n_after:,} of {filt.n_before:,} cells "
              f"({filt.frac_retained * 100:.1f}%)", mem=True)

    # Depth measured from this h5ad, against the pre-downsampling DRAGEN
    # numbers. Needs the QC table, so it runs here rather than with the rest of
    # the sequencing-QC section.
    try:
        _emit_downsampling_check(
            sm, qc_table, split.gex.obs, split.guide, split.hto,
            manifest, cfg, reg,
        )
    except Exception as exc:      # diagnostic panel: never fail the run for it
        _log(cfg, f"  [warn] downsampling check skipped: {exc}")

    if filt.n_after < 50:
        raise PipelineError(
            f"Only {filt.n_after} cells passed QC, which is too few to analyse. "
            f"Thresholds used: {filt.thresholds.as_dict()} "
            f"(sources: {filt.thresholds.source}). Inspect "
            f"{cfg.fig_dir / 'qc_hexbin_prefilter.png'} and set thresholds "
            f"explicitly via the manifest or --min-genes/--max-mito/etc."
        )

    # Persist the thresholds actually used, so the record lives with the
    # experiment rather than only in a shell command.
    try:
        backup = manifest.write_thresholds(filt.thresholds)
        if backup:
            _log(cfg, f"wrote thresholds into manifest (backup: {backup.name})")
    except (ManifestError, OSError) as exc:
        warnings.append(f"Could not write thresholds back to the manifest: {exc}")

    if cfg.explore_only:
        _save_threshold_state(cfg, filt.thresholds)
        _log(cfg, "explore run: stopping after QC")
        reg.note(
            "appendix", "explore_mode", "This is an explore run",
            (
                "The run stopped after the QC stage, before anything was filtered, "
                "clustered or quantified. The five threshold columns in the manifest "
                "have been filled with the values derived from these distributions. "
                "Review the panels above, adjust those numbers if you disagree, then "
                "re-run the same command to produce the full report."
            ),
            order=5,
        )
        return _finish(cfg, reg, filt, warnings, timings, t0, split, None)

    _warn_if_thresholds_unreviewed(cfg, filt.thresholds, reg, warnings)

    # Apply the mask. Modalities are subset alongside, so every table shares
    # one cell population -- the alternative is the class of bug where a guide
    # table has more rows than the QC table.
    keep = filt.mask
    _step(cfg, f"applying QC mask -> {int(keep.sum()):,} cells")
    gex_f = split.gex[keep].copy()

    # Carry the QC metrics onto obs so the embedding stage can regress them out.
    #
    # EmbeddingConfig.regress_out asks for total_counts, pct_counts_mt, S_score
    # and G2M_score, but the QC metrics live in their own table and were never
    # joined onto obs -- so _covariate_matrix only ever found the two
    # cell-cycle scores that scanpy adds, and silently regressed out two of the
    # four. The report's "what was done" text claimed all four. Depth is
    # arguably the most important of them.
    _qc_kept = qc_table.loc[keep] if hasattr(qc_table, "loc") else None
    if _qc_kept is not None:
        for _c in ("total_counts", "n_genes_by_counts", "pct_counts_mt",
                   "pct_counts_ribo", "log10_genes_per_umi"):
            if _c in _qc_kept.columns and _c not in gex_f.obs.columns:
                gex_f.obs[_c] = _qc_kept[_c].to_numpy()

    guide_f = split.guide.subset_cells(keep) if split.guide.present else split.guide
    hto_f = split.hto.subset_cells(keep) if split.hto.present else split.hto
    # The unfiltered gene-expression matrix is superseded by gex_f; release it
    # before the transcriptome stage, which is where the headroom is needed.
    # guide/hto are left alone -- they are small, and guide_f holds its own
    # arrays already.
    split.gex = None
    gc.collect()
    _log(cfg, "released the unfiltered expression matrix", mem=True)
    group_f = {k: v[keep].reset_index(drop=True) for k, v in group_columns.items()}
    for k in group_f:
        group_f[k].index = pd.Index([str(x) for x in gex_f.obs.index])

    # --------------------------------------------------------- transcriptome
    _step(cfg, "stage: transcriptome (normalise, HVG, embed, cluster)")
    step = time.time()
    # The declared condition axes are passed in so the embedding stage can
    # refuse to batch-correct on a column that IS the comparison.
    emb = GEX.run_transcriptome_stage(
        gex_f, cfg, reg, group_f, sample_col,
        step=lambda label: _step(cfg, label),
        prior=prior,
        condition_columns=list(group_columns.keys()),
    )
    timings["transcriptome"] = time.time() - step
    _log(cfg, f"transcriptome: {emb.obs['cluster'].nunique()} clusters "
              f"({emb.backend})", mem=True)

    # ---------------------------------------------------------------- guides
    _step(cfg, "stage: guide assignment")
    step = time.time()
    _sample_for_guides = None
    _sc = resolve_column(gex_f.obs, cfg.modality.sample_col_candidates)
    if _sc is not None:
        _sample_for_guides = gex_f.obs[_sc].astype(str)
        _sample_for_guides.index = pd.Index([str(x) for x in gex_f.obs.index])
    ga = GUIDE.run_guide_stage(guide_f, cfg, reg, group_f, whitelist=guide_wl,
                               sample=_sample_for_guides)
    timings["guides"] = time.time() - step
    if ga is not None:
        _log(cfg, f"guides: {ga.frac_assigned * 100:.1f}% of cells assigned")

    # -------------------------------------------------------------- hashtags
    # Hashtags run BEFORE perturbation, which is a change from v1.1.0. The
    # guide-family vs hashtag-family cross-check flags doublets and index hops,
    # and the perturbation stage excludes those cells -- so the flag has to
    # exist before the comparisons are computed, not after.
    _step(cfg, "stage: hashtags")
    step = time.time()
    calls = HTO.run_hto_stage(
        hto_f, cfg, reg, group_f, manifest_declares_hto=manifest.declares_hto(),
        whitelist=hto_wl,
    )
    timings["hashtags"] = time.time() - step
    if calls is not None:
        key = "pct_resolved" if calls.design_declared else "pct_singlet"
        _log(cfg, f"hashtags: {calls.rates.get(key, float('nan')):.1f}% "
                  f"{'resolved' if calls.design_declared else 'singlets'}")

    # Flags `family_conflict` on ga.per_cell in place, for the stage below.
    if ga is not None and calls is not None:
        HTO.family_crosscheck(ga.per_cell, calls, cfg, reg)

    # --------------------------------------------------------- perturbation
    _step(cfg, "stage: perturbation effects")
    step = time.time()
    pert = PERT.run_perturbation_stage(
        emb.X_log, emb.var_names, emb.pca,
        ga.per_cell if ga else None, ga.mapping if ga else None,
        cfg, reg,
        batch_corrected=emb.batch_corrected,
        group_columns=group_f,
    )
    timings["perturbation"] = time.time() - step

    # ------------------------------------------------------- comparability
    # Asked before any per-condition difference is believed: are the conditions
    # comparable at all? A guide-composition difference between conditions is
    # almost never biology, and it biases every per-guide number in the
    # affected arm.
    _step(cfg, "stage: condition comparability (pseudobulk)")
    step = time.time()
    try:
        PB.run_comparability_stage(
            emb.X_log, emb.var_names, group_f,
            ga.per_cell if ga else None, cfg, reg,
        )
    except Exception as exc:
        reg.skipped("comparability", "all", "Pseudobulk comparability",
                    f"Could not be computed ({exc}).")
    timings["comparability"] = time.time() - step

    # ----------------------------------------------------------- crosschecks
    HTO.run_crosscheck_stage(
        ga.per_cell if ga else None, calls, qc_table, cfg, reg
    )
    if calls is not None:
        _cluster_vs_hto(emb, calls, cfg, reg)

    return _finish(cfg, reg, filt, warnings, timings, t0, split, emb)


def _report_matrix_size(
    adata: Any, cfg: PipelineConfig, reg: Registry, warnings: list[str]
) -> None:
    """Log the matrix size and warn if the data is dense on disk.

    Added after an out-of-memory crash on a real experiment. The pipeline keeps
    the expression matrix sparse throughout, but it cannot un-densify an h5ad
    that was written dense -- and at Perturb-seq scale that difference is the
    difference between 3 GB and 40 GB. Stating the numbers up front means an
    OOM kill is diagnosable from the log rather than from a blank terminal.
    """
    from .stats import is_sparse, nbytes_dense

    n_obs, n_vars = int(adata.n_obs), int(adata.n_vars)
    dense_gb = nbytes_dense(adata.X) / 1e9
    sparse_flag = is_sparse(adata.X)

    nnz = None
    if sparse_flag:
        nnz = int(getattr(adata.X, "nnz", 0))
        actual_gb = nnz * 12 / 1e9         # ~8 B data + 4 B index
        density = nnz / max(n_obs * n_vars, 1)
    else:
        actual_gb = dense_gb
        density = 1.0

    _log(
        cfg,
        f"matrix: {n_obs:,} cells x {n_vars:,} features, "
        f"{'sparse' if sparse_flag else 'DENSE'}, "
        f"~{actual_gb:.1f} GB in memory "
        f"(dense equivalent {dense_gb:.1f} GB)",
    )

    reg.metric("summary", "n_cells_input", "Cells in input", n_obs, order=1)
    reg.note(
        "appendix", "matrix_size", "Input matrix",
        (
            f"{n_obs:,} cells &times; {n_vars:,} features, stored "
            f"{'sparse' if sparse_flag else '<strong>dense</strong>'}"
            + (f" ({density * 100:.1f}% non-zero)" if nnz is not None else "")
            + f". Roughly {actual_gb:.1f} GB in memory; the dense equivalent "
            f"would be {dense_gb:.1f} GB. The pipeline keeps the expression "
            f"matrix sparse and densifies only small column blocks."
        ),
        order=2,
    )

    if not sparse_flag and dense_gb > 2.0:
        msg = (
            f"The expression matrix is stored DENSE and is ~{dense_gb:.1f} GB. "
            f"Single-cell counts are typically 90-95% zeros, so converting this "
            f"h5ad to sparse would cut memory roughly tenfold and make the run "
            f"substantially faster. In Python:\n"
            f"    import scipy.sparse as sp, anndata as ad\n"
            f"    a = ad.read_h5ad(path); a.X = sp.csr_matrix(a.X)\n"
            f"    a.write_h5ad(new_path)"
        )
        warnings.append(msg)
        reg.note("appendix", "dense_matrix", "Input is stored dense",
                 msg.replace("\n", "<br>"), level="warn", order=3)

    if actual_gb > 8.0:
        _log(
            cfg,
            f"WARNING: this is a large matrix (~{actual_gb:.1f} GB). Peak usage "
            f"will be a few times this. If the process is killed with no output, "
            f"it ran out of memory -- request more, or subset the experiment.",
        )


def _save_threshold_state(cfg: PipelineConfig, thresholds: Any) -> None:
    """Record the auto-derived thresholds so the next run can spot edits."""
    payload = {
        "auto_derived": {
            k: (float(v) if v is not None else None)
            for k, v in thresholds.as_dict().items()
        },
        "source": dict(thresholds.source),
        "written": pd.Timestamp.now().isoformat(timespec="seconds"),
        "note": (
            "Written by an explore run. The next full run compares the manifest's "
            "threshold columns against 'auto_derived' to tell an edited value "
            "apart from an untouched one. Deleting this file simply means the "
            "next report cannot make that distinction."
        ),
    }
    try:
        cfg.threshold_state_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    except OSError:
        pass        # advisory only; never fail a run over it


def _load_threshold_state(cfg: PipelineConfig) -> dict[str, float | None] | None:
    try:
        payload = json.loads(cfg.threshold_state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    got = payload.get("auto_derived")
    return got if isinstance(got, dict) else None


def _warn_if_thresholds_unreviewed(
    cfg: PipelineConfig, thresholds: Any, reg: Registry, warnings: list[str]
) -> None:
    """Say plainly whether the thresholds behind this report were reviewed.

    Three cases, and the report states which one applies:

    * derived automatically for this run (no explore step happened)
    * carried over from an explore run and left unchanged
    * carried over and edited by a human

    The distinction matters because a filled-in manifest column looks identical
    in all three cases. Without this, "the thresholds are in the manifest" would
    imply human review that may never have occurred.
    """
    used = thresholds.as_dict()
    sources = thresholds.source

    if cfg.auto_thresholds or all(
        sources.get(k) == "auto" for k in used
    ):
        msg = (
            "The QC thresholds behind this report were derived from the data, not "
            "reviewed by a person. They are a defensible starting point, not a "
            "considered choice: the automatic upper bounds in particular are "
            "deliberately loose, and a real high-RNA population can sit outside "
            "them. Re-run without <code>--auto-thresholds</code> to inspect the "
            "distributions first."
        )
        reg.note("cell_qc", "thresholds_unreviewed",
                 "Thresholds were not reviewed", msg, level="warn", order=2)
        warnings.append("QC thresholds were auto-derived and not reviewed.")
        return

    recorded = _load_threshold_state(cfg)
    if not recorded:
        return

    unchanged, edited = [], []
    for key, value in used.items():
        was = recorded.get(key)
        if was is None or value is None:
            continue
        if abs(float(value) - float(was)) <= 1e-9:
            unchanged.append(key)
        else:
            edited.append(key)

    if unchanged and not edited:
        msg = (
            "Every QC threshold in the manifest is byte-identical to the value the "
            "previous explore run derived automatically, so nothing was changed "
            "after the review step. That is a perfectly reasonable outcome if you "
            "looked at the distributions and agreed with them &mdash; but it is "
            "indistinguishable, from the manifest alone, from not having looked. "
            "Noted here so the report does not imply a judgement that may not have "
            "been made."
        )
        reg.note("cell_qc", "thresholds_unchanged",
                 "Thresholds unchanged since the explore run", msg, level="warn",
                 order=2)
        warnings.append(
            "QC thresholds are unchanged from the auto-derived values."
        )
    elif edited:
        reg.note(
            "cell_qc", "thresholds_edited", "Thresholds were reviewed",
            (
                f"{len(edited)} threshold(s) were changed from the automatically "
                f"derived values after the explore run "
                f"(<code>{', '.join(sorted(edited))}</code>)"
                + (
                    f"; {len(unchanged)} were left as derived "
                    f"(<code>{', '.join(sorted(unchanged))}</code>)."
                    if unchanged else "."
                )
            ),
            level="info", order=2,
        )


def _emit_downsampling_check(
    sm: "SEQ.SeqMetrics | None",
    qc: pd.DataFrame,
    obs: pd.DataFrame,
    guide: Modality,
    hto: Modality,
    manifest: Manifest | None,
    cfg: PipelineConfig,
    reg: Registry,
) -> None:
    """Depth measured from the h5ad, against the upstream DRAGEN metrics.

    The DRAGEN numbers describe the run BEFORE downsampling; the h5ad is what
    is actually being analysed. Presenting only the former invites the reader to
    attribute pre-downsampling depth to the analysed object.

    Units differ and are not reconciled: DRAGEN counts reads, the h5ad holds
    UMIs. The panels are therefore separate, each labelled with its own unit,
    and the interpretable comparison is the spread ACROSS samples -- if
    downsampling targeted a common depth, the coefficient of variation should
    be much smaller after than before.
    """
    import matplotlib.pyplot as plt

    sample_col = resolve_column(obs, cfg.modality.sample_col_candidates)
    if sample_col is None:
        return
    sample = obs[sample_col].astype(str)
    sample.index = pd.Index([str(x) for x in obs.index])
    sample = sample.reindex(qc.index)

    guide_umis = (
        pd.Series(qc["guide_total_umis"]) if "guide_total_umis" in qc.columns
        else None
    )
    hto_umis = (
        pd.Series(qc["hto_total_umis"]) if "hto_total_umis" in qc.columns else None
    )
    obs_depth = SEQ.observed_depth_by_sample(qc, sample, guide_umis, hto_umis)
    if obs_depth.empty:
        return
    obs_depth.to_csv(cfg.table_dir / "observed_depth_by_sample.csv", index=False)

    # Pre-downsampling counterpart, per sample where the manifest maps it.
    pre = None
    if sm is not None and not sm.empty and manifest is not None:
        wide = sm.derived()
        if "sample" in wide.columns:
            keep = [c for c in ("sample", "mean_reads_per_cell",
                                "crispr_mean_reads_per_cell") if c in wide.columns]
            if len(keep) > 1:
                pre = wide[keep].copy()
                # Duplicate prefixes collapse several samples onto one metrics
                # file; average rather than silently taking the first.
                pre = pre.groupby("sample", as_index=False).mean(numeric_only=True)

    panels: list[tuple[pd.DataFrame, str, str, str]] = []
    if pre is not None and "mean_reads_per_cell" in pre.columns:
        panels.append((pre, "mean_reads_per_cell",
                       "BEFORE: mean GEX reads/cell (DRAGEN)", "reads"))
    panels.append((obs_depth, "obs_mean_umis_per_cell",
                   "AFTER: mean GEX UMIs/cell (this h5ad)", "UMIs"))
    if pre is not None and "crispr_mean_reads_per_cell" in pre.columns:
        panels.append((pre, "crispr_mean_reads_per_cell",
                       "BEFORE: mean guide reads/cell (DRAGEN)", "reads"))
    if "obs_mean_guide_umis_per_cell" in obs_depth.columns:
        panels.append((obs_depth, "obs_mean_guide_umis_per_cell",
                       "AFTER: mean guide UMIs/cell (this h5ad)", "UMIs"))

    nrows, ncols = P.grid_dims(len(panels), max_cols=2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.3 * nrows),
                            squeeze=False)
    cv_lines: list[str] = []
    for ax, (src, col, title, unit) in zip(axes.ravel(), panels):
        d = src.dropna(subset=[col])
        labels = d["sample"].astype(str).tolist()
        vals = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)
        x = np.arange(len(labels), dtype=float)
        ax.bar(x, vals, color=P.palette(cfg.figures, len(labels)))
        if vals.size and np.isfinite(vals).any():
            top = float(np.nanmax(vals))
            ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=6)
        cv = SEQ.spread(vals)
        ax.set_title(f"{title}\nspread across samples: CV {cv:.1f}%", fontsize=8)
        ax.set_ylabel(unit)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        cv_lines.append(f"{title}: CV {cv:.1f}%")
    P.blank_unused_axes(axes, len(panels))
    fig.suptitle("depth before vs after downsampling", fontsize=10)
    fig.tight_layout()

    reg.figure(
        "seq_qc", "downsampling", "Depth before vs after downsampling",
        P.save_figure(fig, cfg.fig_dir / "downsampling_check.png", cfg.figures),
        caption=(
            "Top/left panels come from the upstream DRAGEN run and describe the "
            "libraries BEFORE downsampling. The others are measured from the "
            "h5ad being analysed here, AFTER downsampling. "
            "<b>The units differ and are deliberately not reconciled:</b> "
            "DRAGEN counts reads, an h5ad holds deduplicated UMIs, and "
            "downsampling reads reduces UMIs sub-linearly because of "
            "saturation. So the after/before ratio is not the downsampling "
            "factor and should not be read as one. What is directly "
            "interpretable is the spread across samples quoted on each panel: "
            "if downsampling targeted a common depth, the CV should be much "
            "smaller after than before. A CV that is unchanged means the "
            "downsampling did not take effect on this object."
        ),
        order=15, width="full",
    )
    reg.table(
        "seq_qc", "observed_depth", "Depth measured from this h5ad",
        path=cfg.table_dir / "observed_depth_by_sample.csv",
        inline=obs_depth.round(1).to_dict("records"),
        columns=list(obs_depth.columns),
        caption=(
            "Per-sample depth of the object actually analysed. Compare against "
            "the DRAGEN table above, which describes the run before "
            "downsampling."
        ),
        order=25,
    )
    reg.note(
        "seq_qc", "downsampling_summary", "Downsampling check", " | ".join(cv_lines),
        order=16,
    )


def _scale_metric_for_plot(col: str, vals: np.ndarray):
    """``(values, formatter, ylabel suffix)`` for one sequencing-metric panel.

    DRAGEN reports rate metrics as fractions (``pct_reads_in_cells = 0.607``),
    and the panel formatted every value with ``"{:,.0f}"``. That printed "1"
    above a bar of height 0.607 and "0" above one of 0.42 -- digits that look
    like noise because they are. Rates are converted to percent and formatted
    to one decimal; counts keep a thousands separator.
    """
    finite = vals[np.isfinite(vals)]
    is_rate = any(t in col for t in ("pct_", "_pct", "frac", "rate", "saturation"))
    if is_rate and finite.size and float(np.nanmax(finite)) <= 1.5:
        # Fractions on a 0-1 scale: show as percent.
        return vals * 100.0, (lambda v: f"{v:.1f}"), " (%)"
    if is_rate:
        return vals, (lambda v: f"{v:.1f}"), " (%)"
    if finite.size and float(np.nanmax(finite)) < 100:
        return vals, (lambda v: f"{v:,.1f}"), ""
    return vals, (lambda v: f"{v:,.0f}"), ""


def _emit_seq_qc(sm: SEQ.SeqMetrics, cfg: PipelineConfig, reg: Registry) -> None:
    """Sequencing-metric panels and table."""
    import matplotlib.pyplot as plt

    if sm.empty:
        reg.skipped(
            "seq_qc", "all", "Sequencing QC",
            f"No metrics files could be parsed. Prefixes without a metrics file: "
            f"{', '.join(sm.missing) or 'none'}.",
        )
        return

    wide = sm.derived()
    wide.to_csv(cfg.table_dir / "sequencing_metrics.csv", index=False)

    panels = [
        ("pct_reads_in_cells", "% reads in passing cells"),
        ("mean_reads_per_cell", "mean GEX reads per cell"),
        ("crispr_pct_reads_in_cells", "% CRISPR reads in cells"),
        ("crispr_mean_reads_per_cell", "mean guide reads per cell"),
        ("pct_mapped_reads", "% mapped reads"),
        ("estimated_cells", "estimated cells"),
    ]
    available = [(c, lab) for c, lab in panels if c in wide.columns
                 and pd.to_numeric(wide[c], errors="coerce").notna().any()]
    if available:
        nrows, ncols = P.grid_dims(len(available), max_cols=3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.0 * nrows),
                                 squeeze=False)
        # Explicit numeric x positions, NOT category strings.
        #
        # `ax.bar(labels, ...)` treats labels as categories, so duplicate
        # prefixes -- legitimate when two libraries share a sequencing run,
        # e.g. a CSU and an IVT sample off the same lane -- collapse into one
        # bar while the value labels were still placed at index i. The result
        # is numbers floating in space away from any bar.
        prefixes = wide["prefix"].astype(str).tolist()
        if len(set(prefixes)) != len(prefixes) and "sample" in wide.columns:
            labels = [f"{s}\n{p}" for s, p in
                      zip(wide["sample"].astype(str), prefixes)]
        else:
            labels = prefixes
        x = np.arange(len(labels), dtype=float)
        colors = P.palette(cfg.figures, len(labels))
        for ax, (col, lab) in zip(axes.ravel(), available):
            vals = pd.to_numeric(wide[col], errors="coerce").to_numpy(dtype=float)
            vals, fmt, unit = _scale_metric_for_plot(col, vals)
            ax.bar(x, vals, color=colors)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                # Headroom so the value labels are not clipped by the axes box.
                top = float(np.nanmax(finite))
                ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.text(xi, v, fmt(v), ha="center", va="bottom", fontsize=6)
            ax.set_ylabel(lab + unit)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        P.blank_unused_axes(axes, len(available))
        fig.suptitle("sequencing metrics per library", fontsize=10)
        fig.tight_layout()
        reg.figure(
            "seq_qc", "metrics",
            "Sequencing metrics per library (upstream, pre-downsampling)",
            P.save_figure(fig, cfg.fig_dir / "sequencing_metrics.png", cfg.figures),
            caption=SEQ_CAPTION, order=10, width="full",
        )

    reg.table(
        "seq_qc", "table", "Sequencing metrics (upstream, pre-downsampling)",
        path=cfg.table_dir / "sequencing_metrics.csv",
        inline=wide.round(3).to_dict("records"), columns=list(wide.columns),
        order=20,
    )
    if sm.missing:
        reg.note(
            "seq_qc", "missing", "Libraries without metrics",
            f"No metrics file was found for: {', '.join(sm.missing)}. "
            f"Those libraries are absent from the panels above.",
            level="warn", order=5,
        )
    if sm.unmatched:
        reg.note(
            "seq_qc", "unmatched", "Unrecognised metric names",
            f"{len(sm.unmatched)} metric name(s) in the files were not recognised "
            f"and were ignored: {', '.join(sm.unmatched[:10])}"
            + ("..." if len(sm.unmatched) > 10 else "")
            + ". Add them to seqmetrics.METRIC_ALIASES to surface them.",
            order=30,
        )


PRE_DOWNSAMPLE_WARNING = (
    " <b>These numbers describe the upstream DRAGEN run, BEFORE any "
    "downsampling.</b> They are not the depth of the h5ad analysed in this "
    "report. See 'Depth before vs after downsampling' for the object's own "
    "measured depth."
)
SEQ_CAPTION = T.SEQ_QC_DESC + " " + T.SEQ_QC_NOTE + PRE_DOWNSAMPLE_WARNING


def _cluster_vs_hto(
    emb: GEX.EmbeddingResult, calls: HTO.HTOCalls, cfg: PipelineConfig,
    reg: Registry,
) -> None:
    """Cluster composition by hashtag identity.

    In a multiplexed experiment where hashtags encode cell line, this is the
    panel that says whether clustering recovered the known populations -- a
    strong, independent check on the whole transcriptome pipeline that neither
    previous version produced.
    """
    import matplotlib.pyplot as plt

    common = emb.obs.index.intersection(calls.per_cell.index)
    if len(common) < 20:
        return
    singlet = calls.per_cell.loc[common]
    singlet = singlet[singlet["hto_class"] == "Singlet"]
    if singlet.empty:
        return
    clusters = emb.obs.loc[singlet.index, "cluster"].astype(str)
    ct = pd.crosstab(clusters, singlet["hto_call"], normalize="index")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    P.stacked_fraction_bars(axes[0], ct, cfg.figures, legend_title="hashtag")
    axes[0].set_xlabel("cluster")
    axes[0].set_title("hashtag identity composition of each cluster")
    P.scatter_embedding(
        axes[1], emb.umap[[emb.obs.index.get_loc(i) for i in singlet.index]],
        singlet["hto_call"].astype(str), cfg.figures,
        title="embedding coloured by hashtag call (singlets)", categorical=True,
    )
    fig.tight_layout()

    purity = float(ct.max(axis=1).mean()) if not ct.empty else float("nan")
    reg.figure(
        "crosschecks", "cluster_vs_hto", "Clusters vs hashtag identity",
        P.save_figure(fig, cfg.fig_dir / "crosscheck_cluster_vs_hto.png",
                      cfg.figures),
        caption=(
            "Transcriptome clustering and hashtag demultiplexing are independent "
            "measurements. If the hashtags encode distinct populations (cell "
            "lines, timepoints, treatments), clusters should map onto them: mean "
            f"cluster purity here is {purity * 100:.0f}%. A low value means either "
            "the clustering is not resolving the known populations or the hashtag "
            "calls are noisy &mdash; the per-hashtag diagnostic panels distinguish "
            "the two."
        ),
        order=30, width="full",
    )
    reg.metric(
        "summary", "cluster_hto_purity", "Cluster/hashtag agreement",
        round(purity * 100, 1), unit="%",
        level=("good" if purity > 0.8 else "warn" if purity > 0.55 else "poor"),
        order=60,
    )


def _finish(
    cfg: PipelineConfig, reg: Registry, filt, warnings: list[str],
    timings: dict[str, float], t0: float, split: SplitResult,
    emb: GEX.EmbeddingResult | None,
) -> RunResult:
    problems = reg.verify()
    for i, p in enumerate(problems):
        reg.note("appendix", f"missing_artifact_{i}", "Missing output", p,
                 level="poor", order=100 + i)
        warnings.append(p)

    provenance = {
        "pipeline version": __version__,
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(cfg.manifest_path),
        "h5ad": str(cfg.h5ad_path),
        "output": str(cfg.analysis_dir),
        "cells analysed": f"{filt.n_after:,} of {filt.n_before:,}",
        "backend": emb.backend if emb else "n/a (explore mode)",
        "batch correction": emb.batch_corrected if emb else "n/a",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "runtime (s)": f"{time.time() - t0:.1f}",
    }
    provenance.update({f"stage: {k}": f"{v:.1f}s" for k, v in timings.items()})

    cfg.save(cfg.analysis_dir / "config_used.json")
    reg.save()

    title = cfg.report.title or f"Perturb-seq QC report — {cfg.analysis_dir.parent.name}"
    report_path = REPORT.build_report(
        reg, cfg.report_path, cfg.report, title, provenance
    )
    _log(cfg, f"report written to {report_path}")

    return RunResult(
        registry=reg, report_path=report_path, thresholds=filt.thresholds,
        n_cells_before=filt.n_before, n_cells_after=filt.n_after,
        warnings=warnings, timings=timings,
    )
