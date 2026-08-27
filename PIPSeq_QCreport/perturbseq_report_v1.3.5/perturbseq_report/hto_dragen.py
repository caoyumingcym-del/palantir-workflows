"""
Recovering hashtag counts from the DRAGEN output when the h5ad has none.

``modalities.split_modalities`` looks for a hashtag matrix in four places
inside the h5ad itself (var feature-types, obsm, uns, flattened obs columns --
see ``modalities.py``). All four assume the hashtag library was carried
through into the object. It sometimes isn't: the h5ad was built from a GEX-only
feature-barcode matrix and the hashtag counts exist only as DRAGEN's own
per-run output, ``<prefix>.scRNA.cellhashing.tsv``, sitting next to the other
metric files under the manifest's ``dragen_path``.

This module is the fifth resolution path, tried only when the first four have
all failed: read the DRAGEN cellhashing file(s) named in the manifest, align
their barcodes to the h5ad's ``obs_names``, and hand back a ``Modality`` exactly
like any other. Everything here is defensive rather than assumed correct,
because a silent barcode mismatch is worse than no hashtag data at all: it
would attach hashtag B's counts to cell A and every downstream call would be
confidently wrong. Concretely:

* the file is parsed both ways (barcodes as rows, barcodes as columns) and the
  orientation is picked from which axis actually looks like real barcodes, not
  assumed;
* non-numeric columns (a stray ``total`` or ``unmapped`` column some DRAGEN
  versions append) are dropped and reported rather than silently coerced to
  NaN and then to 0;
* the barcode string transform needed to match the h5ad (a ``-1`` suffix
  present in one file and not the other, most commonly) is *chosen* by trying
  several candidates and taking the one with the best overlap, and the run
  fails loudly rather than proceeding on a low-overlap guess;
* every recovered hashtag matrix is reindexed to the h5ad's own ``obs_names``,
  so a cell absent from the cellhashing file gets zeros rather than being
  dropped -- consistent with how ``Modality.reindex_cells`` already treats an
  obsm-derived guide matrix that doesn't cover every cell.

Multiple runs for one sample: sum, or keep the strongest match? Not guessable.
-------------------------------------------------------------------------------
The pipeline is used across many experiments, and a manifest's multiple rows
for one sample do not always mean the same thing. On MDL-1856, comparing
actual hashtag counts for the same barcode across every run (see
``diagnose_cellhashing_tsv.py``'s section G) showed barcodes flagged "passed"
in every run with comparable, non-trivial counts in every run -- the
signature of one physical cell's reads being split across several library
preparations of the SAME cells/cDNA, not of an unrelated barcode collision.
For that experiment, the right thing to do when the same (aligned) barcode
appears in more than one run's cellhashing file is to ADD the counts across
runs -- that is the cell's true total hashtag signal. An experiment whose
multiple runs are instead genuinely independent pools would show the
opposite pattern (real signal in one run, near-zero elsewhere for the same
barcode), and summing there would incorrectly inflate a real cell's count
with an unrelated run's background.

Because both cases produce the exact same thing on disk -- a barcode present
in more than one run's file -- this module never infers which case applies.
It is a required manifest declaration, ``dragen_runs_share_cells``
(``Manifest.dragen_runs_share_cells``; see ``manifest.py``), consumed here as
``combine_mode`` ("sum" when declared yes, "max" -- keep only the
strongest-matching run, discard the rest -- when declared no).
``build_hto_modality_multi_sample`` resolves this per sample across a whole
experiment and refuses to recover a sample's hashtags at all if it has
multiple runs and no declaration, rather than default either way: guessing
wrong changes every reported hashtag count by up to Nx, where N is the
number of runs for that sample.

See ``diagnose_cellhashing_tsv.py`` for the read-only tool used to establish a
given file's actual layout, and which of the two cases applies, before
declaring it in the manifest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import ModalityConfig
from .modalities import Modality

BARCODE_RE = re.compile(r"^[ACGTN]{6,40}(-\d+)?$", re.IGNORECASE)

# Tried in order against the h5ad's obs_names; the first (or best) with
# sufficient overlap wins. Order matters only as a tie-break.
_BARCODE_TRANSFORMS: dict[str, Any] = {
    "as-is": lambda s: s,
    "strip -N suffix": lambda s: s.str.replace(r"-\d+$", "", regex=True),
    "add -1 suffix": lambda s: s.where(s.str.contains(r"-\d+$"), s + "-1"),
    "prefix before first '-'": lambda s: s.str.split("-").str[0],
    "prefix before first '_'": lambda s: s.str.split("_").str[0],
    "suffix after last '_'": lambda s: s.str.split("_").str[-1],
    "uppercase": lambda s: s.str.upper(),
}


class CellHashingLoadError(ValueError):
    """The cellhashing file exists but could not be trusted as hashtag counts."""


@dataclass
class CellHashingFile:
    """One parsed DRAGEN cellhashing.tsv, oriented cells x hashtags."""

    path: Path
    counts: pd.DataFrame          # index = raw barcode string, columns = hashtag names
    orientation: str              # "barcodes_as_rows" | "barcodes_as_columns"
    dropped_columns: list[str]
    notes: list[str]


# ===========================================================================
# Locating the file
# ===========================================================================
def find_cellhashing_file(
    dragen_path: Path, prefix: str, patterns: Sequence[str]
) -> Path | None:
    """First matching file under ``dragen_path`` for this run's prefix."""
    if not dragen_path.exists():
        return None
    for pattern in patterns:
        candidate = pattern.format(prefix=prefix)
        if "*" in candidate:
            hits = sorted(dragen_path.glob(candidate))
            if hits:
                return hits[0]
        else:
            p = dragen_path / candidate
            if p.exists():
                return p
    return None


# ===========================================================================
# Parsing + orientation
# ===========================================================================
def _looks_like_barcodes(values: Sequence[str]) -> float:
    vals = [str(v) for v in values]
    if not vals:
        return 0.0
    return sum(1 for v in vals if BARCODE_RE.match(v)) / len(vals)


def load_cellhashing_file(path: Path) -> CellHashingFile:
    """Parse one DRAGEN cellhashing file into a cells x hashtags frame.

    Raises ``CellHashingLoadError`` when neither axis looks like barcodes, or
    when nothing numeric is left after dropping non-count columns -- both
    cases where guessing would be worse than stopping.
    """
    notes: list[str] = []
    try:
        raw = pd.read_csv(path, sep=None, engine="python", header=0, index_col=0)
    except Exception as exc:
        raise CellHashingLoadError(
            f"Could not parse {path} as a delimited table ({exc}). Run "
            f"diagnose_cellhashing_tsv.py against it to see the raw layout."
        ) from exc

    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise CellHashingLoadError(
            f"{path} parsed to an empty table ({raw.shape}). Nothing to load."
        )

    row_frac = _looks_like_barcodes(list(raw.index[:200]))
    col_frac = _looks_like_barcodes(list(raw.columns[:200]))

    if row_frac >= 0.5 and row_frac >= col_frac:
        counts, orientation = raw, "barcodes_as_rows"
    elif col_frac >= 0.5:
        counts, orientation = raw.T, "barcodes_as_columns"
        notes.append(
            f"{path.name}: barcodes were found on the COLUMN axis (transposed "
            f"relative to the default cells x hashtags layout); transposed "
            f"before use."
        )
    else:
        raise CellHashingLoadError(
            f"Neither axis of {path} looks like 10x barcodes (row match "
            f"{row_frac:.2f}, column match {col_frac:.2f}). Run "
            f"diagnose_cellhashing_tsv.py against this file to see why before "
            f"trusting it."
        )

    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)

    # Drop columns that are not (mostly) numeric -- e.g. a 'total' or
    # 'unmapped' column some DRAGEN versions append -- rather than silently
    # coercing them to NaN/0 and reporting them as a hashtag with no signal.
    numeric_frac = counts.apply(
        lambda s: pd.to_numeric(s, errors="coerce").notna().mean()
    )
    keep = numeric_frac[numeric_frac >= 0.9].index.tolist()
    dropped = [c for c in counts.columns if c not in keep]
    if dropped:
        notes.append(
            f"{path.name}: dropped non-numeric column(s) {dropped} -- not "
            f"treated as hashtags. Check with diagnose_cellhashing_tsv.py if "
            f"any of these was actually meant to be one."
        )
    if not keep:
        raise CellHashingLoadError(
            f"No numeric hashtag-count columns remained in {path} after "
            f"dropping {dropped}. Nothing to load."
        )
    counts = counts[keep].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    dup = counts.index[counts.index.duplicated()].unique().tolist()
    if dup:
        notes.append(
            f"{path.name}: {len(dup)} duplicate barcode(s) in the cellhashing "
            f"file, e.g. {dup[:5]}. Summed rather than dropped, since a 10x "
            f"barcode collision inside one lane means the same droplet was "
            f"reported twice, not two different cells."
        )
        counts = counts.groupby(level=0).sum()

    return CellHashingFile(
        path=path, counts=counts, orientation=orientation,
        dropped_columns=dropped, notes=notes,
    )


# ===========================================================================
# Barcode alignment
# ===========================================================================
@dataclass
class BarcodeMatch:
    transform: str
    n_matched: int
    frac_of_target: float
    frac_of_source: float


def best_barcode_transform(
    source: pd.Index, target: pd.Index
) -> tuple[BarcodeMatch, pd.Index]:
    """Which string transform of ``source`` overlaps ``target`` best.

    Tries every candidate rather than assuming the common '-1' case, because
    the actual mismatch on real data is as likely to be a sample prefix
    (pooled multi-lane objects) as a missing suffix.
    """
    target_set = set(target.astype(str))
    source = pd.Index(source.astype(str))
    best: BarcodeMatch | None = None
    best_transformed = source
    for label, fn in _BARCODE_TRANSFORMS.items():
        try:
            transformed = pd.Index(fn(source.to_series()).to_numpy())
        except Exception:
            continue
        n = len(set(transformed) & target_set)
        m = BarcodeMatch(
            transform=label, n_matched=n,
            frac_of_target=n / len(target_set) if target_set else 0.0,
            frac_of_source=n / len(source) if len(source) else 0.0,
        )
        if best is None or m.n_matched > best.n_matched:
            best, best_transformed = m, transformed
    assert best is not None
    return best, best_transformed


# ===========================================================================
# Public entry point
# ===========================================================================
def build_hto_modality_from_dragen(
    runs: Sequence[dict[str, Any]],
    target_obs_names: Sequence[str],
    cfg: ModalityConfig,
    combine_mode: str = "sum",
) -> tuple[Modality | None, list[str]]:
    """Recover a hashtag ``Modality`` from DRAGEN cellhashing.tsv file(s).

    ``runs`` is ``Manifest.dragen_runs()`` restricted to ONE sample -- one
    dict per (sample, prefix, dragen_path). Every run's file is loaded and
    matched independently (a different barcode transform can be needed per
    run, e.g. a per-run prefix), then combined into one table aligned to
    ``target_obs_names`` before reindexing.

    ``combine_mode`` decides what happens when the same (aligned) barcode is
    recovered from more than one run:

    ``"sum"`` (default)
        Add the counts together. Correct when this sample's multiple runs
        are multiple library preparations of the SAME cells/cDNA -- the
        normal case this fallback was built for (see MODULE docstring) --
        because the barcode's total hashtag signal is split across the runs
        that sequenced it.
    ``"max"``
        Keep only the counts from whichever run had the larger total for
        that barcode, discarding the rest. Use this when the runs for this
        sample are genuinely independent pools that happen to share a row in
        the manifest; a barcode appearing in more than one run's file then
        represents coincidental reuse of the same string from the shared
        combinatorial whitelist, not the same cell, so summing would inflate
        the count with an unrelated run's noise.

    This function does not decide which mode applies -- that is a per-sample
    declaration (``Manifest.dragen_runs_share_cells``) that cannot be
    inferred from the file. Callers with more than one sample should use
    ``build_hto_modality_multi_sample`` instead, which resolves the mode per
    sample and refuses (rather than guesses) when a sample with multiple
    runs has not declared one.

    Returns ``(None, notes)`` when nothing usable was found or the barcode
    overlap never cleared ``cfg.hto_dragen_min_match_frac`` for any run --
    callers should fall back to reporting "no hashtag data", exactly as when
    the h5ad-internal lookups fail, rather than use a low-confidence match.
    """
    if combine_mode not in ("sum", "max"):
        raise ValueError(f"combine_mode must be 'sum' or 'max', got {combine_mode!r}")
    notes: list[str] = []
    target = pd.Index([str(x) for x in target_obs_names])
    if len(target) == 0:
        return None, ["No target cells to align hashtag counts against."]

    per_run_frames: list[pd.DataFrame] = []
    for run in runs:
        prefix = str(run["prefix"])
        dragen_path = Path(run["dragen_path"])
        found = find_cellhashing_file(dragen_path, prefix, cfg.hto_dragen_file_patterns)
        if found is None:
            notes.append(
                f"No cellhashing file found for prefix {prefix!r} under "
                f"{dragen_path} (tried {list(cfg.hto_dragen_file_patterns)})."
            )
            continue
        try:
            parsed = load_cellhashing_file(found)
        except CellHashingLoadError as exc:
            notes.append(f"{prefix}: {exc}")
            continue
        notes.extend(parsed.notes)

        match, transformed = best_barcode_transform(parsed.counts.index, target)
        notes.append(
            f"{found.name}: best barcode transform '{match.transform}' matched "
            f"{match.n_matched:,} of {len(parsed.counts):,} file barcodes "
            f"({100 * match.frac_of_source:.1f}%), covering "
            f"{100 * match.frac_of_target:.1f}% of this h5ad's cells."
        )
        if match.frac_of_target < cfg.hto_dragen_min_match_frac and match.n_matched == 0:
            notes.append(
                f"{prefix}: barcode overlap is zero under every transform tried "
                f"({list(_BARCODE_TRANSFORMS)}). This cellhashing file cannot be "
                f"matched to the h5ad's obs_names -- check it is really the "
                f"right lane/sample for this object (diagnose_cellhashing_tsv.py "
                f"--h5ad <this h5ad> will show the same barcode examples this "
                f"run saw)."
            )
            continue

        aligned = parsed.counts.copy()
        aligned.index = transformed
        # DRAGEN writes this file over (close to) the full combinatorial
        # barcode whitelist, so the vast majority of rows are background/
        # ambient noise for barcodes that are not among this h5ad's cells at
        # all. Dropping them here -- rather than after concatenating every
        # run -- is what keeps a 4-run x 5M-row experiment tractable to sum;
        # it changes nothing about the result, since a barcode absent from
        # `target` is reindexed to zero at the end regardless.
        aligned = aligned.loc[aligned.index.isin(target)]
        if aligned.index.duplicated().any():
            # Two distinct raw barcodes collided onto the same string under
            # this transform (e.g. one already ended in "-2" and the other
            # had "-1" added to it). Summed for the same reason as the
            # cross-run case below: this is a within-run collision, not
            # grounds to silently drop one of the two cells' reads.
            aligned = aligned.groupby(level=0).sum()
        per_run_frames.append(aligned)

    if not per_run_frames:
        notes.append(
            "No hashtag counts could be recovered from any DRAGEN "
            "cellhashing.tsv listed in the manifest."
        )
        return None, notes

    combined = pd.concat(per_run_frames, axis=0)
    n_runs_per_barcode = combined.index.value_counts()
    dup_across_runs = n_runs_per_barcode[n_runs_per_barcode > 1]
    if combine_mode == "sum":
        # This sample's multiple runs are multiple library preparations of
        # the SAME cells/cDNA (declared via dragen_runs_share_cells=yes, and
        # confirmed for MDL-1856 via diagnose_cellhashing_tsv.py's cross-run
        # count comparison: a barcode "passing" in every run showed
        # comparable non-trivial counts in every run, not signal in one and
        # noise in the rest). A cell's total hashtag signal is therefore the
        # SUM of its counts across every run that recovered it.
        if len(dup_across_runs):
            notes.append(
                f"{len(dup_across_runs):,} barcode(s) were recovered from "
                f"more than one run's cellhashing file (median "
                f"{int(dup_across_runs.median())} of {len(per_run_frames)} "
                f"runs recovered per shared barcode); their counts were "
                f"SUMMED across runs (dragen_runs_share_cells=yes for this "
                f"sample)."
            )
        combined = combined.groupby(level=0).sum()
    else:
        # This sample's runs are independent pools that happen to share one
        # manifest row (dragen_runs_share_cells=no). A barcode recovered
        # from more than one run's file is coincidental reuse of the same
        # string from the shared whitelist, not the same cell -- keep only
        # the run with the larger total for that barcode and drop the rest,
        # rather than let an unrelated run's background inflate the count.
        if len(dup_across_runs):
            notes.append(
                f"{len(dup_across_runs):,} barcode(s) were recovered from "
                f"more than one run's cellhashing file even though this "
                f"sample's runs are declared independent "
                f"(dragen_runs_share_cells=no). Kept only the run with the "
                f"larger total count for each such barcode; the rest were "
                f"discarded as coincidental whitelist reuse rather than "
                f"summed."
            )
        # Positional, not label-based: `combined` has duplicate index labels
        # by construction (that's what we're resolving), and `.loc[labels]`
        # on a non-unique index returns every row matching each label rather
        # than a clean permutation, which would silently multiply rows here.
        totals = combined.sum(axis=1).to_numpy()
        order = np.argsort(-totals, kind="stable")
        combined = combined.iloc[order]
        combined = combined[~combined.index.duplicated(keep="first")]

    hashtag_names = list(combined.columns)
    aligned_full = combined.reindex(target, fill_value=0.0)
    n_covered = int(combined.index.isin(target).sum())
    notes.append(
        f"Recovered {len(hashtag_names)} hashtag(s) for {n_covered:,} of "
        f"{len(target):,} cells ({100 * n_covered / len(target):.1f}%) from "
        f"DRAGEN cellhashing output; the remainder default to zero counts "
        f"across all hashtags and will be called Negative."
    )

    mod = Modality(
        kind="hto",
        X=aligned_full.to_numpy(dtype=np.float64),
        names=hashtag_names,
        source=(
            f"DRAGEN cellhashing.tsv ({len(per_run_frames)} of {len(runs)} "
            f"run(s) recovered, combine_mode={combine_mode!r}; not found in "
            f"the h5ad itself)"
        ),
        obs_names=list(target),
    )
    return mod, notes


# ===========================================================================
# Multi-sample orchestration
# ===========================================================================
def build_hto_modality_multi_sample(
    runs: Sequence[dict[str, Any]],
    target_obs_names: Sequence[str],
    sample_of_cell: Sequence[Any] | None,
    share_cells_by_sample: dict[str, bool | None],
    cfg: ModalityConfig,
) -> tuple[Modality | None, list[str]]:
    """Recover hashtag counts across an experiment that may span several
    samples, each with its own answer to "do this sample's runs share cells?"

    ``runs`` is the FULL ``Manifest.dragen_runs()`` (every sample). Grouped
    here by ``run["sample"]`` and recovered independently per sample with
    ``build_hto_modality_from_dragen(..., combine_mode=...)``, then the
    per-sample results are assembled into one ``Modality`` covering
    ``target_obs_names``.

    ``sample_of_cell`` must be the same length as ``target_obs_names``,
    giving each cell's sample label -- normally ``obs[sample_col]`` once
    ``sample_col`` has been resolved. Pass ``None`` only when every run in
    ``runs`` belongs to one sample (no assignment needed); with more than one
    distinct sample and ``sample_of_cell is None``, recovery is refused
    entirely, because there is no way to know which cells a given sample's
    runs apply to.

    For each sample with more than one run, ``share_cells_by_sample`` (from
    ``Manifest.dragen_runs_share_cells`` per sample) decides ``combine_mode``:
    ``True`` -> ``"sum"``, ``False`` -> ``"max"``. A sample with more than one
    run and no declaration (``None``) is SKIPPED -- not recovered, not
    guessed -- with a note explaining exactly what manifest column and value
    would resolve it. A sample with exactly one run needs no declaration.
    """
    notes: list[str] = []
    target = pd.Index([str(x) for x in target_obs_names])
    if len(target) == 0:
        return None, ["No target cells to align hashtag counts against."]

    runs_by_sample: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        runs_by_sample.setdefault(str(r["sample"]), []).append(r)

    if len(runs_by_sample) > 1 and sample_of_cell is None:
        return None, [
            f"The manifest declares dragen runs for {len(runs_by_sample)} "
            f"different samples ({sorted(runs_by_sample)}), but no per-cell "
            f"sample label was available to know which cells each sample's "
            f"runs apply to. Hashtag recovery from DRAGEN output was skipped "
            f"entirely. A resolvable sample/library column in obs "
            f"(ModalityConfig.sample_col_candidates) is required for a "
            f"multi-sample manifest."
        ]

    sample_labels = (
        pd.Series([str(s) for s in sample_of_cell], index=target)
        if sample_of_cell is not None else None
    )

    recovered_mods: list[Modality] = []
    for sample, sample_runs in runs_by_sample.items():
        if sample_labels is not None:
            sample_target = target[sample_labels.to_numpy() == sample]
        else:
            sample_target = target
        if len(sample_target) == 0:
            notes.append(
                f"Sample {sample!r} has dragen run(s) declared but no cells "
                f"in this h5ad labelled with that sample; skipped."
            )
            continue

        if len(sample_runs) > 1:
            share = share_cells_by_sample.get(sample)
            if share is None:
                notes.append(
                    f"Sample {sample!r} has {len(sample_runs)} dragen runs "
                    f"({[r['prefix'] for r in sample_runs]}) and no "
                    f"'dragen_runs_share_cells' declaration in the manifest, "
                    f"so hashtag recovery from DRAGEN output was SKIPPED for "
                    f"this sample's {len(sample_target):,} cell(s). Add a "
                    f"'dragen_runs_share_cells' column with 'yes' if these "
                    f"runs are multiple library preparations of the same "
                    f"cells/cDNA (counts will be summed across runs) or 'no' "
                    f"if they are independent runs/pools (only the "
                    f"strongest-matching run's counts will be kept per cell)."
                )
                continue
            combine_mode = "sum" if share else "max"
        else:
            combine_mode = "sum"  # irrelevant with only one run

        mod, sample_notes = build_hto_modality_from_dragen(
            sample_runs, list(sample_target), cfg, combine_mode=combine_mode,
        )
        notes.append(f"[{sample}] combine_mode={combine_mode!r}:")
        notes.extend(f"  {n}" for n in sample_notes)
        if mod is not None and mod.present:
            recovered_mods.append(mod)

    if not recovered_mods:
        notes.append(
            "No sample's hashtag data could be recovered from DRAGEN output."
        )
        return None, notes

    all_names: list[str] = []
    for mod in recovered_mods:
        for n in mod.names:
            if n not in all_names:
                all_names.append(n)

    combined_X = np.zeros((len(target), len(all_names)), dtype=np.float64)
    name_pos = {n: j for j, n in enumerate(all_names)}
    for mod in recovered_mods:
        expanded = mod.reindex_cells(list(target))
        cols = [name_pos[n] for n in expanded.names]
        combined_X[:, cols] += expanded.X

    n_samples_recovered = len(recovered_mods)
    notes.append(
        f"Combined hashtag recovery across {n_samples_recovered} of "
        f"{len(runs_by_sample)} sample(s) into {len(all_names)} hashtag "
        f"column(s) covering {len(target):,} cells."
    )

    mod = Modality(
        kind="hto", X=combined_X, names=all_names,
        source=(
            f"DRAGEN cellhashing.tsv, {n_samples_recovered} of "
            f"{len(runs_by_sample)} sample(s) recovered (multi-sample "
            f"combine); not found in the h5ad itself"
        ),
        obs_names=list(target),
    )
    return mod, notes
