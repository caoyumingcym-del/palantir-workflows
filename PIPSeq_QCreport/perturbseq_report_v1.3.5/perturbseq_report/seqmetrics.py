"""
DRAGEN / CellRanger sequencing-metric ingestion.

Reads the per-library metric CSVs produced by the alignment pipeline and
normalises them into one tidy DataFrame, so the rest of the pipeline never has
to know which vendor wrote the file or what they called a given metric.

The original looked up metrics by trying a hardcoded candidate list per metric
(``GEX_MEAN_READS_PER_CELL_CANDIDATES``, ``CRISPR_TOTAL_READS_CANDIDATES``,
and four more) inline in notebook cells, with a division guard that did not
actually catch NaN denominators.  Here each metric is declared once in a table,
lookups are case- and punctuation-insensitive, and every derived ratio goes
through ``stats.safe_divide``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .stats import safe_divide


def _norm_key(s: str) -> str:
    """Canonicalise a metric name for matching: lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


# Declarative metric table.  Each entry: canonical name -> the vendor strings
# that mean it.  Adding support for a new aligner is a data change here, not a
# code change in six places.
def observed_depth_by_sample(
    qc: "pd.DataFrame",
    sample: "pd.Series",
    guide_umis: "pd.Series | None" = None,
    hto_umis: "pd.Series | None" = None,
) -> "pd.DataFrame":
    """Per-sample sequencing depth measured FROM the h5ad.

    The DRAGEN metrics describe the upstream run, before any downsampling. This
    describes the object actually being analysed. Comparing the two is how you
    confirm a downsampling step did what it was meant to.

    One thing this cannot do, and the report says so rather than papering over
    it: DRAGEN counts **reads**, an h5ad holds **UMIs**. Downsampling reads
    reduces UMIs sub-linearly, because deduplication means the second read of a
    molecule was never counted as a second UMI. So the post/pre ratio is not
    the downsampling factor, and is not presented as one. What is directly
    interpretable is the *spread across samples*: downsampling to a common
    depth should collapse it.
    """
    import numpy as np
    import pandas as pd

    idx = sample.astype(str)
    frames = {"total_counts": qc["total_counts"].to_numpy(dtype=float)}
    if "n_genes_by_counts" in qc.columns:
        frames["n_genes"] = qc["n_genes_by_counts"].to_numpy(dtype=float)
    if guide_umis is not None:
        frames["guide_umis"] = pd.Series(guide_umis).reindex(qc.index).to_numpy(
            dtype=float)
    if hto_umis is not None:
        frames["hto_umis"] = pd.Series(hto_umis).reindex(qc.index).to_numpy(
            dtype=float)
    df = pd.DataFrame(frames, index=qc.index)
    df["__sample"] = idx.reindex(qc.index).to_numpy()

    g = df.groupby("__sample", dropna=True)
    out = pd.DataFrame({
        "sample": [str(k) for k in g.groups],
    }).set_index("sample")
    out["n_cells"] = g.size()
    out["obs_mean_umis_per_cell"] = g["total_counts"].mean()
    out["obs_median_umis_per_cell"] = g["total_counts"].median()
    out["obs_total_umis"] = g["total_counts"].sum()
    if "n_genes" in df.columns:
        out["obs_median_genes_per_cell"] = g["n_genes"].median()
    if "guide_umis" in df.columns:
        out["obs_mean_guide_umis_per_cell"] = g["guide_umis"].mean()
    if "hto_umis" in df.columns:
        out["obs_mean_hto_umis_per_cell"] = g["hto_umis"].mean()
    return out.reset_index()


def spread(values) -> float:
    """Coefficient of variation, as a percentage. NaN-safe."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    m = float(v.mean())
    return float(100.0 * v.std(ddof=1) / m) if m else float("nan")


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    # --- library level ---
    "total_reads": (
        "Total input reads", "Number of reads", "Total reads",
        "Number of reads in the library",
    ),
    "total_barcoded_reads": ("Total barcoded reads",),
    "reads_with_valid_umi": (
        "Reads with valid molecular identifier sequences",
        "Reads with valid UMI",
    ),
    "pct_mapped_reads": ("Mapped reads", "Reads mapped to genome"),
    "pct_reads_in_cells": (
        "Fraction of reads in passing cells", "Reads in cells",
        "Fraction of reads in cells",
    ),
    # --- cells ---
    "estimated_cells": (
        "Estimated number of cells", "Number of passing cells", "Passing cells",
        "Estimated Number of Cells",
    ),
    "median_genes_per_cell": (
        "Median genes per cell", "Median Genes per Cell",
    ),
    "median_umis_per_cell": (
        "Median UMI counts per cell", "Median UMIs per cell",
    ),
    "mean_reads_per_cell": (
        "Mean reads per cell", "Mean Reads per Cell", "Median reads per cell",
    ),
    "pct_sequencing_saturation": (
        "Sequencing saturation", "Sequencing Saturation",
    ),
    # --- guide / CRISPR library ---
    "crispr_total_reads": (
        "CRISPR Total input reads", "CRISPR Number of reads", "CRISPR Total reads",
    ),
    "crispr_mean_reads_per_cell": (
        "CRISPR Mean reads per cell", "CRISPR Median reads per cell",
    ),
    "crispr_estimated_cells": (
        "CRISPR Estimated number of cells", "CRISPR Number of passing cells",
    ),
    "crispr_pct_reads_in_cells": (
        "CRISPR fraction valid reads in cells",
        "CRISPR Fraction of reads in passing cells",
    ),
    "crispr_pct_cells_with_guide": (
        "CRISPR Fraction of cells with at least one guide",
        "Cells with one or more protospacer detected",
    ),
    # --- hashtag / antibody library ---
    "hto_total_reads": ("HTO Total input reads", "Antibody Number of reads"),
    # --- guide/feature barcode read accounting (v1.2.1) ---
    # These were among the 49 names the MDL-1856 run reported as unrecognised
    # and dropped. They are exactly the metrics that say whether guide and
    # hashtag capture worked, which is what the guide and hashtag sections of
    # this report are for.
    "crispr_reads_missing_barcodes": ("CRISPR reads missing barcodes",),
    "crispr_reads_corrected_barcodes": ("CRISPR reads with corrected barcodes",),
    "crispr_reads_exact_barcodes": (
        "CRISPR reads with exactly matching barcodes",
    ),
    "crispr_reads_nonmatching_barcodes": (
        "CRISPR reads with non-matching barcodes",
    ),
    "crispr_tag_mapping_rate": ("CRISPR tag mapping rate",),
    "feature_matching_reads": ("Feature matching reads",),
    "feature_nonmatching_reads": ("Feature non-matching reads",),
    "feature_reads_missing_barcodes": ("Feature reads missing barcodes",),
    "feature_reads_corrected_barcodes": (
        "Feature reads with corrected barcodes",
    ),
    "feature_reads_exact_barcodes": (
        "Feature reads with exactly matching barcodes",
    ),
    "hto_mean_reads_per_cell": (
        "HTO Mean reads per cell", "Antibody Mean reads per cell",
    ),
    "hto_pct_reads_in_cells": (
        "HTO fraction valid reads in cells",
        "Antibody Fraction of reads in passing cells",
    ),
}

_ALIAS_LOOKUP: dict[str, str] = {}
for canonical, aliases in METRIC_ALIASES.items():
    for alias in aliases:
        _ALIAS_LOOKUP[_norm_key(alias)] = canonical

# Metrics whose natural unit is a percentage.  DRAGEN reports some of these in
# a separate "pct" column and some as the value itself; both are handled.
PERCENT_METRICS = {
    "pct_mapped_reads", "pct_reads_in_cells", "pct_sequencing_saturation",
    "crispr_pct_reads_in_cells", "crispr_pct_cells_with_guide",
    "hto_pct_reads_in_cells",
}

# Candidate filename patterns, most specific first.
METRIC_FILE_PATTERNS: tuple[str, ...] = (
    "{prefix}.scRNA_metrics.csv",
    "{prefix}.scrna_metrics.csv",
    "{prefix}.metrics.csv",
    "{prefix}_metrics.csv",
    "metrics_summary.csv",
    "*.scRNA_metrics.csv",
    "*metrics_summary.csv",
)


@dataclass
class SeqMetrics:
    """Tidy sequencing metrics: one row per (sample, prefix, metric)."""

    long: pd.DataFrame            # sample, prefix, metric, value, pct, raw_name
    files: dict[str, Path]        # prefix -> file actually read
    missing: list[str]            # prefixes with no metrics file
    unmatched: list[str]          # raw metric names we could not map

    @property
    def empty(self) -> bool:
        return self.long.empty

    def wide(self) -> pd.DataFrame:
        """One row per (sample, prefix), one column per canonical metric."""
        if self.long.empty:
            return pd.DataFrame()
        df = self.long.copy()
        # For percentage metrics prefer the dedicated pct column when the value
        # column holds a raw count.
        df["use"] = np.where(
            df["metric"].isin(PERCENT_METRICS) & df["pct"].notna(),
            df["pct"], df["value"],
        )
        out = (
            df.pivot_table(
                index=["sample", "prefix"], columns="metric", values="use",
                aggfunc="first",
            )
            .reset_index()
        )
        out.columns.name = None
        return out

    def derived(self) -> pd.DataFrame:
        """Wide metrics plus ratios computed where the vendor didn't report them."""
        w = self.wide()
        if w.empty:
            return w
        w = w.copy()

        def col(name: str) -> pd.Series:
            return (
                pd.to_numeric(w[name], errors="coerce")
                if name in w.columns
                else pd.Series(np.nan, index=w.index)
            )

        # Mean reads per cell, if absent, from total reads / passing cells.
        for pref, total, cells in (
            ("", "total_reads", "estimated_cells"),
            ("crispr_", "crispr_total_reads", "crispr_estimated_cells"),
            ("hto_", "hto_total_reads", "estimated_cells"),
        ):
            target = f"{pref}mean_reads_per_cell"
            have = col(target)
            fallback = pd.Series(
                safe_divide(col(total).to_numpy(), col(cells).to_numpy()),
                index=w.index,
            )
            w[target] = have.where(have.notna(), fallback)

        # UMI validity and barcode mapping rates
        w["pct_valid_umi"] = pd.Series(
            100.0
            * safe_divide(
                col("reads_with_valid_umi").to_numpy(),
                col("total_barcoded_reads").to_numpy(),
            ),
            index=w.index,
        )
        w["pct_barcoded"] = pd.Series(
            100.0
            * safe_divide(
                col("total_barcoded_reads").to_numpy(), col("total_reads").to_numpy()
            ),
            index=w.index,
        )
        return w


def _parse_metric_file(path: Path) -> pd.DataFrame:
    """Parse one metrics CSV into (raw_name, value, pct) rows.

    Handles both shapes seen in practice:

      DRAGEN:      section,prefix,metric,value[,pct]     (no header)
      CellRanger:  Metric Name,Metric Value              (with header)
    """
    try:
        raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc

    if raw.empty:
        return pd.DataFrame(columns=["raw_name", "value", "pct"])

    ncol = raw.shape[1]
    rows: list[dict[str, Any]] = []

    # Detect a CellRanger-style 2-column header row.
    first = [str(x).strip().lower() for x in raw.iloc[0].tolist()]
    header_like = ncol == 2 and any("metric" in c for c in first)
    body = raw.iloc[1:] if header_like else raw

    for _, r in body.iterrows():
        cells = [str(x).strip() for x in r.tolist()]
        if ncol >= 4:
            # section, prefix, metric, value, [pct]
            name, value = cells[2], cells[3]
            pct = cells[4] if ncol >= 5 else ""
        elif ncol == 3:
            name, value, pct = cells[1], cells[2], ""
        elif ncol == 2:
            name, value, pct = cells[0], cells[1], ""
        else:
            continue
        if not name:
            continue
        rows.append({"raw_name": name, "value": value, "pct": pct})

    df = pd.DataFrame(rows, columns=["raw_name", "value", "pct"])
    if df.empty:
        return df
    # Strip thousands separators and percent signs before coercion.
    for c in ("value", "pct"):
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _find_metric_file(dragen_path: Path, prefix: str) -> Path | None:
    if not dragen_path.exists():
        return None
    for pattern in METRIC_FILE_PATTERNS:
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


def load_sequencing_metrics(runs: Sequence[dict[str, Any]]) -> SeqMetrics:
    """Load metrics for every (sample, prefix, dragen_path) triple.

    A missing directory or file is recorded, not raised: sequencing metrics are
    a nice-to-have for the report, and an experiment whose DRAGEN output has
    been archived should still get a full transcriptome/guide/HTO analysis.
    The report says which prefixes were missing.
    """
    frames: list[pd.DataFrame] = []
    files: dict[str, Path] = {}
    missing: list[str] = []
    unmatched: set[str] = set()

    for run in runs:
        prefix = str(run["prefix"])
        path = Path(run["dragen_path"])
        found = _find_metric_file(path, prefix)
        if found is None:
            missing.append(prefix)
            continue
        try:
            df = _parse_metric_file(found)
        except ValueError:
            missing.append(prefix)
            continue
        if df.empty:
            missing.append(prefix)
            continue
        files[prefix] = found
        df["metric"] = df["raw_name"].map(lambda s: _ALIAS_LOOKUP.get(_norm_key(s)))
        unmatched.update(df.loc[df["metric"].isna(), "raw_name"].tolist())
        df = df.dropna(subset=["metric"])
        df["sample"] = str(run["sample"])
        df["prefix"] = prefix
        frames.append(df[["sample", "prefix", "metric", "value", "pct", "raw_name"]])

    long = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["sample", "prefix", "metric", "value", "pct", "raw_name"]
        )
    )
    # Duplicate (prefix, metric) pairs: keep the first and say so, rather than
    # the original's silent "last row wins".
    if not long.empty:
        long = long.drop_duplicates(subset=["sample", "prefix", "metric"], keep="first")

    return SeqMetrics(
        long=long, files=files, missing=missing, unmatched=sorted(unmatched)
    )
