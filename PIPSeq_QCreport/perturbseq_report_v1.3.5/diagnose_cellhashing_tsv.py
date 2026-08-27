#!/usr/bin/env python
"""
What is actually in a DRAGEN `*.scRNA.cellhashing.tsv`, and does it line up
with the barcodes in the h5ad?

This pipeline needs to inject hashtag counts from the DRAGEN output whenever
the h5ad doesn't already carry a hashtag matrix (see hto.py / modalities.py --
``split_modalities`` currently has no path that reads from
``dragen_output/*.cellhashing.tsv`` at all). Before writing that loader against
a guess about the file's layout, this script establishes the actual layout on
a real file: orientation (barcodes as rows or as columns), which column/row is
the barcode, which are hashtag counts, and whether those barcodes actually
match the h5ad's ``obs_names`` or need a transform first (a ``-1`` suffix
added/stripped, a sample prefix, etc).

Read-only. Nothing is written.

Usage
-----
Preferred: point it at the sample manifest (the same file the pipeline reads).
It resolves ``h5ad_path`` and every run's ``dragen_path``/``prefix`` from
there, locates each run's cellhashing file the same way the pipeline will, and
checks every one against the manifest's h5ad in one pass:

    python diagnose_cellhashing_tsv.py --manifest /path/to/sample_manifest.csv
    python diagnose_cellhashing_tsv.py --manifest /path/to/sample_manifest.csv --sample MDL1856_1_1

Ad hoc, against one file directly (no manifest required):

    python diagnose_cellhashing_tsv.py /path/to/PH20260504_1_1.scRNA.cellhashing.tsv
    python diagnose_cellhashing_tsv.py /path/to/....cellhashing.tsv --h5ad /path/to/object.h5ad
    python diagnose_cellhashing_tsv.py /path/to/....cellhashing.tsv --barcodes-file barcodes.txt

Sections (per file)
--------------------
A  RAW FILE          delimiter, encoding, first lines exactly as they appear on
                      disk -- before pandas has interpreted anything
B  PARSE + ORIENT     load it, decide which axis is barcodes vs hashtags, and
                      whether the first row/column is a header/index or data
C  HASHTAG COLUMNS    per-hashtag summary: dtype, min/max, %zero -- a hashtag
                      column should look like the count columns you already
                      have from a working in-matrix hashtag
D  BARCODE SANITY     format of the barcode strings themselves: length,
                      alphabet, presence/absence of a "-N" suffix, duplicates
E  BARCODE MATCH      overlap against the target h5ad's obs_names, and what
                      overlap you'd get after common transforms (strip suffix,
                      add suffix, take prefix before "-"/"_", case fold)

In ``--manifest`` mode, a final SUMMARY table lists every run with whether its
file was found, the orientation, the best transform, and what fraction of the
h5ad's cells it covers -- the thing to paste back before any loader code is
trusted against this experiment's data.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

BARCODE_RE = re.compile(r"^[ACGTN]{6,40}(-\d+)?$", re.IGNORECASE)

# Kept in sync with ModalityConfig.hto_dragen_file_patterns in
# perturbseq_report/config.py; duplicated here (rather than imported) so this
# script keeps working standalone even if perturbseq_report isn't importable
# from wherever it's copied to on ICA. --manifest mode imports the real config
# instead and uses whatever patterns are actually configured.
DEFAULT_FILE_PATTERNS = (
    "{prefix}.scRNA.cellhashing.tsv",
    "{prefix}.scRNA.cellhashing.tsv.gz",
    "{prefix}.cellhashing.tsv",
    "{prefix}_cellhashing.tsv",
    "*.scRNA.cellhashing.tsv",
    "*cellhashing.tsv",
)


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ===========================================================================
# A. raw file
# ===========================================================================
def sniff_raw(path: Path, n_lines: int = 5) -> str:
    banner("A. RAW FILE")
    print(f"path: {path}")
    if not path.exists():
        print("!! FILE DOES NOT EXIST AT THIS PATH.")
        raise SystemExit(1)
    size = path.stat().st_size
    print(f"size: {size:,} bytes")

    with open(path, "rb") as fh:
        head = fh.read(4096)
    try:
        text = head.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = head.decode("latin-1")
        encoding = "latin-1 (NOT utf-8 -- unusual for a DRAGEN text output)"
    print(f"encoding (best guess from first 4KB): {encoding}")

    lines = text.splitlines()[:n_lines]
    print(f"\nfirst {len(lines)} line(s), raw (whitespace visible as repr):")
    for i, ln in enumerate(lines):
        print(f"  [{i}] {ln!r}")

    # Delimiter sniff off the first non-empty line.
    sample = "\n".join(lines) or text
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,; |")
        delim = dialect.delimiter
        print(f"\ncsv.Sniffer delimiter guess: {delim!r}")
    except csv.Error:
        # Fall back to counting occurrences of the usual suspects.
        counts = {d: sample.count(d) for d in ("\t", ",", ";", "|")}
        delim = max(counts, key=counts.get)
        print(f"\ncsv.Sniffer failed; falling back to character counts {counts} "
              f"-> guessing {delim!r}")
    if delim != "\t":
        print(
            "!! NOTE: filename says .tsv but the sniffed delimiter is not a tab. "
            "Double check this is really tab-separated and not comma/space "
            "padded to look tab-like."
        )
    return delim


# ===========================================================================
# B. parse + orient
# ===========================================================================
def load_candidates(path: Path, delim: str) -> dict[str, pd.DataFrame]:
    """Every plausible way to read this file, so the wrong guess is visible."""
    out: dict[str, pd.DataFrame] = {}
    for label, kwargs in (
        ("header=0, index_col=0", dict(header=0, index_col=0)),
        ("header=0, index_col=None", dict(header=0, index_col=None)),
        ("header=None, index_col=0", dict(header=None, index_col=0)),
        ("header=None, index_col=None", dict(header=None, index_col=None)),
    ):
        try:
            out[label] = pd.read_csv(path, sep=delim, **kwargs)
        except Exception as exc:                        # pragma: no cover
            print(f"  (parse attempt {label!r} failed: {exc})")
    return out


def _looks_like_barcodes(values) -> float:
    """Fraction of values that look like a 10x-style cell barcode."""
    vals = [str(v) for v in values]
    if not vals:
        return 0.0
    hits = sum(1 for v in vals if BARCODE_RE.match(v))
    return hits / len(vals)


def _numeric_fraction(series: pd.Series) -> float:
    coerced = pd.to_numeric(series, errors="coerce")
    return float(coerced.notna().mean()) if len(series) else 0.0


def parse_and_orient(path: Path, delim: str) -> tuple[pd.DataFrame, str, str]:
    banner("B. PARSE + ORIENT")
    candidates = load_candidates(path, delim)
    if not candidates:
        print("!! Could not parse the file under any header/index_col combination.")
        raise SystemExit(1)

    header0 = candidates.get("header=0, index_col=0")
    print("Read with header=0, index_col=0 (most likely DRAGEN layout):")
    if header0 is not None:
        print(f"  shape: {header0.shape}")
        print(f"  index name: {header0.index.name!r}, first index values: "
              f"{list(header0.index[:5])}")
        print(f"  columns: {list(header0.columns[:12])}"
              f"{' ...' if header0.shape[1] > 12 else ''}")

    # Decide orientation: does the row index or the column header look like
    # barcodes?
    row_frac = _looks_like_barcodes(header0.index[:200]) if header0 is not None else 0.0
    col_frac = _looks_like_barcodes(header0.columns[:200]) if header0 is not None else 0.0
    print(f"\nfraction of row index matching barcode pattern: {row_frac:.2f}")
    print(f"fraction of column names matching barcode pattern: {col_frac:.2f}")

    if header0 is None:
        print("!! Falling back to header=None parse.")
        df = candidates["header=None, index_col=None"]
        orientation, barcode_axis = "unknown", "unknown"
    elif row_frac >= 0.5 and row_frac >= col_frac:
        print("-> orientation: barcodes are ROWS (one row per cell, one column "
              "per hashtag). This is the layout the loader should assume by "
              "default.")
        df, orientation, barcode_axis = header0, "barcodes_as_rows", "index"
    elif col_frac >= 0.5:
        print("-> orientation: barcodes are COLUMNS (one row per hashtag, one "
              "column per cell) -- TRANSPOSED relative to the default "
              "assumption. The loader will need `.T` or an explicit transpose "
              "flag.")
        df, orientation, barcode_axis = header0, "barcodes_as_columns", "columns"
    else:
        print("-> orientation UNCLEAR: neither the row index nor the column "
              "names look like 10x barcodes (16-20bp of ACGT, optionally "
              "'-1'). Inspect the raw lines in section A by eye -- this may be "
              "a file with a non-standard first column (e.g. a sample or lane "
              "ID) that needs to be handled specially.")
        df, orientation, barcode_axis = header0, "unclear", "index"

    return df, orientation, barcode_axis


# ===========================================================================
# C. hashtag columns
# ===========================================================================
def describe_hashtag_columns(df: pd.DataFrame, orientation: str) -> pd.DataFrame:
    banner("C. HASHTAG COLUMNS")
    mat = df.T if orientation == "barcodes_as_columns" else df

    rows = []
    for col in mat.columns:
        s = mat[col]
        numeric_frac = _numeric_fraction(s)
        coerced = pd.to_numeric(s, errors="coerce")
        rows.append(
            {
                "column": col,
                "numeric_fraction": round(numeric_frac, 3),
                "dtype": str(s.dtype),
                "min": coerced.min(),
                "max": coerced.max(),
                "mean": round(float(coerced.mean()), 2) if coerced.notna().any() else float("nan"),
                "pct_zero": round(100.0 * float((coerced == 0).mean()), 1) if coerced.notna().any() else float("nan"),
                "n_missing": int(coerced.isna().sum()),
            }
        )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    non_numeric = out.loc[out["numeric_fraction"] < 0.9, "column"].tolist()
    if non_numeric:
        print(
            f"\n!! {len(non_numeric)} column(s) are mostly NOT numeric: "
            f"{non_numeric}. If these are meant to be hashtag count columns, "
            f"something upstream (a stray text row, a total/summary row mixed "
            f"into the data, a units suffix) is corrupting them. If instead "
            f"one of these is a metadata column (e.g. a 'total' or 'unmapped' "
            f"column DRAGEN appends), the loader must exclude it explicitly "
            f"rather than treating every non-barcode column as a hashtag."
        )
    looks_like_counts = out.loc[
        (out["numeric_fraction"] >= 0.9)
        & (out["min"].fillna(0) >= 0)
        & (out["max"].fillna(0) == out["max"].fillna(0).round()),
    ]
    print(
        f"\n{len(looks_like_counts)} of {len(out)} columns look like plausible "
        f"non-negative integer count columns (candidate hashtags)."
    )
    return out


# ===========================================================================
# D. barcode sanity
# ===========================================================================
def barcode_sanity(df: pd.DataFrame, orientation: str, barcode_axis: str) -> pd.Index:
    banner("D. BARCODE SANITY")
    barcodes = (
        pd.Index(df.columns) if barcode_axis == "columns" else pd.Index(df.index)
    )
    barcodes = barcodes.astype(str)
    print(f"n barcodes in file: {len(barcodes):,}")
    print(f"n unique: {barcodes.nunique():,}")
    dup = len(barcodes) - barcodes.nunique()
    if dup:
        print(f"!! {dup} duplicate barcode(s) in the cellhashing file -- these "
              f"will silently collide (last-one-wins, or worse) whatever join "
              f"key is used downstream. Examples: "
              f"{barcodes[barcodes.duplicated()][:5].tolist()}")

    sample = list(barcodes[:10])
    print(f"first 10 barcodes: {sample}")

    lengths = pd.Series(barcodes.str.len())
    print(f"length distribution: min={lengths.min()}, max={lengths.max()}, "
          f"most common={lengths.mode().tolist()}")

    has_suffix = barcodes.str.contains(r"-\d+$").mean()
    print(f"fraction with a '-N' suffix (e.g. '-1'): {has_suffix:.2f}")

    non_acgt = barcodes[~barcodes.str.match(r"^[ACGTNacgtn]+(-\d+)?$")]
    if len(non_acgt):
        print(
            f"!! {len(non_acgt)} barcode(s) contain characters outside A/C/G/T/N "
            f"and an optional '-N' suffix, e.g. {list(non_acgt[:5])}. If these "
            f"look like 'SAMPLE_ACGT...-1', the loader needs to strip a sample "
            f"prefix before matching against the h5ad."
        )
    return barcodes


# ===========================================================================
# E. barcode match against the h5ad (or a plain barcode list)
# ===========================================================================
def _decode_h5_strings(raw) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def _read_obs_index_h5py(obs) -> list[str]:
    """Read obs_names from an h5py obs group across anndata's on-disk encodings.

    Anndata has changed how obs/index is stored across versions (a plain
    string dataset; a categorical group with ``categories``/``codes``; the
    index name recorded as an attribute that may or may not point at a real
    key), and an environment's h5py can read a file written by a newer/older
    anndata than the one installed. This tries every layout actually seen in
    practice rather than assuming one.
    """
    import h5py

    candidate_keys: list[str] = []
    raw_attr = obs.attrs.get("_index")
    if raw_attr is not None:
        candidate_keys.append(
            raw_attr.decode() if isinstance(raw_attr, bytes) else str(raw_attr)
        )
    candidate_keys += ["_index", "index", "obs_names"]
    # de-dupe, keep order
    seen = set()
    candidate_keys = [k for k in candidate_keys if not (k in seen or seen.add(k))]

    diagnostics: list[str] = []
    for key in candidate_keys:
        if key not in obs:
            diagnostics.append(f"{key!r}: not present under obs")
            continue
        node = obs[key]
        try:
            if isinstance(node, h5py.Dataset):
                return _decode_h5_strings(node[:])
            if isinstance(node, h5py.Group):
                enc = node.attrs.get("encoding-type")
                enc = enc.decode() if isinstance(enc, bytes) else enc
                subkeys = list(node.keys())
                # Categorical encoding: categories[codes], codes == -1 is NaN.
                if "categories" in node and "codes" in node:
                    cats = _decode_h5_strings(node["categories"][:])
                    codes = node["codes"][:]
                    return [cats[c] if c >= 0 else "" for c in codes]
                # Nullable/extension-array encoding (numpy_nullable /
                # pandas "string" dtype, pyarrow-backed columns, etc.):
                # values (+ optional boolean mask marking missing entries).
                if "values" in node:
                    values = _decode_h5_strings(node["values"][:])
                    if "mask" in node:
                        mask = node["mask"][:]
                        values = [
                            "" if m else v for v, m in zip(values, mask)
                        ]
                    return values
                diagnostics.append(
                    f"{key!r}: present as a group but not a recognised "
                    f"encoding (encoding-type={enc!r}, keys={subkeys})"
                )
                continue
            diagnostics.append(
                f"{key!r}: present but neither a Dataset nor a Group "
                f"({type(node)!r})"
            )
        except Exception as exc:                          # pragma: no cover
            diagnostics.append(f"{key!r}: raised {exc!r} while reading")

    # Some older anndata files store obs as one structured (compound-dtype)
    # dataset rather than a group of per-column datasets.
    if isinstance(obs, h5py.Dataset) and obs.dtype.names:
        for key in candidate_keys:
            if key in obs.dtype.names:
                return _decode_h5_strings(obs[key])

    raise ValueError(
        "could not locate/decode the obs index dataset. Per-candidate-key "
        "detail:\n    " + "\n    ".join(diagnostics)
    )


def _read_obs_column_h5py(obs, key: str):
    """Generic version of _read_obs_index_h5py for an arbitrary obs column.

    Returns ``(values, error)``: ``values`` is a numpy object array (or None
    on failure) and ``error`` is a human-readable reason when it is None.
    """
    import h5py

    if key not in obs:
        return None, f"{key!r} not present in obs"
    node = obs[key]
    try:
        if isinstance(node, h5py.Dataset):
            return np.asarray(_decode_h5_strings(node[:]), dtype=object), None
        if isinstance(node, h5py.Group):
            if "categories" in node and "codes" in node:
                cats = _decode_h5_strings(node["categories"][:])
                codes = node["codes"][:]
                return np.array(
                    [cats[c] if c >= 0 else None for c in codes], dtype=object
                ), None
            if "values" in node:
                values = node["values"][:]
                out = np.asarray(values, dtype=object)
                if "mask" in node:
                    mask = np.asarray(node["mask"][:], dtype=bool)
                    out = np.where(mask, None, out)
                return out, None
            return None, (
                f"{key!r} is a group with an unrecognised encoding "
                f"(encoding-type={node.attrs.get('encoding-type')!r}, "
                f"keys={list(node.keys())})"
            )
    except Exception as exc:                               # pragma: no cover
        return None, f"{key!r} raised {exc!r} while reading"
    return None, f"{key!r}: unexpected node type {type(node)!r}"


def _as_bool(values) -> np.ndarray:
    out = np.zeros(len(values), dtype=bool)
    for i, v in enumerate(values):
        if v is None:
            continue
        if isinstance(v, bytes):
            v = v.decode()
        if isinstance(v, str):
            out[i] = v.strip().lower() in ("true", "1", "yes", "t", "pass")
        else:
            try:
                out[i] = bool(v)
            except Exception:
                out[i] = False
    return out


def check_lane_assignment(
    h5ad: str, obs_names: pd.Index, prefixes: list[str]
) -> dict[str, np.ndarray] | None:
    """Whether obs already records, per cell, which sequencing run it came from.

    This matters because DRAGEN's cellhashing.tsv is written per run over
    (something close to) the FULL combinatorial barcode whitelist -- millions
    of rows, most of them ambient/noise reads rather than real cells -- so the
    same 28bp barcode string can legitimately appear as a row in more than one
    run's file. A loader that matches purely on barcode string across a pooled
    multi-lane object risks pulling one lane's noise counts for another lane's
    real cell whenever their barcodes coincide. If obs already has a per-run
    'pass_<prefix>' flag (as this experiment's obs keys suggest), that is the
    authoritative source of which run a cell belongs to, and hashtag counts
    should be pulled from that cell's OWN run's file only.
    """
    banner("F. RUN/LANE ASSIGNMENT SANITY (pass_<prefix> columns)")
    try:
        import h5py
    except ImportError:
        print("h5py not available in this environment -- skipping (obs_names "
              "were presumably loaded via anndata instead).")
        return None

    candidate_cols = [f"pass_{p}" for p in prefixes]
    with h5py.File(Path(h5ad), "r") as f:
        obs = f["obs"]
        present = [c for c in candidate_cols if c in obs]
        if not present:
            print(
                f"None of {candidate_cols} found in obs -- cannot check whether "
                f"cells are already disambiguated by run. If no such column "
                f"exists at all, the loader will need another way to know "
                f"which run's cellhashing.tsv a given cell belongs to (a "
                f"sample/lane column, or matching against each run's own "
                f"barcodes.tsv.gz filtered list instead of cellhashing.tsv)."
            )
            return None
        cols: dict[str, np.ndarray] = {}
        for c in present:
            values, err = _read_obs_column_h5py(obs, c)
            if values is None:
                print(f"!! could not read {c!r}: {err}")
                continue
            cols[c] = _as_bool(values)

    if not cols:
        print("No pass_<prefix> column could be decoded; see errors above.")
        return None

    n = len(obs_names)
    counts_per_cell = np.zeros(n, dtype=int)
    for c, mask in cols.items():
        n_true = int(mask.sum())
        print(f"{c}: {n_true:,} of {n:,} cells True ({100 * n_true / n:.1f}%)")
        counts_per_cell += mask.astype(int)

    dist = pd.Series(counts_per_cell).value_counts().sort_index()
    print(f"\nnumber of pass_<prefix> columns True, per cell:\n{dist.to_string()}")
    n_zero = int((counts_per_cell == 0).sum())
    n_multi = int((counts_per_cell > 1).sum())
    if n_multi:
        print(
            f"\n!! {n_multi:,} cell(s) are True in MORE THAN ONE run's "
            f"pass_<prefix> column. Either these columns don't mean "
            f"'this cell's barcode belongs to this run', or some cells are "
            f"legitimately shared/reprocessed across runs -- do not assume "
            f"per-run disjointness until this is explained."
        )
    if n_zero:
        print(
            f"\n{n_zero:,} cell(s) are False in every pass_<prefix> column "
            f"checked -- either these columns don't cover every run in the "
            f"manifest, or those cells' run identity comes from elsewhere."
        )
    if not n_multi and not n_zero:
        print(
            "\nEvery cell is True in EXACTLY ONE pass_<prefix> column -- this "
            "is a reliable per-cell run/lane assignment. The loader should use "
            "it: for each cell, pull hashtag counts only from the "
            "cellhashing.tsv of the run whose pass_<prefix> column is True for "
            "that cell, rather than matching barcodes against every run's "
            "file indiscriminately."
        )
    else:
        print(
            "\nA majority of cells being True in MULTIPLE (often ALL) "
            "pass_<prefix> columns, with pct-True close to the same fraction "
            "for every run, is what you'd see if these four runs are not four "
            "different barcode pools but four sequencing runs/lanes of the "
            "SAME combinatorial library -- i.e. the same physical cell's "
            "barcode is genuinely expected to appear in every run's data, "
            "with per-run counts reflecting that run's own read depth for it. "
            "Section G below checks that directly by comparing actual counts "
            "for the same barcode across all four cellhashing.tsv files."
        )

    return cols


def _grep_barcode_line(path: Path, barcode: str) -> str | None:
    """Exact-match one barcode's row without loading the whole file.

    Anchored on a tab after the barcode so a barcode that is a prefix of
    another (shouldn't happen at fixed length, but cheap to guard) can't
    match the wrong row.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["grep", "-m", "1", "-F", f"{barcode}\t", str(path)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout.splitlines()[0]
        return None
    except Exception:
        return None


def compare_counts_across_runs(
    obs_names: pd.Index,
    cols: dict[str, np.ndarray],
    runs: list[dict],
    n_per_group: int = 4,
) -> None:
    """For a handful of barcodes, print actual hashtag counts from EVERY run.

    Section F can only say a barcode's pass_<prefix> flags overlap across
    runs; it can't say whether that's because the same physical cell was
    genuinely resequenced in every run (expected, real signal in every run)
    or because of coincidental barcode reuse across otherwise-unrelated runs
    (expected, near-zero/noise counts in the runs that aren't the real one).
    This is the check that tells those two apart, on real data rather than by
    assumption.
    """
    banner("G. HASHTAG COUNTS FOR THE SAME BARCODE, ACROSS EVERY RUN")
    keys = list(cols.keys())
    if len(keys) < 2:
        print("(skipped -- need at least two pass_<prefix> columns to compare)")
        return

    bits = np.stack([cols[k] for k in keys], axis=1)
    total_true = bits.sum(axis=1)
    rng = np.random.default_rng(0)

    def sample(mask: np.ndarray, n: int) -> list[int]:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return []
        return sorted(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())

    groups = {
        f"all {len(keys)} runs True": sample(total_true == len(keys), n_per_group),
        "no runs True": sample(total_true == 0, n_per_group),
        "some but not all True": sample(
            (total_true > 0) & (total_true < len(keys)), n_per_group
        ),
    }

    obs_arr = np.asarray(obs_names, dtype=str)
    run_by_prefix = {r["prefix"]: r for r in runs}
    prefixes = [r["prefix"] for r in runs]

    # Resolve each run's actual cellhashing file once.
    files: dict[str, Path | None] = {}
    for r in runs:
        found = _find_cellhashing_file_local(
            Path(r["dragen_path"]), r["prefix"], DEFAULT_FILE_PATTERNS
        )
        files[r["prefix"]] = found
        print(f"{r['prefix']}: {found if found else '(no file found)'}")

    for group_name, idxs in groups.items():
        if not idxs:
            print(f"\n[{group_name}] -- no cells in this group, skipping")
            continue
        print(f"\n[{group_name}]")
        for i in idxs:
            bc = obs_arr[i]
            flags = "".join("1" if bits[i, j] else "0" for j in range(len(keys)))
            print(f"  barcode={bc}  pass-flags({','.join(keys)})={flags}")
            for prefix in prefixes:
                path = files.get(prefix)
                if path is None:
                    print(f"    {prefix}: (file not found)")
                    continue
                line = _grep_barcode_line(path, bc)
                if line is None:
                    print(f"    {prefix}: barcode not present in this run's file")
                    continue
                parts = line.rstrip("\n").split("\t")
                try:
                    total = sum(int(x) for x in parts[1:])
                except ValueError:
                    total = None
                print(f"    {prefix}: counts={parts[1:]}  total={total}")

    print(
        "\nInterpretation: if the 'all runs True' barcodes show comparable, "
        "non-trivial total counts in EVERY run, that supports summing counts "
        "for the same barcode across all runs for one sample (same library "
        "resequenced). If instead only one run has real counts and the "
        "others are near-zero for the same barcode, that supports treating "
        "each run's counts as belonging to a run-specific cell and NOT "
        "summing across runs. Send this section back before any combining "
        "logic is written."
    )


def load_target_barcodes(h5ad: str | None, barcodes_file: str | None) -> pd.Index | None:
    if barcodes_file:
        p = Path(barcodes_file)
        vals = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        print(f"loaded {len(vals):,} barcodes from {p}")
        return pd.Index(vals).astype(str)

    if h5ad:
        p = Path(h5ad)
        try:
            import anndata as ad
            a = ad.read_h5ad(p, backed="r")
            idx = pd.Index([str(x) for x in a.obs.index]).astype(str)
            print(f"loaded {len(idx):,} obs_names from {p} via anndata (backed='r')")
            return idx
        except Exception as exc:
            print(f"anndata read failed ({exc}); trying h5py directly...")
        try:
            import h5py
            with h5py.File(p, "r") as f:
                obs = f["obs"]
                print(f"obs keys on disk: {list(obs.keys()) if hasattr(obs, 'keys') else '(not a group)'}")
                print(f"obs attrs: {dict(obs.attrs)}")
                raw = _read_obs_index_h5py(obs)
                idx = pd.Index([str(x) for x in raw]).astype(str)
                print(f"loaded {len(idx):,} obs_names from {p} via h5py")
                return idx
        except Exception as exc:
            print(f"!! Could not read obs_names from {p} via anndata or h5py: {exc}")
            print(
                "   This is a read robustness gap in THIS SCRIPT, not necessarily "
                "a problem with the h5ad -- if anndata is installed in this "
                "environment, try `import anndata; anndata.read_h5ad(path).obs.index` "
                "directly and share what it prints (version mismatch, missing "
                "layer, etc.) so this can be fixed."
            )
            return None
    return None


def barcode_match_report(barcodes: pd.Index, target: pd.Index) -> pd.DataFrame:
    """Overlap of ``barcodes`` against ``target`` under several transforms.

    Returns the table, sorted best-first, with both ``pct_of_file_barcodes``
    (how much of THIS file matched) and ``pct_of_target`` (how much of the
    h5ad's cells this recovers) -- the second number is what actually matters
    for deciding whether the fallback is worth using.
    """
    source = pd.Index(barcodes).astype(str)
    target_set = set(target.astype(str))
    transforms = {
        "as-is": source,
        "strip '-N' suffix": source.str.replace(r"-\d+$", "", regex=True),
        "add '-1' suffix (if none present)": source.where(
            source.str.contains(r"-\d+$"), source + "-1"
        ),
        "prefix before first '-'": source.str.split("-").str[0],
        "prefix before first '_'": source.str.split("_").str[0],
        "suffix after last '_'": source.str.split("_").str[-1],
        "uppercase": source.str.upper(),
    }
    rows = []
    for label, transformed in transforms.items():
        overlap = len(set(transformed) & target_set)
        rows.append(
            {
                "transform": label,
                "n_matched": overlap,
                "pct_of_file_barcodes": round(100.0 * overlap / len(source), 1)
                if len(source) else 0.0,
                "pct_of_target": round(100.0 * overlap / len(target_set), 1)
                if target_set else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("n_matched", ascending=False).reset_index(drop=True)


def print_barcode_match(barcodes: pd.Index, target: pd.Index | None) -> pd.DataFrame | None:
    banner("E. BARCODE MATCH AGAINST TARGET")
    if target is None or len(target) == 0:
        print("(skipped -- no target barcodes available)")
        return None
    print(f"target barcode examples: {list(target[:10])}")
    print(f"target barcode count: {len(target):,}, unique: {pd.Index(target).nunique():,}")
    out = barcode_match_report(barcodes, target)
    print(out.to_string(index=False))
    best = out.iloc[0]
    if best["n_matched"] == 0:
        print(
            "\n!! NONE of the tried transforms produced any overlap at all. "
            "The two barcode sets may come from genuinely different cell "
            "populations (e.g. this cellhashing.tsv is for a different "
            "sample/lane than the h5ad), or the barcode alphabet/casing is "
            "doing something not covered above -- look at the raw examples in "
            "section D and the target's examples above by eye."
        )
    elif best["pct_of_target"] < 50:
        print(
            f"\n!! Best transform ({best['transform']!r}) only recovers "
            f"{best['pct_of_target']:.1f}% of the h5ad's cells. Partial overlap "
            f"this low usually means the two files are from different "
            f"pools/lanes rather than a fixable string-format mismatch."
        )
    else:
        print(
            f"\n-> use transform: {best['transform']!r} "
            f"({best['pct_of_target']:.1f}% of h5ad cells covered)"
        )
    return out


def run_one_file(path: Path, target: pd.Index | None, n_lines: int) -> dict:
    delim = sniff_raw(path, n_lines)
    df, orientation, barcode_axis = parse_and_orient(path, delim)
    describe_hashtag_columns(df, orientation)
    barcodes = barcode_sanity(df, orientation, barcode_axis)
    match = print_barcode_match(barcodes, target)
    best = match.iloc[0] if match is not None and len(match) else None
    return {
        "file": str(path),
        "delimiter": delim,
        "orientation": orientation,
        "n_barcodes": len(barcodes),
        "best_transform": best["transform"] if best is not None else None,
        "pct_target_matched": best["pct_of_target"] if best is not None else None,
    }


# ===========================================================================
# --manifest mode
# ===========================================================================
def _find_cellhashing_file_local(dragen_path: Path, prefix: str, patterns) -> Path | None:
    """Standalone file finder -- does not depend on anything added to the
    installed perturbseq_report package, only on the manifest's own
    (sample, prefix, dragen_path) triples, so this script works against
    whatever version of the package happens to be deployed."""
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


def run_manifest_mode(
    manifest_path: str, sample_filter: str | None, n_lines: int, all_runs: bool = False,
) -> None:
    # Only perturbseq_report.manifest is used, and only for what this script
    # already confirmed works against a real deployment: parsing the manifest
    # and resolving h5ad_path / dragen_runs(). Deliberately NOT importing
    # ModalityConfig or hto_dragen here -- those may not exist (or may not have
    # the hto_dragen_* fields yet) in whatever copy of the package is actually
    # installed at this path, and this script must run standalone regardless.
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    try:
        from perturbseq_report.manifest import ManifestError, read_manifest
    except ImportError as exc:
        print(
            f"!! Could not import perturbseq_report.manifest from {repo_root} "
            f"({exc}). Run this script from inside the perturbseq_report "
            f"package directory (or copy it there), or use the ad hoc "
            f"single-file mode instead: "
            f"`diagnose_cellhashing_tsv.py <file.tsv> [--h5ad ...]`."
        )
        raise SystemExit(1)

    banner("MANIFEST")
    try:
        manifest = read_manifest(manifest_path, require_paths=True)
    except ManifestError as exc:
        print(f"!! Could not read manifest: {exc}")
        raise SystemExit(1)

    print(f"manifest: {manifest.path}")
    print(f"samples ({manifest.n_samples}): {manifest.samples}")
    print(f"h5ad_path: {manifest.h5ad_path}  (exists: {manifest.h5ad_path.exists()})")

    runs = manifest.dragen_runs()
    if not runs:
        print(
            "\n!! The manifest has no usable 'prefix'/'dragen_path' columns "
            "(RUN_COLUMNS in manifest.py), so no run's dragen_output directory "
            "can be located. Add those columns, pointing at each run's "
            "dragen_output directory, and re-run."
        )
        raise SystemExit(1)

    if sample_filter:
        runs = [r for r in runs if r["sample"] == sample_filter]
        if not runs:
            print(f"\n!! No dragen runs for sample {sample_filter!r}.")
            raise SystemExit(1)

    print(f"\n{len(runs)} dragen run(s) declared for the selected sample(s):")
    for r in runs:
        print(f"  sample={r['sample']!r}  prefix={r['prefix']!r}  "
              f"dragen_path={r['dragen_path']}")

    all_declared_runs = list(runs)
    if not all_runs and len(runs) > 1:
        skipped = runs[1:]
        runs = runs[:1]
        print(
            f"\nChecking only the first run ({runs[0]['prefix']!r}) -- "
            f"{len(skipped)} other run(s) for this sample "
            f"({', '.join(r['prefix'] for r in skipped)}) are skipped. Every "
            f"run for one sample shares the same cellhashing.tsv layout and "
            f"hashtag panel, so one run's structure and barcode-match result "
            f"is representative. Pass --all-runs to check every run anyway."
        )

    banner("LOADING h5ad OBS_NAMES (once, shared by every run below)")
    target = load_target_barcodes(str(manifest.h5ad_path), None)
    if target is None:
        print(
            "!! Could not load obs_names from the manifest's h5ad_path. "
            "Section E (barcode match) will be skipped for every run below; "
            "sections A-D (file structure) still run."
        )
    else:
        # Checked against ALL declared runs for this sample, not just the one
        # (or ones) whose cellhashing.tsv is parsed below -- this is cheap
        # (obs columns only) and answers a question section E cannot: DRAGEN's
        # cellhashing.tsv is written over (close to) the full combinatorial
        # barcode whitelist, so the same barcode string can appear as a row in
        # more than one run's file. If a per-cell 'pass_<prefix>' column
        # already says which run each cell truly belongs to, the loader must
        # use that instead of matching barcodes against every run's file
        # indiscriminately.
        try:
            lane_cols = check_lane_assignment(
                str(manifest.h5ad_path), target,
                [r["prefix"] for r in all_declared_runs],
            )
        except Exception as exc:                          # pragma: no cover
            print(f"!! Section F could not run: {exc!r}")
            lane_cols = None

        if lane_cols:
            try:
                compare_counts_across_runs(target, lane_cols, all_declared_runs)
            except Exception as exc:                       # pragma: no cover
                print(f"!! Section G could not run: {exc!r}")

    patterns = DEFAULT_FILE_PATTERNS

    summary_rows = []
    for r in runs:
        banner(f"RUN: sample={r['sample']!r} prefix={r['prefix']!r}")
        dpath = Path(r["dragen_path"])
        print(f"dragen_path: {dpath}  (exists: {dpath.exists()})")
        if dpath.exists() and dpath.is_dir():
            listing = sorted(p.name for p in dpath.iterdir())
            shown = listing[:40]
            print(f"directory contents ({len(listing)} total"
                  f"{', showing first 40' if len(listing) > 40 else ''}): {shown}")
            hashy = [n for n in listing if "hash" in n.lower()]
            if hashy:
                print(f"  (name(s) containing 'hash': {hashy})")

        found = _find_cellhashing_file_local(dpath, r["prefix"], patterns)

        if found is None:
            print(
                f"\n!! No cellhashing file found under {dpath} for prefix "
                f"{r['prefix']!r}. Patterns tried: {list(patterns)}. Check the "
                f"directory listing above for the actual filename -- if it "
                f"doesn't match any pattern (e.g. a different suffix or "
                f"casing), tell me the real name and I'll add it to the "
                f"pattern list."
            )
            summary_rows.append({
                "sample": r["sample"], "prefix": r["prefix"], "file_found": None,
                "orientation": None, "n_barcodes": None,
                "best_transform": None, "pct_target_matched": None,
            })
            continue

        print(f"\nfound: {found}")
        result = run_one_file(found, target, n_lines)
        summary_rows.append({
            "sample": r["sample"], "prefix": r["prefix"],
            "file_found": Path(result["file"]).name,
            "orientation": result["orientation"],
            "n_barcodes": result["n_barcodes"],
            "best_transform": result["best_transform"],
            "pct_target_matched": result["pct_target_matched"],
        })

    banner("SUMMARY ACROSS ALL RUNS")
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    if summary["file_found"].isna().any():
        print(
            "\n!! One or more runs had no cellhashing file located -- see the "
            "per-run section above for the directory listing and patterns "
            "tried."
        )
    if target is None:
        print(
            "\n!! Barcode-match (section E) could not run for any run because "
            "obs_names could not be loaded from the h5ad -- install anndata "
            "or h5py in this environment, or re-run with the ad hoc single-"
            "file mode and --barcodes-file instead."
        )
    else:
        pct = pd.to_numeric(summary["pct_target_matched"], errors="coerce")
        if (pct.fillna(0) < 50).any():
            print(
                "\n!! One or more runs matched under 50% of the h5ad's cells "
                "even under the best transform tried -- do not wire this file "
                "in as the hashtag source for that run until that's understood "
                "(see section E for that run, above)."
            )
    print(
        "\nSend this SUMMARY table (and, for any run flagged above, its full "
        "per-run output) back so the loader in perturbseq_report can be "
        "confirmed or corrected against the real layout before it's trusted."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cellhashing_tsv", nargs="?", default=None,
                     help="Path to one *.cellhashing.tsv file (ad hoc mode; "
                          "omit this and use --manifest instead when possible)")
    ap.add_argument("--manifest", default=None,
                     help="Sample manifest (same file/format the pipeline "
                          "reads). Resolves h5ad_path and every run's "
                          "dragen_path/prefix automatically -- the preferred "
                          "way to run this.")
    ap.add_argument("--sample", default=None,
                     help="With --manifest, restrict to one sample")
    ap.add_argument("--all-runs", action="store_true",
                     help="With --manifest, check every run instead of just "
                          "the first one for the selected sample(s)")
    ap.add_argument("--h5ad", default=None,
                     help="Ad hoc mode only: h5ad to compare barcodes against")
    ap.add_argument("--barcodes-file", default=None,
                     help="Ad hoc mode only: plain text file of barcodes, one "
                          "per line, to compare against instead of an h5ad")
    ap.add_argument("--n-lines", type=int, default=5,
                     help="Number of raw lines to print in section A")
    args = ap.parse_args()

    if args.manifest:
        run_manifest_mode(args.manifest, args.sample, args.n_lines, args.all_runs)
        return

    if not args.cellhashing_tsv:
        ap.error("pass either a cellhashing.tsv path or --manifest")

    path = Path(args.cellhashing_tsv)
    target = load_target_barcodes(args.h5ad, args.barcodes_file)
    result = run_one_file(path, target, args.n_lines)

    banner("SUMMARY")
    print(f"file: {result['file']}")
    print(f"delimiter: {result['delimiter']!r}")
    print(f"orientation: {result['orientation']}")
    if result["best_transform"]:
        print(f"best barcode transform: {result['best_transform']!r} "
              f"({result['pct_target_matched']:.1f}% of target matched)")
    print(
        "\nSend the full output of this script back so the loader in "
        "perturbseq_report can be written against the confirmed layout "
        "(barcode axis, which columns are hashtags, and any barcode "
        "transform needed to match the h5ad)."
    )


if __name__ == "__main__":
    main()
