"""
Sample-manifest reading, validation and (safe) threshold write-back.

ONE implementation, used by the pipeline, the CLI and the report builder.  The
original had two near-verbatim copies of the delimiter-sniffing parser -- one
in ``run_pipeline.py`` and one in ``build_qc_report.py`` -- with a comment
claiming the duplication was deliberate to avoid coupling, even though
``run_pipeline`` already ``exec_module``'d the report script directly.  They
were free to drift, and did.

Two correctness issues from the original are fixed here:

1. **Header keys were not stripped.**  ``csv.DictReader`` keys keep their
   original spacing, so a manifest whose header reads ``" sample"`` passed the
   ``"sample" in fieldnames`` membership check (fieldnames *were* stripped) but
   then ``row.get("sample")`` returned ``None`` -- silently reporting zero
   samples.  Here the rows themselves are re-keyed on stripped names.

2. **The manifest was rewritten in place with no backup.**  Any ragged row or
   whitespace-padded header raised ``ValueError`` from ``DictWriter``
   *mid-write*, truncating the user's source-of-truth file.  Write-back is now
   atomic (temp file + replace) and takes a timestamped backup first.
"""
from __future__ import annotations

import csv
import datetime as _dt
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .config import THRESHOLD_KEYS, QCThresholds

# Values that mean "blank" in a hand-edited spreadsheet export.  The original
# only recognised "" and "nan", so a cell containing "NA" or "None" was
# treated as a real threshold value.
BLANK_TOKENS = {
    "", "na", "n/a", "nan", "none", "null", "nat", "-", "--", ".", "?",
}

# Columns that must be present, non-blank, and identical on every row: they
# describe the experiment, not the sample.
GLOBAL_COLUMNS = ("h5ad_path", "output_path")

# Optional pointers to the two whitelist files. Like h5ad_path they describe
# the experiment, so they must be identical (or blank) on every row -- but
# unlike h5ad_path they are not required, so they are validated separately.
#
# They exist because two things cannot be recovered from the data:
#   * which guide population ("family") a guide belongs to, which determines
#     which NTC cells are a valid control for it;
#   * which hashtag combinations are a real design rather than a doublet.
WHITELIST_COLUMNS = ("grna_whitelist", "hashtag_whitelist")

# Nominates which metadata columns are the comparisons that matter.
#
# Autodetection takes any column with 2..max_compare_levels distinct values,
# and when more than three qualify it kept the three with the FEWEST levels --
# an arbitrary rule that could quietly drop the comparison the experiment was
# designed around. This column lets the manifest say so explicitly. Value is a
# ``|``-separated list of column names, identical on every row.
CONDITION_COLUMN = "condition_columns"

# ...and the names people actually write. `condition_columns` was invented for
# this feature; the obvious thing to type is `condition`, and a manifest that
# says `condition = fixation` plainly means "compare fixation" -- it should not
# be ignored because it used the shorter word.
#
# `condition` is ambiguous, though: it is equally plausible as a column holding
# a condition VALUE ("untreated"/"TGFb"). Disambiguated by asking whether the
# values name real columns of this manifest. If they do, it is nominating other
# columns; if they do not, it is ordinary metadata and left alone.
CONDITION_COLUMN_ALIASES = (
    CONDITION_COLUMN, "condition_column", "compare_columns",
    "conditions", "condition", "priority_conditions",
)

# Columns that identify a sequencing run within a sample.
RUN_COLUMNS = ("prefix", "dragen_path")

# Declares whether one sample's multiple dragen runs (multiple RUN_COLUMNS
# rows sharing one `sample`) are multiple library preparations built from the
# SAME cells/cDNA, or independent runs/pools that happen to share one row in
# this manifest. This is not recoverable from the data: DRAGEN's
# cellhashing.tsv is written over (close to) the full combinatorial barcode
# whitelist, so a shared barcode across two runs looks identical on disk
# whether it means "the same cell, resequenced" (sum the counts) or
# "coincidence between two unrelated pools" (do not sum -- see hto_dragen.py).
# Getting this wrong changes reported hashtag counts by up to Nx, where N is
# the number of runs for that sample, so it is required whenever a sample has
# more than one run rather than defaulted either way.
#
# Excluded from metadata/condition-column autodetection below: it is a
# declaration about library preparation, not a biological condition, even
# though (unlike h5ad_path) its value can legitimately differ between samples
# in the same manifest -- some samples might be straightforward single-run
# experiments while others in the same manifest are split across several
# library preps of the same cells.
DRAGEN_RUNS_SHARE_CELLS_COLUMN = "dragen_runs_share_cells"

SAMPLE_COL_CANDIDATES = ("sample", "sample_id", "Sample", "library")


class ManifestError(Exception):
    """Raised for any manifest problem that must stop the run.

    Deliberately raised (not ``sys.exit``) so the CLI owns exit codes and the
    library stays importable/testable.
    """


@dataclass
class Manifest:
    """A validated sample manifest."""

    path: Path
    df: pd.DataFrame
    delimiter: str
    fieldnames: list[str]
    sample_col: str
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ accessors
    @property
    def samples(self) -> list[str]:
        return list(pd.unique(self.df[self.sample_col].astype(str)))

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_runs(self) -> int:
        return len(self.df)

    def global_value(self, column: str) -> str:
        """The single value of a global column (validated identical on all rows)."""
        if column not in self.df.columns:
            # read_manifest(required_columns=...) can validly produce a Manifest
            # where a GLOBAL_COLUMNS entry is entirely absent (the caller is
            # overriding it some other way, e.g. --output-path). Reaching here
            # anyway means something tried to use this column's value despite
            # that -- a real caller bug, but still a ManifestError rather than
            # a raw pandas KeyError, since callers already catch the former.
            raise ManifestError(
                f"Column {column!r} is not present in {self.path.name}, and "
                f"nothing overrode it, so its value cannot be read."
            )
        vals = {
            str(v).strip()
            for v in self.df[column]
            if not is_blank(v)
        }
        if len(vals) != 1:
            raise ManifestError(
                f"Column {column!r} must have one identical non-blank value on every "
                f"row of {self.path.name}; found {sorted(vals) or ['nothing']}."
            )
        return vals.pop()

    @property
    def h5ad_path(self) -> Path:
        """Resolved absolute path to the .h5ad.

        Relative values resolve against the *manifest's own directory*, not the
        caller's working directory.  The original resolved against CWD, which
        broke its own advertised "location-independent" design the moment you
        ran it from anywhere else.
        """
        return self._resolve(self.global_value("h5ad_path"))

    @property
    def output_path(self) -> Path:
        return self._resolve(self.global_value("output_path"))

    # ----------------------------------------------------------- whitelists
    def _optional_global_path(self, column: str) -> Path | None:
        """Resolved path from an optional global column, or None if absent/blank.

        Raises if the column holds more than one distinct non-blank value: two
        different whitelists for one experiment is a mistake we should not
        silently pick a winner for.
        """
        if column not in self.df.columns:
            return None
        vals = {str(v).strip() for v in self.df[column] if not is_blank(v)}
        if not vals:
            return None
        if len(vals) > 1:
            raise ManifestError(
                f"Column {column!r} in {self.path.name} must name one file for the "
                f"whole experiment; found {sorted(vals)}."
            )
        p = self._resolve(vals.pop())
        if not p.exists():
            raise ManifestError(
                f"{column} points to {p}, which does not exist. Paths resolve "
                f"relative to the manifest's own directory ({self.path.parent})."
            )
        return p

    def _parse_nominating(self, col: str) -> list[str] | None:
        """Names listed in ``col``, or None if it is not nominating columns."""
        if col not in self.df.columns:
            return None
        vals = {str(v).strip() for v in self.df[col] if not is_blank(v)}
        if not vals:
            return None
        if len(vals) > 1:
            if col == CONDITION_COLUMN:
                raise ManifestError(
                    f"Column {col!r} in {self.path.name} must be identical on "
                    f"every row; found {sorted(vals)}."
                )
            return None            # varies per row -> ordinary metadata
        names = [c.strip() for c in re.split(r"[|,;]", vals.pop()) if c.strip()]
        if not names:
            return None
        unknown = [c for c in names if c not in self.df.columns]
        if unknown:
            if col == CONDITION_COLUMN:
                # The canonical name is unambiguous, so a value that is not a
                # column is a typo and must not be silently ignored.
                #
                # Listing the available columns from df.columns directly, NOT
                # via metadata_columns(): that calls nominating_column(), which
                # calls back into here, which recurses until the stack dies.
                available = [
                    c for c in self.df.columns
                    if c not in set(GLOBAL_COLUMNS) | set(WHITELIST_COLUMNS)
                    | set(RUN_COLUMNS) | set(THRESHOLD_KEYS)
                    | set(CONDITION_COLUMN_ALIASES) | {self.sample_col}
                ]
                raise ManifestError(
                    f"{col} names column(s) {unknown} that are not in "
                    f"{self.path.name}. Available metadata columns: "
                    f"{available}."
                )
            return None            # an alias whose values are not column names
        return names

    def nominating_column(self) -> str | None:
        """Which column, if any, is naming the conditions to compare."""
        for col in CONDITION_COLUMN_ALIASES:
            if self._parse_nominating(col):
                return col
        return None

    def declared_condition_columns(self) -> list[str]:
        """Condition columns the manifest explicitly nominates, in order.

        Empty when no nominating column is present, in which case
        autodetection applies.
        """
        for col in CONDITION_COLUMN_ALIASES:
            names = self._parse_nominating(col)
            if names:
                return names
        return []

    @property
    def grna_whitelist_path(self) -> Path | None:
        return self._optional_global_path("grna_whitelist")

    @property
    def hashtag_whitelist_path(self) -> Path | None:
        return self._optional_global_path("hashtag_whitelist")

    def _resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.path.parent / p).resolve()

    @property
    def experiment_id(self) -> str:
        """Best-effort experiment identifier for report titles.

        Looks for an ``ABC1234``-style token in the manifest filename, then in
        the output path, then falls back to the output directory name.
        """
        for candidate in (self.path.name, str(self.output_path)):
            m = re.search(r"([A-Z]{2,5}[-_]?\d{3,6})", candidate)
            if m:
                return m.group(1)
        return Path(self.output_path).name or "experiment"

    def runs_for_sample(self, sample: str) -> pd.DataFrame:
        return self.df[self.df[self.sample_col].astype(str) == str(sample)]

    def metadata_columns(self) -> list[str]:
        """Columns that describe experimental conditions.

        Everything that is not a path, a threshold, a run identifier or the
        sample column.  Determined from the file, not from a hardcoded list --
        the original report hardcoded ``[gRNA_method, acoh,
        resuspension_buffer, fixation, HTO, cell_type]`` in one module while
        its docstring promised the opposite, so a new condition column was
        invisible to the report until someone edited the source.
        """
        exclude = (
            set(GLOBAL_COLUMNS)
            | set(WHITELIST_COLUMNS)
            | set(RUN_COLUMNS)
            | set(THRESHOLD_KEYS)
            | {self.sample_col, DRAGEN_RUNS_SHARE_CELLS_COLUMN}
        )
        # Only the column actually nominating conditions is excluded. An alias
        # like `condition` that holds real values ("untreated"/"TGFb") stays
        # metadata, which is the whole point of the ambiguity check.
        nominating = self.nominating_column()
        if nominating:
            exclude.add(nominating)
        return [c for c in self.df.columns if c not in exclude]

    def condition_columns(self, max_levels: int = 12) -> list[str]:
        """Metadata columns worth using as comparison axes.

        A column qualifies when it has between 2 and ``max_levels`` distinct
        non-blank values across samples.  Single-valued columns describe the
        experiment (nothing to compare) and very-high-cardinality columns are
        identifiers, not conditions.
        """
        out = []
        per_sample = self.df.drop_duplicates(subset=[self.sample_col])
        for col in self.metadata_columns():
            vals = {
                str(v).strip() for v in per_sample[col] if not is_blank(v)
            }
            if 2 <= len(vals) <= max_levels:
                out.append(col)
        return out

    def condition_label(self, sample: str, columns: Sequence[str]) -> str:
        """Human-readable condition label for one sample, e.g. "PCR | CSB | HTO+"."""
        rows = self.runs_for_sample(sample)
        if rows.empty:
            return str(sample)
        parts = []
        for col in columns:
            if col not in rows.columns:
                continue
            vals = {str(v).strip() for v in rows[col] if not is_blank(v)}
            if vals:
                parts.append("/".join(sorted(vals)))
        return " | ".join(parts) if parts else str(sample)

    def sample_metadata_frame(self) -> pd.DataFrame:
        """One row per sample, metadata columns only, for joining onto obs."""
        cols = [self.sample_col] + self.metadata_columns()
        out = self.df[cols].drop_duplicates(subset=[self.sample_col]).copy()
        out[self.sample_col] = out[self.sample_col].astype(str)
        return out.set_index(self.sample_col)

    def dragen_runs(self) -> list[dict[str, Any]]:
        """(sample, prefix, dragen_path) triples with a usable dragen_path."""
        if not {"prefix", "dragen_path"} <= set(self.df.columns):
            return []
        out = []
        for _, r in self.df.iterrows():
            if is_blank(r.get("dragen_path")) or is_blank(r.get("prefix")):
                continue
            out.append(
                {
                    "sample": str(r[self.sample_col]),
                    "prefix": str(r["prefix"]).strip(),
                    "dragen_path": self._resolve(str(r["dragen_path"]).strip()),
                }
            )
        return out

    def total_cell_input(self) -> pd.Series | None:
        """Loaded cell input per sample, if the manifest tracks it."""
        if "cell_input" not in self.df.columns:
            return None
        d = self.df.drop_duplicates(subset=[self.sample_col])
        s = pd.to_numeric(d["cell_input"], errors="coerce")
        s.index = d[self.sample_col].astype(str)
        return s.dropna() if s.notna().any() else None

    def dragen_runs_share_cells(self, sample: str) -> bool | None:
        """Whether SAMPLE's multiple dragen runs are the same cells/cDNA.

        ``None`` means undeclared. Unlike ``declares_hto``, this is not
        allowed to fall back to a guess: DRAGEN's cellhashing.tsv looks
        identical on disk whether a barcode shared between two runs means
        "the same cell, resequenced" (sum) or "coincidence between two
        unrelated pools" (do not sum), and the two interpretations disagree
        by up to Nx on every reported hashtag count. Callers with more than
        one run for a sample and ``None`` here must refuse to auto-recover
        hashtags for that sample rather than pick a default -- see
        ``hto_dragen.build_hto_modality_multi_sample``.

        Irrelevant, and not required, for a sample with only one dragen run.
        """
        col = DRAGEN_RUNS_SHARE_CELLS_COLUMN
        if col not in self.df.columns:
            return None
        rows = self.runs_for_sample(sample)
        if col not in rows.columns or rows.empty:
            return None
        vals = {str(v).strip().lower() for v in rows[col] if not is_blank(v)}
        if not vals:
            return None
        yes = vals & {"yes", "y", "true", "1"}
        no = vals & {"no", "n", "false", "0"}
        if yes and no:
            raise ManifestError(
                f"{col!r} has conflicting values for sample {sample!r} in "
                f"{self.path.name}: {sorted(vals)}. It must be one "
                f"consistent yes/no across every run row for a given sample."
            )
        if yes:
            return True
        if no:
            return False
        raise ManifestError(
            f"{col!r} for sample {sample!r} in {self.path.name} has "
            f"unrecognised value(s) {sorted(vals)}; use yes/no."
        )

    def declares_hto(self) -> bool | None:
        """Whether the manifest says this experiment used hashtags.

        Returns None when the manifest is silent, in which case detection falls
        back to what is actually in the h5ad.  Manifest and data are
        cross-checked in ``pipeline`` and a disagreement is reported rather
        than silently resolved -- "the manifest says HTO=yes but no hashtag
        matrix was found" is exactly the kind of thing a QC report should say
        out loud.
        """
        for col in ("HTO", "hto", "hashtag", "HTO_used"):
            if col in self.df.columns:
                vals = {str(v).strip().lower() for v in self.df[col] if not is_blank(v)}
                if not vals:
                    return None
                if vals & {"yes", "y", "true", "1"}:
                    return True
                if vals & {"no", "n", "false", "0"}:
                    return False
        return None

    # -------------------------------------------------------- threshold I/O
    def read_thresholds(self) -> QCThresholds:
        """QC thresholds stored in the manifest, if any.

        Each threshold column must be blank or hold one identical value across
        all rows -- a per-row threshold would be meaningless because filtering
        is applied globally to the concatenated matrix.
        """
        th = QCThresholds()
        for key in THRESHOLD_KEYS:
            if key not in self.df.columns:
                continue
            vals = {
                str(v).strip() for v in self.df[key] if not is_blank(v)
            }
            if not vals:
                continue
            if len(vals) > 1:
                raise ManifestError(
                    f"Threshold column {key!r} in {self.path.name} has conflicting "
                    f"values {sorted(vals)}. QC filtering is applied globally, so "
                    f"each threshold column must be blank or hold one value."
                )
            raw = vals.pop()
            try:
                setattr(th, key, float(raw))
            except ValueError:
                raise ManifestError(
                    f"Threshold column {key!r} in {self.path.name} contains "
                    f"{raw!r}, which is not a number."
                ) from None
            th.source[key] = "manifest"
        return th

    def write_thresholds(
        self, thresholds: QCThresholds, backup: bool = True
    ) -> Path | None:
        """Persist the thresholds actually used back into the manifest.

        This keeps the record of "what was filtered" with the experiment rather
        than only in a shell command someone has to remember -- the genuinely
        good idea from the original pipeline, kept.

        Unlike the original, the write is atomic and backed up, so a failure
        cannot leave the user with a half-written manifest.
        """
        return self._rewrite(
            {k: v for k, v in thresholds.as_dict().items() if v is not None},
            backup=backup,
        )

    def ensure_threshold_columns(self, backup: bool = True) -> Path | None:
        """Add blank threshold columns so they can be filled in by hand."""
        missing = [k for k in THRESHOLD_KEYS if k not in self.df.columns]
        if not missing:
            return None
        return self._rewrite({k: None for k in missing}, backup=backup)

    def _rewrite(self, updates: dict[str, Any], backup: bool) -> Path | None:
        if not updates:
            return None
        backup_path = None
        if backup and self.path.exists():
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = self.path.with_suffix(self.path.suffix + f".bak-{stamp}")
            shutil.copy2(self.path, backup_path)

        df = self.df.copy()
        for col, value in updates.items():
            df[col] = "" if value is None else value

        # Preserve original column order, appending genuinely new columns.
        ordered = [c for c in self.fieldnames if c in df.columns]
        ordered += [c for c in df.columns if c not in ordered]
        df = df[ordered]

        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".manifest-", suffix=".tmp"
        )
        try:
            import os

            os.close(tmp_fd)
            df.to_csv(
                tmp_name, sep=self.delimiter, index=False, encoding="utf-8", na_rep=""
            )
            Path(tmp_name).replace(self.path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            if backup_path is not None:
                shutil.copy2(backup_path, self.path)
            raise
        self.fieldnames = ordered
        self.df = df
        return backup_path


# ===========================================================================
# Parsing
# ===========================================================================
def is_blank(value: Any) -> bool:
    """True for anything a human would consider an empty manifest cell."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in BLANK_TOKENS


def _sniff(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    """Read the file with whichever delimiter actually yields a real table.

    ``csv.Sniffer`` is not reliable on real manifest exports (a single comma
    inside a quoted cell-type field like "HeLa/MCF10A" is enough to fool it),
    so each plausible delimiter is tried and scored on whether it produces
    multiple columns including a recognisable sample column.
    """
    best: tuple[int, list[dict[str, str]], list[str], str] | None = None
    for delim in (",", "\t", ";", "|"):
        try:
            with open(path, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh, delimiter=delim)
                raw_fields = reader.fieldnames or []
                fields_ = [f.strip() for f in raw_fields if f is not None]
                rows: list[dict[str, str]] = []
                for raw in reader:
                    # Re-key on stripped header names.  This is the fix for the
                    # silent zero-samples bug in the original.
                    clean = {}
                    for k, v in raw.items():
                        if k is None:
                            continue        # ragged extra fields -> restkey
                        clean[k.strip()] = "" if v is None else v
                    rows.append(clean)
        except (OSError, csv.Error):
            continue

        if len(fields_) < 2:
            continue
        score = len(fields_) * 10 + len(rows)
        if any(c in fields_ for c in SAMPLE_COL_CANDIDATES):
            score += 1000
        if "h5ad_path" in fields_:
            score += 500
        if best is None or score > best[0]:
            best = (score, rows, fields_, delim)

    if best is None:
        raise ManifestError(
            f"Could not parse {path} as a delimited table with at least two "
            f"columns. Tried comma, tab, semicolon and pipe."
        )
    return best[1], best[2], best[3]


def read_table(
    path: str | Path, what: str = "table"
) -> tuple[pd.DataFrame, str, list[str]]:
    """Read any delimited side-car file the same way manifests are read.

    Reuses ``_sniff`` so the whitelists inherit the manifest's delimiter
    detection and, more importantly, its header-stripping: a file whose header
    reads ``" family"`` must not silently produce a column nobody can find.
    """
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"{what} not found: {p}")
    rows, fields_, delim = _sniff(p)
    if not rows:
        raise ManifestError(f"{what} {p.name} has a header but no data rows.")
    df = pd.DataFrame(rows, columns=fields_)
    return df, delim, fields_


def read_manifest(
    path: str | Path,
    require_paths: bool = True,
    strict: bool = False,
    required_columns: Sequence[str] | None = None,
) -> Manifest:
    """Read and validate a sample manifest.

    Parameters
    ----------
    require_paths
        Enforce that the columns in ``required_columns`` (``h5ad_path`` and
        ``output_path`` by default) are present, non-blank and identical on
        every row.  Turn off only for report-only rebuilds.
    strict
        Promote warnings (partially-filled metadata columns, per-sample
        metadata disagreements) into errors.
    required_columns
        Which of ``GLOBAL_COLUMNS`` to actually enforce; defaults to both.
        A caller that is going to override one of them anyway (``--h5ad``,
        ``--output-path``) can drop it from this list, since the tool never
        reads the corresponding manifest column once an override is given --
        see ``cli.py``'s ``main()``, which does exactly that. The column
        stays excluded from metadata/condition-column autodetection either
        way (that exclusion is unconditional -- see ``metadata_columns``),
        so this only affects whether its absence raises.

    Raises
    ------
    ManifestError
        With a message naming the file and the specific problem.  Validation is
        fail-fast and happens *before* the expensive h5ad load, which is the
        one design decision from the original worth keeping verbatim.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise ManifestError(f"Manifest not found: {p}")
    if p.is_dir():
        raise ManifestError(f"Manifest path is a directory, not a file: {p}")

    rows, fieldnames, delimiter = _sniff(p)
    if not rows:
        raise ManifestError(f"Manifest {p.name} has a header but no data rows.")

    df = pd.DataFrame(rows, columns=fieldnames)
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    sample_col = next((c for c in SAMPLE_COL_CANDIDATES if c in df.columns), None)
    if sample_col is None:
        raise ManifestError(
            f"Manifest {p.name} needs a sample column (one of "
            f"{', '.join(SAMPLE_COL_CANDIDATES)}); found columns: "
            f"{', '.join(map(str, df.columns))}"
        )

    warnings_: list[str] = []

    if require_paths:
        cols_to_require = GLOBAL_COLUMNS if required_columns is None else required_columns
        for col in cols_to_require:
            if col not in df.columns:
                raise ManifestError(
                    f"Manifest {p.name} is missing required column {col!r}. "
                    f"It must appear once, with the same value on every row."
                )
            vals = {str(v).strip() for v in df[col] if not is_blank(v)}
            if not vals:
                raise ManifestError(
                    f"Column {col!r} in {p.name} is blank on every row."
                )
            if len(vals) > 1:
                raise ManifestError(
                    f"Column {col!r} in {p.name} must be identical on every row "
                    f"(it describes the experiment, not the sample); found "
                    f"{sorted(vals)}."
                )
            if df[col].map(is_blank).any():
                warnings_.append(
                    f"{col!r} is blank on some rows of {p.name}; using {vals!r} "
                    f"for all rows."
                )

    # Sanity-check run identifiers
    if "prefix" in df.columns:
        prefixes = [str(v).strip() for v in df["prefix"] if not is_blank(v)]
        dupes = {x for x in prefixes if prefixes.count(x) > 1}
        if dupes:
            warnings_.append(
                f"Duplicate prefix value(s) {sorted(dupes)} in {p.name}; "
                f"sequencing metrics for these will collide."
            )

    # Metadata completeness: blank-for-all is "untracked" and fine; blank for
    # only some rows means someone forgot to fill it in.
    threshold_cols = set(THRESHOLD_KEYS)
    for col in df.columns:
        if (
            col in GLOBAL_COLUMNS
            or col in WHITELIST_COLUMNS
            or col in CONDITION_COLUMN_ALIASES
            or col in threshold_cols
            or col == sample_col
        ):
            continue
        blanks = df[col].map(is_blank)
        if blanks.any() and not blanks.all():
            msg = (
                f"Column {col!r} in {p.name} is filled on "
                f"{int((~blanks).sum())}/{len(df)} rows. Blank-everywhere means "
                f"'not tracked'; partially blank usually means a typo or an "
                f"unfinished edit."
            )
            if strict:
                raise ManifestError(msg)
            warnings_.append(msg)

    # Per-sample metadata should agree across that sample's runs.
    for col in df.columns:
        if (
            col in GLOBAL_COLUMNS
            or col in WHITELIST_COLUMNS
            or col in CONDITION_COLUMN_ALIASES
            or col in threshold_cols
            or col == sample_col
        ):
            continue
        if col in RUN_COLUMNS:
            continue
        grouped = df.groupby(df[sample_col].astype(str))[col].nunique(dropna=True)
        bad = grouped[grouped > 1]
        if len(bad):
            msg = (
                f"Column {col!r} in {p.name} differs between runs of the same "
                f"sample ({', '.join(map(str, bad.index))}). It is a per-sample "
                f"property, so this is probably an error."
            )
            if strict:
                raise ManifestError(msg)
            warnings_.append(msg)

    if "cell_input" in df.columns:
        coerced = pd.to_numeric(df["cell_input"], errors="coerce")
        bad = df["cell_input"].notna() & coerced.isna()
        if bad.any():
            warnings_.append(
                f"{int(bad.sum())} row(s) of {p.name} have a non-numeric "
                f"'cell_input'; cell-retention efficiency will be skipped for those."
            )

    return Manifest(
        path=p,
        df=df,
        delimiter=delimiter,
        fieldnames=list(fieldnames),
        sample_col=sample_col,
        warnings=warnings_,
    )


def write_manifest_template(
    path: str | Path,
    samples: Iterable[str] | None = None,
    metadata_columns: Sequence[str] = (
        "gRNA_method", "resuspension_buffer", "fixation", "HTO", "cell_type",
        "cell_input",
    ),
) -> Path:
    """Write a blank manifest skeleton for a new experiment."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = (
        ["sample", "h5ad_path", "output_path"]
        + list(metadata_columns)
        + ["prefix", "dragen_path"]
        + list(THRESHOLD_KEYS)
    )
    sample_ids = list(samples) if samples is not None else ["sample_1"]
    if not sample_ids:
        sample_ids = ["sample_1"]
    rows = [{c: "" for c in cols} for _ in sample_ids]
    for row, s in zip(rows, sample_ids):
        row["sample"] = s
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    return p
