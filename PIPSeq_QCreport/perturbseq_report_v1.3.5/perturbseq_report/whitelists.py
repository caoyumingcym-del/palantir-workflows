"""
Guide and hashtag whitelists (v1.2.0).

Two things about a Perturb-seq experiment cannot be recovered from the data,
and v1.1.0 guessed at both:

1. **Which guide population a guide belongs to.** An experiment with four cell
   lines and four gRNA libraries has four distinct NTC populations, and a
   guide-positive cell is only comparable against NTC cells from its own.
   v1.1.0 pooled all 60 NTC guides in MDL-1856 (26,056 cells) into a single
   baseline for every knockdown, E-distance and DE comparison. The family
   cannot be inferred from the ID -- ``NTC_10_ACGT...`` carries no library
   information, and two libraries can independently use ``NTC_10``.

2. **Which hashtag combinations are a design rather than a doublet.** v1.1.0
   hardcoded "1 positive = singlet, >=2 = multiplet" and graded MDL-1856's
   57.3% multiplet rate as ``poor``, when combinatorial tagging means some of
   those cells are exactly what was intended.

Both files are optional. Without them the pipeline falls back to v1.1.0
behaviour and says so in the report, rather than pretending it knows.

Design notes that are easy to get wrong:

* ``family`` is deliberately NOT "cell line". One cell line screened with two
  libraries is two families; two cell lines screened with one library is also
  two families. It is an opaque ID for "the set of guides whose NTCs control
  for each other". ``cell_line`` and ``library`` are ordinary metadata columns.
* In the hashtag whitelist, uniqueness is on ``hashtag_set``, NOT on
  ``demux_id``. One biological sample may be split into aliquots tagged
  differently -- one combinatorially, one singly -- and both must resolve to
  the same sample.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .manifest import BLANK_TOKENS, ManifestError, is_blank, read_table

# Columns the loaders understand. Anything else in the file is carried through
# as metadata, so a lab can keep its own bookkeeping columns in the same sheet.
GUIDE_WHITELIST_REQUIRED = ("guide_id", "family")
GUIDE_WHITELIST_KNOWN = GUIDE_WHITELIST_REQUIRED + (
    "parse_target_from", "target_gene", "target_ensg", "role", "control_of",
    "short_label",
)
HASHTAG_WHITELIST_REQUIRED = ("sample", "demux_id", "hashtag_set")
HASHTAG_WHITELIST_KNOWN = HASHTAG_WHITELIST_REQUIRED + (
    "aliquot", "family",
)

VALID_ROLES = ("targeting", "ntc", "matched_control", "safe_harbour")
VALID_PARSE_FROM = ("first", "second")

_ENSG_RE = re.compile(r"^ENS[A-Z]*G\d+$")
_HASHTAG_SEP = "+"

# Explicit "this sample carried no hashtag on purpose" tokens for hashtag_set.
#
# Deliberately does NOT include "none", "na" or "-": those are in BLANK_TOKENS
# and mean "not filled in". An intentionally untagged sample and a cell someone
# forgot to fill in must not look the same, because the whole point of
# declaring it is to distinguish design from omission.
UNTAGGED_TOKENS = frozenset({
    "untagged", "unhashed", "no_hashtag", "no_hashtags", "nohashtag",
    "unlabelled", "unlabeled",
})


def _blank_to_empty(v: object) -> str:
    """Normalise a spreadsheet cell to a plain string, blanks to ''."""
    return "" if is_blank(v) else str(v).strip()


def is_untagged_token(value: object) -> bool:
    """Is this hashtag_set cell an explicit 'no hashtag expected' marker?

    Read from the RAW cell, before blank-normalisation, because several of the
    words a person might reach for here ("none") are also blank tokens.
    """
    return str(value).strip().lower() in UNTAGGED_TOKENS


def normalise_hashtag_set(value: object) -> tuple[str, ...]:
    """``"hash.B+hash.A"`` -> ``("hash.A", "hash.B")``.

    Sorted so that set identity does not depend on the order someone happened
    to type the tags in. An untagged marker normalises to the empty tuple,
    which is the key a cell with no positive hashtags looks up.
    """
    if is_untagged_token(value):
        return ()
    s = _blank_to_empty(value)
    if not s:
        return ()
    parts = [p.strip() for p in s.split(_HASHTAG_SEP)]
    parts = [p for p in parts if p]
    return tuple(sorted(set(parts)))


# ===========================================================================
# gRNA whitelist
# ===========================================================================
@dataclass
class GuideWhitelist:
    """Per-guide family and optional target annotation.

    Only ``guide_id`` and ``family`` are required. Every other column is
    derived from the guide ID when blank, per row -- so a whitelist can be two
    columns wide, or can override a single awkward guide without filling in the
    rest.
    """

    path: Path
    df: pd.DataFrame                  # indexed by guide_id
    warnings: list[str] = field(default_factory=list)

    @property
    def families(self) -> list[str]:
        return sorted(set(self.df["family"].astype(str)))

    @property
    def guide_ids(self) -> list[str]:
        return list(self.df.index.astype(str))

    def declared_columns(self) -> list[str]:
        """Optional columns that carry at least one non-blank value.

        Reported so the run states which annotations were declared and which
        were derived -- a whitelist should never quietly do more than intended.
        """
        out = []
        for c in GUIDE_WHITELIST_KNOWN:
            if c in GUIDE_WHITELIST_REQUIRED or c not in self.df.columns:
                continue
            if (self.df[c].astype(str).str.len() > 0).any():
                out.append(c)
        return out

    def row_for(self, guide_id: str) -> dict[str, str] | None:
        gid = str(guide_id)
        if gid not in self.df.index:
            return None
        return {k: _blank_to_empty(v) for k, v in self.df.loc[gid].items()}

    def coverage(self, guide_ids: Sequence[str]) -> tuple[list[str], list[str]]:
        """``(in_object_not_listed, listed_not_in_object)``."""
        obj = [str(g) for g in guide_ids]
        listed = set(self.df.index.astype(str))
        missing = [g for g in obj if g not in listed]
        extra = sorted(listed - set(obj))
        return missing, extra


def load_guide_whitelist(path: Path | str) -> GuideWhitelist:
    p = Path(path)
    df, _delim, _fields = read_table(p, what="gRNA whitelist")

    missing = [c for c in GUIDE_WHITELIST_REQUIRED if c not in df.columns]
    if missing:
        raise ManifestError(
            f"gRNA whitelist {p.name} is missing required column(s) "
            f"{missing}. Only {list(GUIDE_WHITELIST_REQUIRED)} are required; "
            f"everything else is derived from the guide ID when absent."
        )

    for col in GUIDE_WHITELIST_KNOWN:
        if col not in df.columns:
            df[col] = ""
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].map(_blank_to_empty)

    warnings_: list[str] = []

    # --- guide_id -----------------------------------------------------------
    blank_ids = df["guide_id"] == ""
    if blank_ids.any():
        raise ManifestError(
            f"gRNA whitelist {p.name} has {int(blank_ids.sum())} row(s) with a "
            f"blank guide_id."
        )
    dupes = df.loc[df["guide_id"].duplicated(keep=False), "guide_id"]
    if len(dupes):
        raise ManifestError(
            f"gRNA whitelist {p.name} has duplicate guide_id values: "
            f"{sorted(set(dupes))[:10]}. Each guide needs exactly one row -- a "
            f"guide cannot belong to two families."
        )

    # --- family -------------------------------------------------------------
    blank_fam = df["family"] == ""
    if blank_fam.any():
        raise ManifestError(
            f"gRNA whitelist {p.name} has {int(blank_fam.sum())} row(s) with a "
            f"blank family. Family determines which NTC cells control for a "
            f"guide, so it cannot be left to a default."
        )
    bad_fam = sorted({f for f in df["family"] if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,8}", f)})
    if bad_fam:
        raise ManifestError(
            f"gRNA whitelist {p.name} has family values that are not short "
            f"tokens: {bad_fam[:10]}. Family appears in every plot label, so it "
            f"must match [A-Za-z0-9_.-]{{1,8}} (e.g. A, B, lib1)."
        )

    # --- parse_target_from --------------------------------------------------
    bad_parse = sorted({
        v for v in df["parse_target_from"] if v and v not in VALID_PARSE_FROM
    })
    if bad_parse:
        raise ManifestError(
            f"gRNA whitelist {p.name}: parse_target_from must be one of "
            f"{list(VALID_PARSE_FROM)} or blank; found {bad_parse}."
        )

    # --- role ---------------------------------------------------------------
    bad_role = sorted({v for v in df["role"] if v and v not in VALID_ROLES})
    if bad_role:
        raise ManifestError(
            f"gRNA whitelist {p.name}: role must be one of {list(VALID_ROLES)} "
            f"or blank; found {bad_role}."
        )

    # --- target_ensg --------------------------------------------------------
    bad_ensg = sorted({
        v for v in df["target_ensg"] if v and not _ENSG_RE.match(v)
    })
    if bad_ensg:
        raise ManifestError(
            f"gRNA whitelist {p.name}: target_ensg must look like ENSG00000123456; "
            f"found {bad_ensg[:10]}."
        )

    # --- control_of ---------------------------------------------------------
    # A matched control mirrors a gene that must actually be targeted in the
    # same family, otherwise the pairing points at nothing.
    targets_by_family: dict[str, set[str]] = {}
    for fam, sub in df.groupby("family"):
        targets_by_family[str(fam)] = {
            g for g in sub["target_gene"] if g and g.upper() != "NTC"
        }
    dangling = []
    for _, row in df.iterrows():
        co, fam = row["control_of"], str(row["family"])
        if co and targets_by_family.get(fam) and co not in targets_by_family[fam]:
            dangling.append((co, fam))
    if dangling:
        warnings_.append(
            f"gRNA whitelist {p.name}: control_of names a gene that is not a "
            f"declared target in the same family for "
            f"{sorted(set(dangling))[:6]}. The matched-control pairing will not "
            f"resolve for those guides."
        )

    df = df.set_index("guide_id", drop=True)
    return GuideWhitelist(path=p, df=df, warnings=warnings_)


# ===========================================================================
# Hashtag whitelist
# ===========================================================================
@dataclass
class HashtagWhitelist:
    """Declared hashtag combinations and what they identify.

    ``demux_id`` is a biological sample and MAY appear on several rows: one
    aliquot tagged combinatorially, another tagged singly, both resolving to
    the same sample. Uniqueness lives on ``hashtag_set``.
    """

    path: Path
    df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    @property
    def sets(self) -> list[tuple[str, ...]]:
        return list(self.df["hashtag_set_key"])

    @property
    def hashtags(self) -> list[str]:
        """Every hashtag named anywhere in the file."""
        out: set[str] = set()
        for s in self.df["hashtag_set_key"]:
            out |= set(s)
        return sorted(out)

    @property
    def families(self) -> list[str]:
        return sorted({f for f in self.df["family"].astype(str) if f})

    @property
    def demux_ids(self) -> list[str]:
        return sorted(set(self.df["demux_id"].astype(str)))

    @property
    def has_aliquots(self) -> bool:
        """True when any biological sample was split across tagging schemes."""
        return bool(self.df["demux_id"].duplicated().any())

    @property
    def has_untagged(self) -> bool:
        """True when a sample was declared as carrying no hashtag by design."""
        return bool(self.df.get("is_untagged", pd.Series(dtype=bool)).any())

    @property
    def untagged_rows(self) -> pd.DataFrame:
        if "is_untagged" not in self.df.columns:
            return self.df.iloc[0:0]
        return self.df[self.df["is_untagged"]]

    def lookup(self) -> dict[tuple[str, ...], dict[str, str]]:
        """``frozen hashtag set -> row``, the map used to classify cells."""
        out = {}
        for _, row in self.df.iterrows():
            out[tuple(row["hashtag_set_key"])] = {
                k: _blank_to_empty(v)
                for k, v in row.items()
                if k != "hashtag_set_key"
            }
        return out

    def metadata_columns(self) -> list[str]:
        return [
            c for c in self.df.columns
            if c not in HASHTAG_WHITELIST_KNOWN
            + ("hashtag_set_key", "is_untagged")
        ]


def load_hashtag_whitelist(
    path: Path | str, known_hashtags: Iterable[str] | None = None
) -> HashtagWhitelist:
    p = Path(path)
    df, _delim, _fields = read_table(p, what="hashtag whitelist")

    missing = [c for c in HASHTAG_WHITELIST_REQUIRED if c not in df.columns]
    if missing:
        raise ManifestError(
            f"Hashtag whitelist {p.name} is missing required column(s) {missing}."
        )
    for col in HASHTAG_WHITELIST_KNOWN:
        if col not in df.columns:
            df[col] = ""
    df = df.copy()
    raw_sets = df["hashtag_set"].astype(str)
    for col in df.columns:
        df[col] = df[col].map(_blank_to_empty)

    warnings_: list[str] = []

    # Untagged status is read from the RAW column, before blank-normalisation
    # wiped it: "none" is a blank token, and an intentionally untagged sample
    # must not be confused with a cell nobody filled in.
    df["is_untagged"] = raw_sets.map(is_untagged_token)
    df["hashtag_set_key"] = df["hashtag_set"].map(normalise_hashtag_set)
    empty = (df["hashtag_set_key"].map(len) == 0) & ~df["is_untagged"]
    if empty.any():
        raise ManifestError(
            f"Hashtag whitelist {p.name} has {int(empty.sum())} row(s) with a "
            f"blank hashtag_set. If a sample was intentionally not hashtag "
            f"labelled, write 'untagged' in hashtag_set rather than leaving "
            f"the cell blank or omitting the row -- otherwise its cells are "
            f"indistinguishable from hashtag capture failure, and they carry "
            f"no family, so they drop out of the family-scoped comparisons. "
            f"Accepted markers: {', '.join(sorted(UNTAGGED_TOKENS))}."
        )

    # At most one untagged row per sample. Cells with no positive hashtag all
    # look identical, so two untagged declarations in one pool could not be
    # told apart.
    n_untagged = df.groupby("sample")["is_untagged"].sum()
    bad = n_untagged[n_untagged > 1]
    if len(bad):
        raise ManifestError(
            f"Hashtag whitelist {p.name} declares more than one untagged "
            f"sample within sample(s) {list(bad.index)}. Cells carrying no "
            f"hashtag are indistinguishable from each other, so only one "
            f"untagged population per pool can be resolved."
        )
    if (df["demux_id"] == "").any():
        raise ManifestError(
            f"Hashtag whitelist {p.name} has row(s) with a blank demux_id."
        )

    # --- (sample, hashtag_set) is the primary key ---------------------------
    key = df["sample"].astype(str) + "||" + df["hashtag_set_key"].map(
        lambda s: _HASHTAG_SEP.join(s)
    )
    dup_sets = key[key.duplicated(keep=False)]
    if len(dup_sets):
        offending = sorted({k.split("||", 1)[1] for k in dup_sets})
        raise ManifestError(
            f"Hashtag whitelist {p.name} declares the same hashtag combination "
            f"twice within a sample: {offending[:6]}. A combination can only "
            f"identify one thing."
        )

    # --- aliquot is required once a demux_id repeats ------------------------
    for (sample, demux), sub in df.groupby(["sample", "demux_id"], sort=False):
        if len(sub) == 1:
            continue
        aliq = list(sub["aliquot"])
        if any(a == "" for a in aliq) or len(set(aliq)) != len(aliq):
            raise ManifestError(
                f"Hashtag whitelist {p.name}: demux_id {demux!r} in sample "
                f"{sample!r} appears on {len(sub)} rows, so each needs a "
                f"distinct non-blank aliquot to tell them apart on obs. "
                f"Found aliquots {aliq}. (Repeating a demux_id is legitimate --"
                f" it is how one sample split across single and combinatorial "
                f"tagging is declared.)"
            )

    # --- rows sharing a demux_id must agree on everything else --------------
    # The same biological sample cannot be two cell lines or two conditions.
    # This catches the copy-paste error that would silently merge unrelated
    # cells under one sample label.
    meta_cols = [
        c for c in df.columns
        if c not in ("hashtag_set", "hashtag_set_key", "aliquot", "demux_id")
    ]
    for (sample, demux), sub in df.groupby(["sample", "demux_id"], sort=False):
        if len(sub) == 1:
            continue
        for col in meta_cols:
            vals = {v for v in sub[col] if v != ""}
            if len(vals) > 1:
                raise ManifestError(
                    f"Hashtag whitelist {p.name}: rows sharing demux_id "
                    f"{demux!r} disagree on {col!r} ({sorted(vals)}). Aliquots "
                    f"of one biological sample must agree on everything except "
                    f"their hashtag_set and aliquot."
                )

    # --- hashtags must exist in the object ----------------------------------
    if known_hashtags is not None:
        known = {str(h) for h in known_hashtags}
        named = {h for s in df["hashtag_set_key"] for h in s}
        unknown = sorted(named - known)
        if unknown:
            raise ManifestError(
                f"Hashtag whitelist {p.name} names hashtag(s) that are not in "
                f"the data: {unknown}. Hashtags present are: {sorted(known)}. "
                f"Names must match exactly (e.g. 'prot:hash.A', not 'hash.A')."
            )
        unused = sorted(known - named)
        if unused:
            warnings_.append(
                f"{len(unused)} hashtag(s) present in the data appear in no "
                f"declared combination ({', '.join(unused[:8])}). They were "
                f"sequenced but are not part of the declared design, so no cell "
                f"can resolve to them."
            )

    return HashtagWhitelist(path=p, df=df, warnings=warnings_)


def cross_check_families(
    guides: GuideWhitelist | None, hashtags: HashtagWhitelist | None
) -> list[str]:
    """Families named in the hashtag whitelist must exist in the gRNA whitelist.

    This is the join that makes family-scoped controls work: a cell's hashtag
    identifies its family, and its family selects its NTC pool. A family with
    no guides cannot be analysed.
    """
    if guides is None or hashtags is None:
        return []
    declared = set(guides.families)
    used = set(hashtags.families)
    unknown = sorted(used - declared)
    if unknown:
        raise ManifestError(
            f"Hashtag whitelist {hashtags.path.name} refers to famil(y/ies) "
            f"{unknown}, which do not appear in gRNA whitelist "
            f"{guides.path.name} (families there: {sorted(declared)}). The two "
            f"files must share one family vocabulary -- that link is what "
            f"scopes each cell's NTC control pool."
        )
    notes = []
    unhashed = sorted(declared - used)
    if unhashed:
        notes.append(
            f"{len(unhashed)} guide famil(y/ies) ({', '.join(unhashed)}) are "
            f"declared in the gRNA whitelist but identified by no hashtag "
            f"combination. Cells from them can still be analysed via their "
            f"guide, but their family cannot be cross-checked."
        )
    return notes
