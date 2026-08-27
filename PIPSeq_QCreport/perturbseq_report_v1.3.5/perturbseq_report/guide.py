"""
Guide-RNA assignment and guide-library performance.

Consolidates what the original spread across sections 10, 11, 12, 13 and 22,
which between them contained:

* two different metrics both called "purity" (top1/(top1+top2) in one section,
  top1/total in another), reported side by side with no distinction;
* four ``run_*_comparison`` wrapper functions with the same shape -- loop over
  comparison axes, compute, save CSV, plot, print -- and no shared helper;
* NTC detection implemented three separate times (a literal string equality, a
  substring test, and a regex fallback) because the first two kept missing
  cases;
* two independent guide-matrix getters with different candidate lists.

Here the per-cell statistics are computed once by ``stats.compute_guide_stats``,
both purity definitions are named distinctly, guide-ID parsing is one function,
and the per-axis looping is one generic driver.

Dual-guide ("iBAR") libraries are supported, which the original did not handle
at all: each construct carries two barcoded guides targeting the same gene, the
positions are thresholded independently, and cells carrying two guides from
*different* constructs indicate recombination during library prep.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as P
from . import text as T
from .artifacts import Registry
from .config import FigureConfig, GuideConfig, PipelineConfig
from .modalities import Modality
from .stats import (
    GuideStats, assign_guides, assignment_sweep, compute_guide_stats, gini,
)


# ===========================================================================
# Guide-ID parsing
# ===========================================================================
@dataclass
class ParsedGuide:
    guide_id: str
    target_gene: str
    target_ensg: str | None
    family: str                 # guide population; scopes the NTC control pool
    role: str                   # targeting | ntc | matched_control | safe_harbour
    control_of: str | None      # for matched controls, the gene mirrored
    construct: str | None       # dual-guide construct this position belongs to
    ibar: str | None            # "1" or "2" for dual-guide libraries
    is_ntc: bool
    matched_pattern: int | None  # index into the regex list, None if unparsed
    name_first: str | None      # gene-symbol prefix
    name_second: str | None     # spacer_target field
    name_conflict: bool         # the two names disagree about targeting status
    guide_suffix: str           # "_v1.0s2" / "_g1" / "_sg", for labels
    annotation_source: str      # whitelist | parsed | mixed


class GuideParser:
    """Guide IDs -> targets, families and display labels.

    Three things changed in v1.2.0, each fixing a measured failure:

    **Two names, and which one wins.** Many IDs carry a gene-symbol prefix AND
    a ``spacer_target`` field::

        ABT1_target_version_1.0_..._spacer_target_ENSG00000146109_...   agree
        ABT1_target_version_1.1_..._spacer_target_ONE_INTERGENIC_SITE_... DISAGREE

    The ``.1`` versions are per-gene matched intergenic controls: same
    backbone, same prefix, spacer redirected. The prefix says what the
    construct was designed from; the spacer target says what it cuts. Which one
    is authoritative is declared per guide (``parse_target_from``), defaulting
    to ``second``.

    **NTC detection is structured-first.** v1.1.0 ran ``ntc_regex`` over the
    whole ID unconditionally, which got this library right by luck and would
    misclassify a gene legitimately named ``CTRL1``. The regex is now a
    fallback used only when no structured target field was captured.

    **Family.** Declared, never inferred. ``NTC_10_ACGT...`` carries no library
    information and two libraries can independently use ``NTC_10``, so pooling
    controls by name -- as v1.1.0 did across all 60 NTC guides in MDL-1856 --
    silently mixes control populations.
    """

    def __init__(self, cfg: GuideConfig, whitelist: Any = None):
        self.cfg = cfg
        self.whitelist = whitelist
        self._patterns = [re.compile(p) for p in cfg.guide_id_regexes]
        self._ibar = [re.compile(p) for p in cfg.ibar_regexes]
        self._ntc = re.compile(cfg.ntc_regex)
        self._control_target = re.compile(cfg.control_target_regex)
        self._ensg = re.compile(r"^ENS[A-Z]*G\d+$")
        self.unparsed: list[str] = []
        self.unlisted: list[str] = []
        self.conflicts: list[str] = []
        # Set in parse_all once every family is known. With a single family
        # the suffix is pure noise -- "ABT1_unassigned_v1.0s2" and
        # "NTC_unassigned" on every axis of a single-library experiment, where
        # v1.1.0 simply said "ABT1" and "NTC". It only earns its place when
        # there is more than one population to tell apart.
        self._suppress_family = False

    # -------------------------------------------------------------- helpers
    def _is_ensg(self, v: str | None) -> bool:
        return bool(v) and bool(self._ensg.match(str(v)))

    def _is_control_token(self, v: str | None) -> bool:
        return bool(v) and bool(self._control_target.search(str(v)))

    def _guide_suffix(self, gd: dict[str, Any], which: int | None) -> str:
        """Compact per-guide discriminator for plot labels.

        Built from whichever fields the matching pattern captured. ``version``
        keeps its literal value: ``1.0`` and ``1.1`` are different reagents, and
        stripping the ``.0`` collapses six distinct matched controls onto one
        label.
        """
        ver, num = gd.get("version"), gd.get("spacer_num")
        if ver and num:
            return f"_v{ver}s{num}"
        if ver:
            return f"_v{ver}"
        idx = gd.get("idx")
        if idx:
            return f"_g{idx}"
        if which is not None and "singleguide" in self.cfg.guide_id_regexes[which]:
            return "_sg"
        return ""

    # ---------------------------------------------------------------- parse
    def parse(self, guide_id: str) -> ParsedGuide:
        gid = str(guide_id)
        cfg = self.cfg

        wl_row = None
        if self.whitelist is not None:
            wl_row = self.whitelist.row_for(gid)
            if wl_row is None:
                self.unlisted.append(gid)

        family = (wl_row or {}).get("family") or cfg.unassigned_family
        parse_from = (wl_row or {}).get("parse_target_from") or cfg.parse_target_from

        # --- dual-guide position ------------------------------------------
        construct, ibar = None, None
        for pat in self._ibar:
            m = pat.match(gid)
            if m:
                construct = m.group("construct")
                ibar = m.group("ibar")
                break

        # --- structural parse ---------------------------------------------
        core = construct if construct is not None else gid
        gd: dict[str, Any] = {}
        which = None
        for i, pat in enumerate(self._patterns):
            m = pat.match(core) or pat.match(gid)
            if m:
                gd = m.groupdict()
                which = i
                break

        first = (gd.get("gene") or "").strip() or None
        second = (gd.get("ensg") or "").strip() or None
        suffix = self._guide_suffix(gd, which)

        # --- which name is authoritative ----------------------------------
        if second is None:
            authoritative = first
        elif first is None:
            authoritative = second
        else:
            authoritative = second if parse_from == "second" else first

        # --- classify ------------------------------------------------------
        ensg = second if self._is_ensg(second) else None
        second_is_control = self._is_control_token(second)
        first_is_control = self._is_control_token(first)
        name_conflict = bool(
            first and second and (second_is_control != first_is_control)
        )
        if name_conflict:
            self.conflicts.append(gid)

        if self._is_ensg(authoritative):
            # A structured Ensembl target settles it: this guide is targeting,
            # whatever the rest of the string looks like. The symbol from the
            # first name is kept for display.
            is_ntc = False
            gene = first or authoritative
        elif self._is_control_token(authoritative):
            is_ntc = True
            gene = cfg.ntc_label
        elif second is not None:
            # A structured target we do not recognise as either -- trust it as
            # a plain symbol rather than falling back to a substring search.
            is_ntc = False
            gene = authoritative or gid
        else:
            # No structured target field at all: this is the only situation in
            # which the whole-ID substring heuristic is appropriate.
            is_ntc = bool(self._ntc.search(gid))
            gene = cfg.ntc_label if is_ntc else (authoritative or core)
            if which is None and construct is None and not is_ntc:
                self.unparsed.append(gid)

        role = "ntc" if is_ntc else "targeting"
        control_of = None
        if is_ntc and first and not first_is_control:
            # Prefix is a real gene but the spacer targets a control site:
            # this is a matched control. Confirmed against the family's
            # targeted genes in the second pass (see parse_all).
            control_of = first

        # --- whitelist overrides ------------------------------------------
        source = "parsed"
        if wl_row:
            declared = False
            if wl_row.get("target_gene"):
                gene = wl_row["target_gene"]
                is_ntc = gene.upper() == cfg.ntc_label.upper()
                declared = True
            if wl_row.get("target_ensg"):
                ensg = wl_row["target_ensg"]
                declared = True
            if wl_row.get("role"):
                role = wl_row["role"]
                is_ntc = role in ("ntc", "matched_control", "safe_harbour")
                declared = True
            if wl_row.get("control_of"):
                control_of = wl_row["control_of"]
                declared = True
            if declared:
                source = "whitelist"
            elif self.whitelist is not None:
                source = "mixed"      # family declared, annotation derived
        if is_ntc and role == "targeting":
            role = "ntc"

        return ParsedGuide(
            guide_id=gid, target_gene=gene, target_ensg=ensg, family=str(family),
            role=role, control_of=control_of, construct=construct, ibar=ibar,
            is_ntc=is_ntc, matched_pattern=which, name_first=first,
            name_second=second, name_conflict=name_conflict,
            guide_suffix=suffix, annotation_source=source,
        )

    # ------------------------------------------------------------ labelling
    def _base_label(self, r: ParsedGuide) -> str:
        if r.role == "matched_control" and r.control_of:
            return f"{r.control_of}{self.cfg.matched_control_marker}"
        if r.is_ntc:
            return self.cfg.ntc_label
        return r.target_gene

    def short_label(self, r: ParsedGuide) -> str:
        """``ABT1_A_v1.0s2`` -- gene, family, guide discriminator.

        Family is always shown: two libraries can use the same gene symbol with
        different constructs, and a label that hides which population a guide
        came from is worse than a long one.
        """
        if self._suppress_family:
            return f"{self._base_label(r)}{r.guide_suffix}"
        sep = self.cfg.family_label_sep
        return f"{self._base_label(r)}{sep}{r.family}{r.guide_suffix}"

    def target_key(self, r: ParsedGuide) -> str:
        """Grouping key for every per-target comparison.

        Family-scoped, because an NTC from one library is not a control for a
        guide from another. ``pool_ntc_across_families`` collapses the control
        pools back together for libraries where that is genuinely correct.
        """
        if r.is_ntc and self.cfg.pool_ntc_across_families:
            return self.cfg.ntc_label
        if self._suppress_family:
            return r.target_gene
        return f"{r.target_gene}{self.cfg.family_label_sep}{r.family}"

    # ------------------------------------------------------------ parse_all
    def parse_all(self, guide_ids: Sequence[str]) -> pd.DataFrame:
        rows = [self.parse(g) for g in guide_ids]

        # Decide once, before any label or key is built, whether the family
        # suffix carries information here.
        self._suppress_family = len({r.family for r in rows}) <= 1

        # --- second pass: confirm matched controls -------------------------
        # A control is "matched" only if the gene its prefix names is actually
        # targeted somewhere in the same family. Otherwise the pairing points
        # at nothing and it is just an NTC.
        targeted: dict[str, set[str]] = {}
        for r in rows:
            if not r.is_ntc and r.target_gene:
                targeted.setdefault(r.family, set()).add(r.target_gene)
        for r in rows:
            if r.role in ("ntc", "matched_control") and r.control_of:
                if r.control_of in targeted.get(r.family, set()):
                    r.role = "matched_control"
                else:
                    r.role = "ntc"
                    r.control_of = None

        # --- labels, with a guaranteed-uniqueness pass ---------------------
        labels = [self.short_label(r) for r in rows]
        counts = Counter(labels)
        if any(v > 1 for v in counts.values()):
            # Uniqueness must not depend on the library's naming discipline.
            # Disambiguate with the spacer when we have one, else a stable hash
            # of the full ID.
            seen: Counter = Counter()
            for i, r in enumerate(rows):
                if counts[labels[i]] == 1:
                    continue
                gd_spacer = None
                if r.matched_pattern is not None:
                    m = self._patterns[r.matched_pattern].match(r.guide_id)
                    if m:
                        gd_spacer = m.groupdict().get("spacer")
                if gd_spacer:
                    tag = str(gd_spacer)[:4]
                else:
                    tag = f"{abs(hash(r.guide_id)) % 0xFFFF:04x}"
                seen[labels[i]] += 1
                labels[i] = f"{labels[i]}.{tag}"
            # Pathological case: identical spacers. Fall back to an ordinal.
            final = Counter(labels)
            if any(v > 1 for v in final.values()):
                bump: Counter = Counter()
                for i, lab in enumerate(labels):
                    if final[lab] > 1:
                        bump[lab] += 1
                        labels[i] = f"{lab}.{bump[lab]}"

        df = pd.DataFrame(
            [
                {
                    "guide_id": r.guide_id,
                    "target_gene": r.target_gene,
                    "target_ensg": r.target_ensg,
                    "family": r.family,
                    "target_key": self.target_key(r),
                    "role": r.role,
                    "control_of": r.control_of,
                    "construct": r.construct,
                    "ibar": r.ibar,
                    "is_ntc": r.is_ntc,
                    "matched_pattern": r.matched_pattern,
                    "name_first": r.name_first,
                    "name_second": r.name_second,
                    "name_conflict": r.name_conflict,
                    "annotation_source": r.annotation_source,
                    "short_label": labels[i],
                }
                for i, r in enumerate(rows)
            ]
        )
        return df

    def detect_dual_guide(self, mapping: pd.DataFrame) -> bool:
        """Is this a dual-guide library?

        True when a clear majority of guides parse into constructs that carry
        exactly two positions.  Autodetection beats a config flag here because
        getting it wrong is silent: treating a dual-guide library as
        single-guide makes every two-guide cell look like a purity failure.
        """
        if mapping["construct"].isna().all():
            return False
        sizes = (
            mapping.dropna(subset=["construct"])
            .groupby("construct")["ibar"]
            .nunique()
        )
        if sizes.empty:
            return False
        return bool((sizes == 2).mean() > 0.5 and sizes.size >= 2)


# ===========================================================================
# Assignment
# ===========================================================================
@dataclass
class GuideAssignment:
    """Per-cell guide calls plus the library-level summaries."""

    per_cell: pd.DataFrame           # indexed by barcode
    mapping: pd.DataFrame            # per guide: target, family, role, label...
    stats: GuideStats
    dual_guide: bool
    unparsed_guides: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unlisted_guides: list[str] = field(default_factory=list)
    name_conflicts: list[str] = field(default_factory=list)

    @property
    def families(self) -> list[str]:
        return sorted(set(self.mapping["family"].astype(str)))

    @property
    def n_targets(self) -> int:
        """Distinct comparison groups, excluding controls."""
        m = self.mapping
        return int(m.loc[~m["is_ntc"], "target_key"].nunique())

    @property
    def n_assigned(self) -> int:
        return int(self.per_cell["guide_is_assigned"].sum())

    @property
    def frac_assigned(self) -> float:
        n = len(self.per_cell)
        return self.n_assigned / n if n else float("nan")


def call_guides(
    guide: Modality,
    cfg: GuideConfig,
    obs_index: Sequence[str] | None = None,
    whitelist: Any = None,
) -> GuideAssignment:
    """Assign guides to cells and annotate each cell with its target.

    For dual-guide libraries each position is thresholded independently and the
    two calls are combined, so a cell can end up with 0, 1 or 2 assigned guides
    -- the behaviour the collaborator's library requires and the original could
    not express.

    ``whitelist`` is an optional :class:`~.whitelists.GuideWhitelist` supplying
    each guide's family, and optionally its target annotation. Without one every
    guide lands in ``cfg.unassigned_family`` and controls are pooled across the
    whole library, which is v1.1.0 behaviour and is wrong for any experiment
    running more than one guide population.
    """
    parser = GuideParser(cfg, whitelist)
    mapping = parser.parse_all(guide.names)
    dual = cfg.dual_guide if cfg.dual_guide is not None else parser.detect_dual_guide(mapping)

    notes: list[str] = []
    if parser.unparsed:
        shown = ", ".join(parser.unparsed[:5])
        notes.append(
            f"{len(parser.unparsed)} of {len(guide.names)} guide IDs did not match "
            f"any target-parsing pattern (e.g. {shown}). Their guide ID is being "
            f"used as the target name, so those guides will appear as separate "
            f"single-guide 'targets'. Add a pattern to "
            f"GuideConfig.guide_id_regexes, or give them a target_gene in the "
            f"gRNA whitelist."
        )
    if whitelist is not None and parser.unlisted:
        shown = ", ".join(parser.unlisted[:5])
        notes.append(
            f"{len(parser.unlisted)} of {len(guide.names)} guides are not in the "
            f"gRNA whitelist (e.g. {shown}). They were parsed from their IDs and "
            f"placed in family {cfg.unassigned_family!r}, which is treated as its "
            f"own control population -- their cells are never compared against "
            f"another family's NTCs. Add them to the whitelist to place them."
        )
    if parser.conflicts:
        shown = ", ".join(parser.conflicts[:3])
        notes.append(
            f"{len(parser.conflicts)} guide ID(s) carry two target names that "
            f"disagree about whether the guide is targeting (e.g. {shown}). "
            f"Resolved using the {cfg.parse_target_from!r} name. These are "
            f"normally matched intergenic controls, which is expected; a large "
            f"count can also mean the wrong parse_target_from for this library."
        )
    n_fam = mapping["family"].nunique()
    if n_fam > 1 and not cfg.pool_ntc_across_families:
        per_fam = (
            mapping[mapping["is_ntc"]].groupby("family")["guide_id"].size().to_dict()
        )
        notes.append(
            f"{n_fam} guide families are present, so controls are scoped per "
            f"family: NTC guides per family = {per_fam}. A guide-positive cell "
            f"is only ever compared against NTC cells from its own family."
        )

    idx = pd.Index(
        [str(x) for x in (obs_index if obs_index is not None else guide.obs_names)]
        or range(guide.n_cells),
        name="barcode",
    )

    # Per-guide lookups reused by both calling paths.
    by_guide = mapping.set_index("guide_id")

    if not dual:
        stats = compute_guide_stats(guide.X)
        per_cell = stats.to_frame(idx)
        calls = assign_guides(stats, guide.names, cfg.min_reads, cfg.purity_min)
        calls.index = idx
        per_cell = per_cell.join(calls)
        per_cell["n_assigned_guides"] = per_cell["guide_is_assigned"].astype(int)
        for col in ("target_gene", "target_ensg", "family", "target_key",
                    "role", "control_of", "short_label", "construct"):
            per_cell[col] = per_cell["assigned_guide"].map(by_guide[col])
        per_cell["is_ntc"] = per_cell["assigned_guide"].map(
            by_guide["is_ntc"]
        ).fillna(False).astype(bool)
        per_cell["guide_pair_status"] = np.where(
            per_cell["guide_is_assigned"], "single", "unassigned"
        )
    else:
        notes.append(
            f"Dual-guide (iBAR) library detected: "
            f"{mapping['construct'].nunique()} constructs across "
            f"{len(guide.names)} guide positions. Each position is thresholded "
            f"independently."
        )
        per_cell, extra = _call_dual_guide(guide, mapping, cfg, idx)
        notes.extend(extra)
        stats = compute_guide_stats(guide.X)

    # The dual-guide path builds its own frame and does not know about
    # families, so backfill anything it left out. Doing it here rather than in
    # both branches means a new per-guide annotation cannot be added to one
    # calling path and silently forgotten in the other.
    for col in ("target_gene", "target_ensg", "family", "target_key",
                "role", "control_of", "short_label"):
        if col not in per_cell.columns:
            per_cell[col] = per_cell["assigned_guide"].map(by_guide[col])
    # Cells with no assignment still need a family for grouping; an
    # unassigned cell belongs to no control population.
    per_cell["family"] = per_cell["family"].where(
        per_cell["guide_is_assigned"], None
    )

    # Overall depth columns, always present so downstream code need not check.
    per_cell["guide_total_umis"] = guide.X.sum(axis=1)
    per_cell["n_guides_detected"] = (guide.X > 0).sum(axis=1)

    return GuideAssignment(
        per_cell=per_cell, mapping=mapping, stats=stats, dual_guide=dual,
        unparsed_guides=parser.unparsed, notes=notes,
        unlisted_guides=parser.unlisted, name_conflicts=parser.conflicts,
    )


def _call_dual_guide(
    guide: Modality, mapping: pd.DataFrame, cfg: GuideConfig, idx: pd.Index
) -> tuple[pd.DataFrame, list[str]]:
    """Threshold each iBAR position independently, then combine."""
    notes: list[str] = []
    name_to_col = {n: i for i, n in enumerate(guide.names)}
    frames: dict[str, pd.DataFrame] = {}

    for ibar_value in sorted(mapping["ibar"].dropna().unique()):
        cols = [
            name_to_col[n]
            for n in mapping.loc[mapping["ibar"] == ibar_value, "guide_id"]
            if n in name_to_col
        ]
        if not cols:
            continue
        sub_names = [guide.names[c] for c in cols]
        st = compute_guide_stats(guide.X[:, cols])
        calls = assign_guides(st, sub_names, cfg.min_reads, cfg.purity_min)
        f = st.to_frame(idx)
        calls.index = idx
        f = f.join(calls)
        frames[str(ibar_value)] = f

    if not frames:
        notes.append(
            "Dual-guide layout was detected but no guide positions could be "
            "grouped; falling back to single-guide calling."
        )
        st = compute_guide_stats(guide.X)
        per_cell = st.to_frame(idx).join(
            assign_guides(st, guide.names, cfg.min_reads, cfg.purity_min).set_index(idx)
        )
        per_cell["n_assigned_guides"] = per_cell["guide_is_assigned"].astype(int)
        return per_cell, notes

    per_cell = pd.DataFrame(index=idx)
    assigned_cols = []
    for ib, f in frames.items():
        per_cell[f"guide_ibar{ib}"] = f["assigned_guide"]
        per_cell[f"purity_ibar{ib}"] = f["guide_purity_pct"]
        per_cell[f"umis_ibar{ib}"] = f["guide_total_umis"]
        per_cell[f"assigned_ibar{ib}"] = f["guide_is_assigned"]
        assigned_cols.append(f"assigned_ibar{ib}")

    per_cell["n_assigned_guides"] = per_cell[assigned_cols].sum(axis=1).astype(int)
    per_cell["guide_is_assigned"] = per_cell["n_assigned_guides"] > 0

    construct_of = dict(zip(mapping["guide_id"], mapping["construct"]))
    gene_of = dict(zip(mapping["guide_id"], mapping["target_gene"]))
    ib_keys = sorted(frames)

    def combine(row) -> tuple[str | None, str | None, str]:
        gs = [row.get(f"guide_ibar{ib}") for ib in ib_keys]
        present = [g for g in gs if isinstance(g, str)]
        if not present:
            return None, None, "unassigned"
        constructs = {construct_of.get(g) for g in present}
        constructs.discard(None)
        if len(present) == 1:
            g = present[0]
            return gene_of.get(g), construct_of.get(g), "single_position"
        if len(constructs) == 1:
            g = present[0]
            return gene_of.get(g), next(iter(constructs)), "expected_pair"
        return None, None, "recombinant_pair"

    combined = per_cell.apply(combine, axis=1, result_type="expand")
    combined.columns = ["target_gene", "construct", "guide_pair_status"]
    per_cell = pd.concat([per_cell, combined], axis=1)
    per_cell["is_ntc"] = per_cell["target_gene"].eq(cfg.ntc_label)
    per_cell["assigned_guide"] = per_cell[f"guide_ibar{ib_keys[0]}"]

    n_recomb = int((per_cell["guide_pair_status"] == "recombinant_pair").sum())
    n_pairs = int(per_cell["guide_pair_status"].isin(
        ["expected_pair", "recombinant_pair"]).sum())
    if n_pairs:
        rate = 100.0 * n_recomb / n_pairs
        notes.append(
            f"Recombination rate: {rate:.1f}% of two-guide cells "
            f"({n_recomb:,} of {n_pairs:,}) carry guides from different "
            f"constructs. Those cells have no single well-defined perturbation "
            f"and are excluded from perturbation analysis."
        )
    return per_cell, notes


# ===========================================================================
# Summaries
# ===========================================================================
def guide_efficiency_by_group(
    per_cell: pd.DataFrame, group: pd.Series, min_reads: int | None = None
) -> pd.DataFrame:
    """Assignment rate and guide depth per group -- one function, one shape.

    Replaces the four near-identical ``run_*_comparison`` wrappers.

    ``pct_above_min_reads`` used to be computed as ``guide_total_umis > 0``,
    i.e. the percentage of cells with *any* guide UMI, under a name promising
    the assignment threshold. Both quantities are now emitted, each under a
    name that matches what it measures; ``pct_above_min_reads`` is NaN when no
    threshold is supplied rather than quietly meaning something else.
    """
    g = group.astype(str).reindex(per_cell.index)
    rows = []
    for value in sorted(g.dropna().unique()):
        sel = (g == value).to_numpy()
        sub = per_cell.loc[sel]
        n = len(sub)
        if n == 0:
            continue
        row: dict[str, Any] = {
            "group": value,
            "n_cells": n,
            "n_assigned": int(sub["guide_is_assigned"].sum()),
            "pct_assigned": 100.0 * float(sub["guide_is_assigned"].mean()),
            "median_guide_umis": float(sub["guide_total_umis"].median()),
            "mean_guide_umis": float(sub["guide_total_umis"].mean()),
            "median_guides_detected": float(sub["n_guides_detected"].median()),
            "pct_with_any_guide_umi": 100.0 * float(
                (sub["guide_total_umis"] > 0).mean()
            ),
            "pct_above_min_reads": (
                100.0 * float((sub["guide_total_umis"] >= min_reads).mean())
                if min_reads is not None else float("nan")
            ),
        }
        if "guide_purity_pct" in sub.columns:
            row["median_purity"] = float(
                pd.to_numeric(sub["guide_purity_pct"], errors="coerce").median()
            )
        if "guide_pair_status" in sub.columns:
            counts = sub["guide_pair_status"].value_counts(normalize=True) * 100
            for status in ("expected_pair", "recombinant_pair", "single_position",
                           "single", "unassigned"):
                if status in counts.index:
                    row[f"pct_{status}"] = float(counts[status])
        rows.append(row)
    return pd.DataFrame(rows)


def guide_representation(
    guide: Modality, per_cell: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-guide and per-target cell counts and UMI shares, plus skew."""
    umis = guide.X.sum(axis=0)
    per_guide = pd.DataFrame(
        {
            "guide_id": guide.names,
            "total_umis": umis,
            "frac_umis": umis / umis.sum() if umis.sum() else np.nan,
            "n_cells_detected": (guide.X > 0).sum(axis=0),
        }
    ).merge(mapping, on="guide_id", how="left")

    assigned = per_cell.loc[per_cell["guide_is_assigned"]]
    counts = assigned["assigned_guide"].value_counts()
    per_guide["n_cells_assigned"] = per_guide["guide_id"].map(counts).fillna(0)

    per_target = (
        per_guide.groupby("target_gene", dropna=False)
        .agg(
            n_guides=("guide_id", "size"),
            total_umis=("total_umis", "sum"),
            n_cells_assigned=("n_cells_assigned", "sum"),
        )
        .reset_index()
        .sort_values("n_cells_assigned", ascending=False)
    )
    # n_cells_with_target was identical to n_cells_assigned in every run --
    # both count the cells assigned to that target -- so it was two columns
    # saying one thing. Dropped rather than kept as a decoy.
    return per_guide, per_target


# ===========================================================================
# Figures
# ===========================================================================
def plot_purity_overview(
    stats: GuideStats, cfg: GuideConfig, fcfg: FigureConfig, path: Path
) -> Path:
    """Purity distributions, the two purity definitions, and the strict gate."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))

    P.histogram_by_group(
        axes[0],
        {"cells with guide reads": stats.top1_over_top2},
        fcfg, xlabel="purity: top1 / (top1 + top2)  (%)",
        vlines=[cfg.purity_min], bins=60,
    )
    axes[0].set_title("assignment criterion")

    P.hexbin_panel(
        axes[1], stats.total, stats.top1_over_top2, fcfg,
        xlabel="total guide UMIs in cell", ylabel="purity top1/(top1+top2) (%)",
        vlines=[cfg.min_reads], hlines=[cfg.purity_min], log_x=True,
        title="purity vs depth",
    )

    P.hexbin_panel(
        axes[2], stats.top1_over_total, stats.top12_over_total, fcfg,
        xlabel="top1 / total", ylabel="(top1 + top2) / total",
        vlines=[cfg.gate_top1_over_total_min],
        hlines=[cfg.gate_top12_over_total_min],
        title="purity triangle",
    )
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_purity_by_group(
    stats: GuideStats, group: pd.Series, group_name: str,
    cfg: GuideConfig, fcfg: FigureConfig, path: Path,
) -> Path:
    """Assignment criterion and purity triangle, one column per group level.

    The pooled panels answer "did guide calling work?". These answer "did it
    work *equally*?", which is the question that matters when the experiment is
    a comparison: an assignment rate that differs by condition is a systematic
    difference in guide capture, and it will propagate into every per-condition
    number downstream without ever looking like a bug.
    """
    levels = [str(g) for g in sorted(pd.unique(group.astype(str)))]
    n = len(levels)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        P.annotate_empty(ax, f"no {group_name} levels to compare")
        return P.save_figure(fig, path, fcfg)

    g = group.astype(str).to_numpy()
    fig, axes = plt.subplots(2, n, figsize=(4.3 * n, 7.4), squeeze=False)
    for j, lvl in enumerate(levels):
        m = g == lvl
        n_cells = int(m.sum())
        # Row 1: the assignment criterion, with the cut-off drawn on.
        P.histogram_by_group(
            axes[0][j], {lvl: stats.top1_over_top2[m]}, fcfg,
            xlabel="top1 / (top1 + top2)  (%)",
            vlines=[cfg.purity_min], bins=50,
        )
        passing = float(np.nanmean(stats.top1_over_top2[m] >= cfg.purity_min) * 100) \
            if n_cells else float("nan")
        axes[0][j].set_title(
            f"{group_name} = {lvl}\nassignment criterion  "
            f"({n_cells:,} cells, {passing:.1f}% above cut-off)",
            fontsize=8.5,
        )
        # Row 2: the purity triangle.
        P.hexbin_panel(
            axes[1][j], stats.top1_over_total[m], stats.top12_over_total[m], fcfg,
            xlabel="top1 / total", ylabel="(top1 + top2) / total",
            vlines=[cfg.gate_top1_over_total_min],
            hlines=[cfg.gate_top12_over_total_min],
            title="purity triangle",
        )
    fig.suptitle(f"guide purity by {group_name}", fontsize=10)
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_purity_sweep(
    stats: GuideStats, per_cell: pd.DataFrame, group: pd.Series | None,
    cfg: GuideConfig, fcfg: FigureConfig, path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if group is None:
        sweeps = {"all cells": assignment_sweep(stats, cfg.purity_sweep, cfg.min_reads)}
    else:
        g = group.astype(str).reindex(per_cell.index).to_numpy()
        sweeps = {}
        for value in sorted(pd.unique(g[pd.notna(g)])):
            sel = g == value
            sub = GuideStats(
                total=stats.total[sel], top1=stats.top1[sel], top2=stats.top2[sel],
                top1_over_top2=stats.top1_over_top2[sel],
                top1_over_total=stats.top1_over_total[sel],
                top12_over_total=stats.top12_over_total[sel],
                n_detected=stats.n_detected[sel], top1_index=stats.top1_index[sel],
            )
            sweeps[str(value)] = assignment_sweep(sub, cfg.purity_sweep, cfg.min_reads)

    colors = P.palette(fcfg, len(sweeps))
    for (label, df), c in zip(sweeps.items(), colors):
        ax.plot(df["purity_threshold"], df["frac_assigned"] * 100, "-o", ms=2.5,
                color=c, label=str(label), lw=1.3)
    ax.axvline(cfg.purity_min, color="#C44E52", ls="--", lw=1.1)
    ax.set_xlabel("purity threshold: top1/(top1+top2) (%)")
    ax.set_ylabel("% of cells assigned a guide")
    ax.set_title("sensitivity of assignment to the purity cut-off")
    if len(sweeps) > 1:
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_moi(per_cell: pd.DataFrame, fcfg: FigureConfig, path: Path,
             group: pd.Series | None = None) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))

    n_det = pd.to_numeric(per_cell["n_guides_detected"], errors="coerce").fillna(0)
    counts = n_det.clip(upper=10).value_counts().sort_index()
    axes[0].bar(counts.index.astype(int).astype(str), counts.to_numpy(),
                color="#4C72B0")
    for i, v in enumerate(counts.to_numpy()):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=6.5)
    axes[0].set_xlabel("distinct guides detected per cell (10 = 10 or more)")
    axes[0].set_ylabel("cells")
    axes[0].set_title("multiplicity of infection (detected)")

    if "n_assigned_guides" in per_cell.columns:
        a = per_cell["n_assigned_guides"].value_counts().sort_index()
        axes[1].bar(a.index.astype(int).astype(str), a.to_numpy(), color="#55A868")
        for i, v in enumerate(a.to_numpy()):
            axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=6.5)
        axes[1].set_xlabel("guides assigned per cell (after purity gate)")
        axes[1].set_ylabel("cells")
        axes[1].set_title("assigned guides per cell")
    else:
        P.annotate_empty(axes[1], "no assignment column")
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_abundance(
    guide: Modality, per_guide: pd.DataFrame, group_matrices: dict[str, np.ndarray],
    fcfg: FigureConfig, path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    P.rank_abundance(
        axes[0], group_matrices or {"all cells": guide.X.sum(axis=0)}, fcfg,
        ylabel="guide UMIs", xlabel="guide rank",
    )
    g = gini(per_guide["total_umis"].to_numpy())
    axes[0].set_title(f"guide rank abundance (Gini = {g:.2f})")

    ordered = per_guide.sort_values("n_cells_assigned", ascending=False)
    show = ordered.head(60)
    axes[1].bar(np.arange(len(show)), show["n_cells_assigned"].to_numpy(),
                color="#4C72B0")
    axes[1].set_xlabel(f"guide (top {len(show)} of {len(ordered)})")
    axes[1].set_ylabel("cells assigned")
    axes[1].set_title("cells per guide")
    n_zero = int((ordered["n_cells_assigned"] == 0).sum())
    if n_zero:
        axes[1].text(
            0.98, 0.95, f"{n_zero} guides with 0 assigned cells",
            transform=axes[1].transAxes, ha="right", va="top", fontsize=7,
            color="#C44E52",
        )

    P.histogram_by_group(
        axes[2], {"cells": guide.X.sum(axis=1)}, fcfg,
        xlabel="total guide UMIs per cell", log_x=True, bins=60,
    )
    axes[2].set_title("guide depth per cell")
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_efficiency(
    eff: pd.DataFrame, group_name: str, fcfg: FigureConfig, path: Path
) -> Path:
    cols = [c for c in ("pct_assigned", "median_guide_umis", "median_purity")
            if c in eff.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 3.6),
                             squeeze=False)
    colors = P.palette(fcfg, len(eff))
    for ax, col in zip(axes.ravel(), cols):
        ax.bar(eff["group"].astype(str), eff[col].to_numpy(), color=colors)
        for i, v in enumerate(eff[col].to_numpy()):
            if np.isfinite(v):
                ax.text(i, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_ylabel(col.replace("_", " "))
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"guide performance by {group_name}", fontsize=10)
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_pair_status(
    per_cell: pd.DataFrame, group: pd.Series | None, fcfg: FigureConfig, path: Path
) -> Path:
    """Dual-guide pair composition, including the recombination rate."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if group is None:
        frac = (
            per_cell["guide_pair_status"].value_counts(normalize=True)
            .to_frame().T
        )
        frac.index = ["all cells"]
    else:
        g = group.astype(str).reindex(per_cell.index)
        frac = (
            pd.crosstab(g, per_cell["guide_pair_status"], normalize="index")
        )
    P.stacked_fraction_bars(ax, frac, fcfg, legend_title="pair status")
    ax.set_title("dual-guide pair composition")
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


# ===========================================================================
# Stage driver
# ===========================================================================
def run_guide_stage(
    guide: Modality,
    cfg: PipelineConfig,
    reg: Registry,
    group_columns: dict[str, pd.Series],
    whitelist: Any = None,
    sample: pd.Series | None = None,
) -> GuideAssignment | None:
    """Full guide section: call, summarise, plot, register.

    ``sample`` is the per-cell sample label, used for the per-sample purity
    panels. Optional: without it those panels fall back to whatever comparison
    axes were supplied.
    """
    if not guide.present:
        reg.skipped(
            "guides", "all", "Guide analysis",
            f"No guide matrix was found in the h5ad. {guide.source}",
        )
        return None

    gcfg, fcfg = cfg.guide, cfg.figures
    fig_dir, table_dir = cfg.fig_dir, cfg.table_dir

    ga = call_guides(guide, gcfg, whitelist=whitelist)
    for i, note in enumerate(ga.notes):
        level = (
            "warn"
            if ("did not match" in note or "Recombination" in note
                or "not in the gRNA whitelist" in note)
            else "info"
        )
        reg.note("guides", f"note_{i}", "Guide calling", note, level=level,
                 order=5 + i)

    # Guide-ID -> target mapping, always emitted. This table is how a reader
    # checks that parsing did the right thing; v1.1.0 only surfaced a warning
    # counting the failures, which meant the 168 mis-parsed guides in MDL-1856
    # were invisible unless you knew to look at the target list.
    ga.mapping.to_csv(table_dir / "guide_target_mapping.csv", index=False)
    reg.table(
        "guides", "mapping", "Guide ID to target mapping",
        path=table_dir / "guide_target_mapping.csv",
        inline=ga.mapping.head(cfg.report.max_table_rows).to_dict("records"),
        columns=list(ga.mapping.columns),
        caption=(
            "Every guide with the target, family and short label it resolved "
            "to, and which pattern matched. annotation_source says whether the "
            "values were declared in the gRNA whitelist or derived from the "
            "guide ID."
        ),
        order=8,
    )
    if not ga.mapping.empty:
        n_fam = ga.mapping["family"].nunique()
        reg.metric("summary", "n_guide_families", "Guide families", n_fam,
                   order=22)
        shared = (
            ga.mapping[~ga.mapping["is_ntc"]]
            .groupby("target_gene")["family"].nunique()
        )
        n_shared = int((shared > 1).sum())
        if n_shared:
            reg.note(
                "guides", "shared_symbols", "Gene symbols in several families",
                (
                    f"{n_shared} gene symbol(s) appear in more than one guide "
                    f"family. They are kept as separate targets, because the "
                    f"same symbol reached by different constructs in different "
                    f"populations is not the same experiment."
                ),
                order=9,
            )

    reg.note(
        "guides", "purity_definitions", "Two definitions of purity",
        T.GUIDE_PURITY_NOTE, order=4,
    )

    # --- metrics -----------------------------------------------------------
    reg.metric("summary", "n_guides", "Guides in library", guide.n_features,
               order=20)
    reg.metric(
        "summary", "pct_cells_with_guide", "Cells with an assigned guide",
        round(100.0 * ga.frac_assigned, 1), unit="%",
        level=("good" if ga.frac_assigned > 0.7
               else "warn" if ga.frac_assigned > 0.4 else "poor"),
        order=21,
    )
    # Excluding the non-targeting pool: it is a control, not a target, and
    # counting it made this headline metric high by one.
    _assigned = ga.per_cell.loc[ga.per_cell["guide_is_assigned"]]
    if "is_ntc" in _assigned.columns:
        _assigned = _assigned.loc[~_assigned["is_ntc"].fillna(False).astype(bool)]
    n_targets = int(
        _assigned.loc[_assigned["target_gene"] != gcfg.ntc_label,
                      "target_gene"].nunique()
    )
    reg.metric("summary", "n_targets", "Distinct targets assigned", n_targets,
               order=22)

    # --- figures -----------------------------------------------------------
    reg.figure(
        "guides", "purity", "Guide purity and the assignment gate",
        plot_purity_overview(ga.stats, gcfg, fcfg, fig_dir / "guide_purity.png"),
        caption=T.guide_section_desc(gcfg.min_reads, gcfg.purity_min, ga.dual_guide)
        + " " + T.PURITY_GATE_DESC,
        order=10, width="full",
    )
    primary = next(iter(group_columns.values()), None)
    reg.figure(
        "guides", "purity_sweep", "Assignment rate vs purity cut-off",
        plot_purity_sweep(ga.stats, ga.per_cell, primary, gcfg, fcfg,
                          fig_dir / "guide_purity_sweep.png"),
        caption=T.PURITY_SWEEP_DESC, order=20,
    )
    # Purity per sample and per condition. The pooled panels say whether guide
    # calling worked; these say whether it worked equally, which is the only
    # version of the question that matters for a comparison experiment.
    purity_axes: dict[str, pd.Series] = {}
    if sample is not None:
        purity_axes["sample"] = pd.Series(sample).astype(str)
    for axis_name, series in group_columns.items():
        purity_axes.setdefault(axis_name, series.astype(str))

    for i, (axis_name, series) in enumerate(purity_axes.items()):
        s = series.reindex(ga.per_cell.index) if hasattr(series, "reindex") else series
        if s is None or pd.Series(s).astype(str).nunique() < 2:
            continue
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(axis_name))
        reg.figure(
            "guides", f"purity_by_{safe}",
            f"Guide purity by {axis_name}",
            plot_purity_by_group(
                ga.stats, pd.Series(s), str(axis_name), gcfg, fcfg,
                fig_dir / f"guide_purity_by_{safe}.png",
            ),
            caption=(
                f"Assignment criterion (top) and purity triangle (bottom) for "
                f"each level of <b>{axis_name}</b>, with the same cut-offs drawn "
                f"as the pooled panels. Compare the percentage above the cut-off "
                f"across levels: a difference there is a difference in guide "
                f"capture, not in biology, and it propagates silently into every "
                f"per-condition number downstream."
            ),
            order=22 + i, width="full",
        )

    reg.figure(
        "guides", "moi", "Guides per cell",
        plot_moi(ga.per_cell, fcfg, fig_dir / "guide_moi.png"),
        caption=T.MOI_DESC, order=30,
    )

    per_guide, per_target = guide_representation(guide, ga.per_cell, ga.mapping)
    per_guide.to_csv(table_dir / "guide_representation.csv", index=False)
    per_target.to_csv(table_dir / "target_representation.csv", index=False)

    group_matrices: dict[str, np.ndarray] = {}
    if primary is not None:
        g = primary.astype(str).reindex(ga.per_cell.index).to_numpy()
        for value in sorted(pd.unique(g[pd.notna(g)])):
            group_matrices[str(value)] = guide.X[g == value].sum(axis=0)

    reg.figure(
        "guides", "abundance", "Guide library representation",
        plot_abundance(guide, per_guide, group_matrices, fcfg,
                       fig_dir / "guide_abundance.png"),
        caption=T.GUIDE_ABUNDANCE_DESC, order=40, width="full",
    )
    reg.table(
        "guides", "target_table", "Cells per target",
        path=table_dir / "target_representation.csv",
        inline=per_target.head(cfg.report.max_table_rows).to_dict("records"),
        columns=list(per_target.columns),
        caption=(
            "Targets with very few assigned cells cannot support a perturbation "
            "call regardless of how well the rest of the experiment worked."
        ),
        order=60,
    )

    for i, (axis_name, series) in enumerate(group_columns.items()):
        eff = guide_efficiency_by_group(ga.per_cell, series,
                                        min_reads=gcfg.min_reads)
        if eff.empty:
            continue
        eff.to_csv(table_dir / f"guide_efficiency_by_{axis_name}.csv", index=False)
        reg.figure(
            "guides", f"efficiency_{axis_name}", f"Guide performance by {axis_name}",
            plot_efficiency(eff, axis_name, fcfg,
                            fig_dir / f"guide_efficiency_by_{axis_name}.png"),
            caption=T.GUIDE_EFFICIENCY_DESC, order=50 + i, width="full",
        )
        reg.table(
            "guides", f"efficiency_table_{axis_name}",
            # Distinct from the figure's title: report.py emits an <h3> from
            # the figure path and again from the table path, so a figure and a
            # table sharing one title printed the same heading twice.
            f"Guide performance by {axis_name} (table)",
            path=table_dir / f"guide_efficiency_by_{axis_name}.csv",
            inline=eff.to_dict("records"), columns=list(eff.columns),
            order=55 + i,
        )

    if ga.dual_guide:
        reg.figure(
            "guides", "pair_status", "Dual-guide pair composition",
            plot_pair_status(ga.per_cell, primary, fcfg,
                             fig_dir / "guide_pair_status.png"),
            caption=T.RECOMBINATION_DESC, order=45,
        )

    ga.per_cell.to_csv(table_dir / "guide_calls_per_cell.csv.gz", compression="gzip")
    # guide_target_mapping.csv is already written earlier in this stage; the
    # second identical write was a leftover.
    return ga
