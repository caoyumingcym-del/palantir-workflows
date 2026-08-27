"""
The artifact registry: how analysis output reaches the report.

This is the single biggest structural change from the original pipeline.

**Before.**  The analysis script wrote PNGs into a flat directory with
convention-encoded names (``qc_by_<axis>_<stage>.png``,
``purity_sweep_by_<axis>.png``, ...).  ``build_qc_report.py`` then globbed that
directory and ran each filename against an ordered list of ~40 regexes to guess
which section it belonged to and what caption to give it.  The consequences:

* Figures whose names didn't match any pattern fell into an "Other" bucket, so
  adding a plot upstream silently degraded the report.
* Two tools had to agree on a filename grammar with nothing enforcing it.
  ``run_pipeline.py`` then ``exec_module``'d the report script purely to reuse
  its regexes for stale-figure cleanup -- the coupling the comments claimed to
  be avoiding.
* Ordering was lexicographic, so ``postfilter`` sorted before ``prefilter``
  and panels appeared in the wrong order.
* Perturbation figures were written to a subdirectory the report's
  non-recursive glob couldn't see, so the analysis code duplicated every plot
  call into the flat directory to work around it.

**Now.**  A figure is registered at the moment it is created, together with its
section, title, caption and sort order.  The registry is serialised to
``artifacts.json``.  The report reads that file and renders exactly what the
analysis said it produced -- no globbing, no regexes, no filename grammar, and
no possibility of an unclassified figure.  Tables and scalar metrics go through
the same registry, which is what lets the report show numbers next to plots.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Literal

ArtifactKind = Literal["figure", "table", "metric", "note"]


# The ordered section spine of the report.  Sections with no artifacts are
# dropped at render time (the collaborator's graceful-omission pattern), so an
# experiment without hashtags simply has no hashtag section rather than a
# section full of "missing" placeholders.
SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("summary",      "Summary",                        "Headline numbers and the QC verdict for this experiment."),
    ("seq_qc",       "Sequencing QC",                  "Read-level metrics from the upstream alignment pipeline."),
    ("cell_qc",      "Per-cell QC & filtering",        "Which cells were kept, on what basis, and what was lost."),
    ("transcriptome","Transcriptome, embedding & clusters", "Normalisation, feature selection, embedding, clustering and markers."),
    ("guides",       "Guide assignment & performance", "Whether guides can be called, and how well."),
    ("perturbation", "Perturbation effects",           "Whether the perturbations did anything, and what."),
    ("hashtags",     "Hashtag performance",            "Demultiplexing quality and sample composition."),
    ("comparability","Condition comparability",        "Whether the conditions being compared are comparable in the first place."),
    ("crosschecks",  "Cross-checks",                   "Independent measurements compared against each other."),
    ("appendix",     "Appendix",                       "Configuration, provenance and the review checklist."),
)

SECTION_ORDER = {key: i for i, (key, _, _) in enumerate(SECTIONS)}
SECTION_TITLES = {key: title for key, title, _ in SECTIONS}
SECTION_BLURBS = {key: blurb for key, _, blurb in SECTIONS}


@dataclass
class Artifact:
    """One thing the analysis produced that the report may show."""

    kind: ArtifactKind
    section: str
    key: str                       # stable identifier, unique within a section
    title: str
    path: str | None = None        # relative to the analysis directory
    caption: str = ""
    order: int = 100
    width: Literal["half", "full"] = "half"
    # Free-form payload: scalar metrics, small tables rendered inline, the
    # provenance of a threshold, etc.
    data: dict[str, Any] = field(default_factory=dict)
    # Set when the analysis attempted this artifact and could not produce it.
    # Recording *why* something is absent is far more useful in a QC report
    # than silently omitting it -- "no hashtag matrix found in the h5ad" is
    # itself a finding.
    skipped_reason: str | None = None

    def exists(self, root: Path) -> bool:
        if self.path is None:
            return self.kind in ("metric", "note")
        return (root / self.path).exists()


class Registry:
    """Collects artifacts during a run and serialises them for the report."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._items: list[Artifact] = []
        self._seen: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ add
    def add(self, artifact: Artifact) -> Artifact:
        if artifact.section not in SECTION_ORDER:
            raise ValueError(
                f"Unknown report section {artifact.section!r}. Valid sections: "
                f"{', '.join(SECTION_ORDER)}"
            )
        ident = (artifact.section, artifact.key)
        if ident in self._seen:
            # A duplicate key means two code paths are producing the same
            # artifact -- exactly the drift that produced two near-identical
            # `plot_umap` definitions in the original file.  Fail loudly.
            raise ValueError(
                f"Duplicate artifact key {artifact.key!r} in section "
                f"{artifact.section!r}. Keys must be unique within a section."
            )
        self._seen.add(ident)
        self._items.append(artifact)
        return artifact

    def figure(
        self,
        section: str,
        key: str,
        title: str,
        path: Path | str,
        caption: str = "",
        order: int = 100,
        width: Literal["half", "full"] = "half",
        **data: Any,
    ) -> Artifact:
        rel = self._relative(path)
        return self.add(
            Artifact(
                kind="figure", section=section, key=key, title=title,
                path=rel, caption=caption, order=order, width=width, data=data,
            )
        )

    def table(
        self,
        section: str,
        key: str,
        title: str,
        path: Path | str | None = None,
        caption: str = "",
        order: int = 100,
        inline: list[dict[str, Any]] | None = None,
        columns: list[str] | None = None,
        **data: Any,
    ) -> Artifact:
        payload = dict(data)
        if inline is not None:
            payload["rows"] = inline
        if columns is not None:
            payload["columns"] = columns
        return self.add(
            Artifact(
                kind="table", section=section, key=key, title=title,
                path=self._relative(path) if path is not None else None,
                caption=caption, order=order, width="full", data=payload,
            )
        )

    def metric(
        self,
        section: str,
        key: str,
        title: str,
        value: Any,
        unit: str = "",
        caption: str = "",
        order: int = 10,
        level: str = "info",
        **data: Any,
    ) -> Artifact:
        return self.add(
            Artifact(
                kind="metric", section=section, key=key, title=title,
                caption=caption, order=order,
                data={"value": value, "unit": unit, "level": level, **data},
            )
        )

    def note(
        self,
        section: str,
        key: str,
        title: str,
        body: str,
        order: int = 200,
        level: str = "info",
    ) -> Artifact:
        return self.add(
            Artifact(
                kind="note", section=section, key=key, title=title,
                order=order, data={"body": body, "level": level},
            )
        )

    def skipped(
        self,
        section: str,
        key: str,
        title: str,
        reason: str,
        order: int = 100,
        kind: ArtifactKind = "figure",
    ) -> Artifact:
        """Record that something was deliberately not produced, and why."""
        return self.add(
            Artifact(
                kind=kind, section=section, key=key, title=title,
                order=order, skipped_reason=reason,
            )
        )

    # -------------------------------------------------------------- helpers
    def _relative(self, path: Path | str) -> str:
        p = Path(path)
        try:
            return str(p.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            # Outside the analysis root: store the absolute path so the report
            # can still find it, rather than silently dropping the artifact.
            return str(p).replace("\\", "/")

    def has_section(self, section: str) -> bool:
        return any(
            a.section == section and a.skipped_reason is None for a in self._items
        )

    def by_section(self) -> dict[str, list[Artifact]]:
        out: dict[str, list[Artifact]] = {}
        for a in sorted(
            self._items, key=lambda x: (SECTION_ORDER[x.section], x.order, x.key)
        ):
            out.setdefault(a.section, []).append(a)
        return out

    def get(self, section: str, key: str) -> Artifact | None:
        for a in self._items:
            if a.section == section and a.key == key:
                return a
        return None

    def metrics(self) -> dict[str, Any]:
        return {
            a.key: a.data.get("value")
            for a in self._items
            if a.kind == "metric"
        }

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterable[Artifact]:
        return iter(self._items)

    # ------------------------------------------------------------- serialise
    def verify(self) -> list[str]:
        """Cross-check every registered figure/table against the filesystem.

        Keeps the one genuinely valuable check from the original
        (``verify_claimed_figures_exist``): if the analysis says it wrote a
        figure and the file is not there, that is a bug worth surfacing rather
        than a blank space in the report.
        """
        problems = []
        for a in self._items:
            if a.skipped_reason is not None or a.path is None:
                continue
            if not (self.root / a.path).exists():
                problems.append(
                    f"{a.section}/{a.key}: registered {a.path} but the file does "
                    f"not exist"
                )
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "root": str(self.root),
            "sections": [
                {"key": k, "title": t, "blurb": b} for k, t, b in SECTIONS
            ],
            "artifacts": [asdict(a) for a in self._items],
        }

    def save(self, path: Path | None = None) -> Path:
        target = Path(path) if path else (self.root / "artifacts.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> "Registry":
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        # Root is taken from the file's own location, not the recorded absolute
        # path, so a report can be rebuilt after the directory has been moved
        # or copied to another machine.  The original hardcoded ``figures/`` as
        # the link prefix, which broke every image if --out was not a sibling
        # of the figures directory.
        reg = cls(p.parent)
        known = set(Artifact.__dataclass_fields__)
        for item in payload.get("artifacts", []):
            clean = {k: v for k, v in item.items() if k in known}
            reg._items.append(Artifact(**clean))
            reg._seen.add((clean["section"], clean["key"]))
        return reg
