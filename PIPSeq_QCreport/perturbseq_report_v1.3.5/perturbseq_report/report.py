"""
Self-contained HTML report builder.

Reads the artifact registry (``artifacts.json``) and renders exactly what the
analysis registered.  No globbing, no filename regexes, no "Other / additional
figures" bucket.

Adopted from the collaborator's report builder
----------------------------------------------
* The ``plot-desc`` / ``method-note`` / ``plot-row`` / ``plot-col`` structure,
  and their habit of putting interpretive prose next to every panel.
* Graceful degradation: a section whose inputs are missing is omitted entirely
  rather than rendered full of placeholders, and an individual missing figure
  becomes a labelled placeholder rather than a crash.
* A summary table at the top.

Fixed relative to both previous builders
----------------------------------------
* **Everything is escaped.** Neither previous builder escaped anything, so a
  gene name or path containing ``<`` or ``&`` corrupted the page. The lightbox
  caption in particular was assigned via ``innerHTML`` from an
  already-escaped attribute, which double-unescapes and re-parses as markup.
  Here captions go through ``textContent``.
* **Actually self-contained.** The previous report claimed to be safe to email
  while loading Google Fonts on every open. This uses a system font stack and
  base64-embeds figures, so the single ``.html`` file works offline.
* **Link mode paths are computed**, not hardcoded to ``figures/``. The old
  builder emitted ``figures/<name>`` regardless of where ``--out`` pointed, so
  every image broke unless the report happened to sit next to the figure
  directory.
* Figure order comes from the registry, not from lexicographic filename
  sorting (which put ``postfilter`` before ``prefilter``).
"""
from __future__ import annotations

import base64
import datetime as _dt
import html
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .artifacts import (
    Artifact, Registry, SECTION_BLURBS, SECTION_ORDER, SECTION_TITLES,
)
from .config import ReportConfig
from . import text as T


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# ===========================================================================
# CSS
# ===========================================================================
CSS = """
:root {
  --bg: #ffffff;
  --panel: #ffffff;
  --ink: #1b1f24;
  --ink-soft: #4a5259;
  --ink-faint: #7a838c;
  --line: #e3e7eb;
  --line-soft: #f0f3f5;
  --accent: #2f5d8a;
  --accent-soft: #eef4fa;
  --good: #2f7d4f;
  --good-soft: #ecf7f0;
  --warn: #9a6414;
  --warn-soft: #fdf4e5;
  --poor: #a3342f;
  --poor-soft: #fdeeed;
  --radius: 8px;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
          "Liberation Mono", monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
          Arial, "Noto Sans", sans-serif;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--sans); font-size: 15px; line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); }

/* ---------- layout ---------- */
.wrap { display: flex; align-items: flex-start; max-width: 1680px; margin: 0 auto; }
nav {
  position: sticky; top: 0; align-self: flex-start; flex: 0 0 250px;
  height: 100vh; overflow-y: auto; padding: 26px 18px 40px 24px;
  border-right: 1px solid var(--line); background: var(--bg);
}
nav .navtitle {
  font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
  color: var(--ink-faint); margin-bottom: 12px; font-weight: 600;
}
nav ol { list-style: none; margin: 0; padding: 0; counter-reset: sec; }
nav ol li { counter-increment: sec; margin: 1px 0; }
nav ol li a {
  display: block; padding: 6px 10px; border-radius: 6px; text-decoration: none;
  color: var(--ink-soft); font-size: 13.5px; border-left: 2px solid transparent;
}
nav ol li a::before { content: counter(sec) ". "; color: var(--ink-faint); }
nav ol li a:hover { background: var(--line-soft); color: var(--ink); }
nav ol li a.active {
  background: var(--accent-soft); color: var(--accent); font-weight: 600;
  border-left-color: var(--accent);
}
main { flex: 1 1 auto; min-width: 0; padding: 30px 34px 90px 34px; }

/* ---------- header ---------- */
header.top { border-bottom: 1px solid var(--line); padding-bottom: 20px; margin-bottom: 8px; }
header.top h1 { font-size: 25px; margin: 0 0 4px 0; letter-spacing: -.01em; }
header.top .sub { color: var(--ink-soft); font-size: 14.5px; }
header.top .meta {
  margin-top: 12px; font-family: var(--mono); font-size: 11.5px;
  color: var(--ink-faint); display: flex; flex-wrap: wrap; gap: 6px 18px;
}

/* ---------- sections ---------- */
section { padding-top: 30px; scroll-margin-top: 12px; }
section > h2 {
  font-size: 19px; margin: 0 0 3px 0; counter-increment: s;
  padding-bottom: 7px; border-bottom: 1px solid var(--line);
}
section > h2::before { content: counter(s) ". "; color: var(--ink-faint); }
body { counter-reset: s; }
section > .blurb { color: var(--ink-faint); font-size: 13.5px; margin: 6px 0 4px 0; }
h3.item { font-size: 15px; margin: 22px 0 2px 0; font-weight: 600; }

/* ---------- prose ---------- */
.plot-desc { color: var(--ink-soft); font-size: 13.5px; margin: 4px 0 10px 0; }
.plot-desc ul { margin: 6px 0 6px 20px; padding: 0; }
.plot-desc li { margin: 2px 0; }
.method-note {
  border-left: 3px solid var(--accent); background: var(--accent-soft);
  padding: 11px 14px; border-radius: 0 var(--radius) var(--radius) 0;
  font-size: 13.5px; color: var(--ink-soft); margin: 12px 0;
}
.method-note.warn { border-left-color: var(--warn); background: var(--warn-soft); }
.method-note.poor { border-left-color: var(--poor); background: var(--poor-soft); }
.method-note .nt {
  display: block; font-weight: 600; color: var(--ink); margin-bottom: 3px;
  font-size: 13px;
}

/* ---------- figures ---------- */
.plot-row { display: flex; flex-wrap: wrap; gap: 20px; margin: 6px 0 4px 0; }
.plot-col { flex: 1 1 440px; min-width: 320px; }
.plot-col.full-width { flex: 1 1 100%; }
.figbox {
  border: 1px solid var(--line); border-radius: var(--radius);
  background: var(--panel); padding: 8px; overflow: hidden;
}
.figbox img {
  width: 100%; height: auto; display: block; cursor: zoom-in; border-radius: 3px;
}
.plot-col.missing .figbox {
  border-style: dashed; border-color: #d9b9b7; background: var(--poor-soft);
  padding: 16px; color: var(--poor); font-size: 13px;
}
.missing-title { font-weight: 600; margin-bottom: 3px; }

/* ---------- metrics ---------- */
.metrics { display: flex; flex-wrap: wrap; gap: 12px; margin: 14px 0 4px 0; }
.metric {
  flex: 0 1 176px; border: 1px solid var(--line); border-radius: var(--radius);
  padding: 12px 14px; background: var(--panel);
}
.metric .mv { font-size: 22px; font-weight: 650; letter-spacing: -.02em; }
.metric .mu { font-size: 13px; font-weight: 500; color: var(--ink-soft); }
.metric .ml {
  font-size: 11.5px; color: var(--ink-faint); text-transform: uppercase;
  letter-spacing: .05em; margin-top: 2px;
}
.metric.good { border-color: #b9dcc6; background: var(--good-soft); }
.metric.good .mv { color: var(--good); }
.metric.warn { border-color: #ecd4a8; background: var(--warn-soft); }
.metric.warn .mv { color: var(--warn); }
.metric.poor { border-color: #e5b7b4; background: var(--poor-soft); }
.metric.poor .mv { color: var(--poor); }

/* ---------- tables ---------- */
.tablewrap { overflow-x: auto; margin: 8px 0; border: 1px solid var(--line);
  border-radius: var(--radius); }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
thead th {
  background: var(--line-soft); text-align: left; padding: 7px 11px;
  font-weight: 600; border-bottom: 1px solid var(--line); white-space: nowrap;
  position: sticky; top: 0;
}
tbody td { padding: 6px 11px; border-bottom: 1px solid var(--line-soft);
  white-space: nowrap; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: #fbfcfd; }
td.num { text-align: right; font-variant-numeric: tabular-nums;
  font-family: var(--mono); font-size: 11.5px; }
.tablenote { font-size: 12px; color: var(--ink-faint); padding: 5px 11px; }
code { font-family: var(--mono); font-size: .9em; background: var(--line-soft);
  padding: 1px 4px; border-radius: 3px; }

/* ---------- checklist ---------- */
ul.checklist { list-style: none; padding: 0; margin: 8px 0; }
ul.checklist li {
  border: 1px solid var(--line); border-radius: var(--radius);
  padding: 10px 13px; margin: 7px 0; font-size: 13.5px; color: var(--ink-soft);
  display: flex; gap: 10px;
}
ul.checklist li::before { content: "\\2610"; color: var(--ink-faint); font-size: 15px; }

/* ---------- lightbox ---------- */
#lb {
  position: fixed; inset: 0; background: rgba(12,15,18,.94); display: none;
  z-index: 999; padding: 26px; cursor: zoom-out;
}
#lb.on { display: flex; flex-direction: column; align-items: center;
  justify-content: center; }
#lb img { max-width: 98%; max-height: 88%; object-fit: contain;
  background: #fff; border-radius: 4px; }
#lbcap { color: #e8ecef; font-size: 13px; margin-top: 14px; text-align: center;
  max-width: 900px; }
#lbclose { position: absolute; top: 16px; right: 22px; color: #e8ecef;
  font-size: 30px; line-height: 1; cursor: pointer; }

footer {
  margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--line);
  color: var(--ink-faint); font-size: 12px;
}

@media (max-width: 1000px) {
  .wrap { flex-direction: column; }
  nav { position: static; height: auto; width: 100%; flex: none;
    border-right: none; border-bottom: 1px solid var(--line); padding: 16px 20px; }
  nav ol { display: flex; flex-wrap: wrap; gap: 4px; }
  main { padding: 20px; }
}
@media print {
  nav, #lb { display: none !important; }
  .wrap { display: block; }
  main { padding: 0; }
  section { page-break-inside: avoid; }
  .figbox img { cursor: default; }
  a { text-decoration: none; color: var(--ink); }
}
"""

JS = """
(function () {
  var lb = document.getElementById('lb');
  var img = document.getElementById('lbimg');
  var cap = document.getElementById('lbcap');

  document.querySelectorAll('.figbox img').forEach(function (el) {
    el.addEventListener('click', function () {
      img.src = el.currentSrc || el.src;
      // textContent, not innerHTML: the caption is plain text and assigning it
      // as HTML would re-parse already-escaped entities as markup.
      cap.textContent = el.getAttribute('data-caption') || '';
      lb.classList.add('on');
    });
  });
  function close() { lb.classList.remove('on'); img.src = ''; }
  lb.addEventListener('click', close);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') close();
  });

  var links = Array.prototype.slice.call(document.querySelectorAll('nav a'));
  var sections = links.map(function (a) {
    return document.querySelector(a.getAttribute('href'));
  });
  function onScroll() {
    var best = 0;
    for (var i = 0; i < sections.length; i++) {
      if (sections[i] && sections[i].getBoundingClientRect().top <= 90) best = i;
    }
    links.forEach(function (a, i) { a.classList.toggle('active', i === best); });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
"""


# ===========================================================================
# Rendering helpers
# ===========================================================================
def _embed_image(path: Path) -> str | None:
    """Base64 data URI for a figure, or None if unreadable."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _relative_src(fig_path: Path, out_path: Path) -> str:
    """URL for link mode, computed from the actual output location."""
    try:
        rel = os.path.relpath(fig_path, out_path.parent)
    except ValueError:  # different drives on Windows
        rel = str(fig_path)
    return "/".join(rel.split(os.sep))


def _format_cell(value: Any) -> tuple[str, bool]:
    """Render one table cell; returns (html, is_numeric)."""
    if value is None:
        return "&ndash;", False
    if isinstance(value, bool):
        return ("yes" if value else "no"), False
    if isinstance(value, (int,)):
        return f"{value:,}", True
    if isinstance(value, float):
        if value != value:
            return "&ndash;", True
        if value == int(value) and abs(value) < 1e15:
            return f"{int(value):,}", True
        if abs(value) < 1e-4 and value != 0:
            return f"{value:.2e}", True
        return f"{value:,.4g}", True
    return esc(value), False


def _render_table(rows: Sequence[dict], columns: Sequence[str] | None,
                  max_rows: int) -> str:
    if not rows:
        return '<p class="tablenote">No rows.</p>'
    cols = list(columns) if columns else list(rows[0].keys())
    shown = rows[:max_rows]
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = []
    for r in shown:
        tds = []
        for c in cols:
            txt, num = _format_cell(r.get(c))
            tds.append(f'<td class="num">{txt}</td>' if num else f"<td>{txt}</td>")
        body.append(f"<tr>{''.join(tds)}</tr>")
    note = ""
    if len(rows) > max_rows:
        note = (
            f'<p class="tablenote">Showing {max_rows:,} of {len(rows):,} rows; '
            f"the full table is in the CSV alongside this report.</p>"
        )
    return (
        f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>{note}"
    )


def _render_metric(a: Artifact) -> str:
    value = a.data.get("value")
    unit = a.data.get("unit", "")
    level = a.data.get("level", "info")
    cls = level if level in ("good", "warn", "poor") else ""
    txt, _ = _format_cell(value)
    return (
        f'<div class="metric {cls}">'
        f'<div class="mv">{txt}<span class="mu">{esc(unit)}</span></div>'
        f'<div class="ml">{esc(a.title)}</div></div>'
    )


def _render_note(a: Artifact) -> str:
    level = a.data.get("level", "info")
    cls = level if level in ("warn", "poor") else ""
    # Note bodies are authored HTML from text.py, not user data, so they are
    # intentionally not escaped. Anything interpolated INTO them was escaped at
    # construction time via text.esc().
    return (
        f'<div class="method-note {cls}"><span class="nt">{esc(a.title)}</span>'
        f"{a.data.get('body', '')}</div>"
    )


def _render_figure(
    a: Artifact, root: Path, out_path: Path, cfg: ReportConfig
) -> str:
    width_cls = "full-width" if a.width == "full" else ""
    caption_html = a.caption or ""
    plain_caption = html.unescape(
        __import__("re").sub(r"<[^>]+>", " ", caption_html)
    ).strip()

    if a.skipped_reason is not None or a.path is None:
        if not cfg.show_missing_placeholders:
            return ""
        return (
            f'<div class="plot-col {width_cls} missing"><div class="figbox">'
            f'<div class="missing-title">{esc(a.title)} &mdash; not produced</div>'
            f"<div>{esc(a.skipped_reason or 'no output path recorded')}</div>"
            f"</div></div>"
        )

    fig_path = (root / a.path) if not Path(a.path).is_absolute() else Path(a.path)
    if not fig_path.exists():
        return (
            f'<div class="plot-col {width_cls} missing"><div class="figbox">'
            f'<div class="missing-title">{esc(a.title)} &mdash; file missing</div>'
            f"<div>The analysis registered <code>{esc(a.path)}</code> but the file "
            f"is not on disk.</div></div></div>"
        )

    if cfg.embed_figures:
        src = _embed_image(fig_path)
        if src is None:
            src = _relative_src(fig_path, out_path)
    else:
        src = _relative_src(fig_path, out_path)

    return (
        f'<div class="plot-col {width_cls}">'
        f'<h3 class="item">{esc(a.title)}</h3>'
        + (f'<div class="plot-desc">{caption_html}</div>' if caption_html else "")
        + f'<div class="figbox"><img src="{esc(src)}" alt="{esc(a.title)}" '
          f'loading="lazy" data-caption="{esc(plain_caption)}"></div>'
        f"</div>"
    )


def _render_section(
    key: str, items: Sequence[Artifact], root: Path, out_path: Path,
    cfg: ReportConfig,
) -> str:
    """Render one section, or return '' if it has nothing real in it.

    The collaborator's whole-section omission pattern: a section is dropped
    entirely when it contains no produced artifact, so an experiment without
    hashtags simply has no hashtag section.
    """
    produced = [a for a in items if a.skipped_reason is None]
    if not produced:
        # Keep a single explanatory line if everything in the section was
        # skipped for a stated reason -- "this experiment had no hashtags" is
        # worth one sentence, but not a whole section of placeholders.
        reasons = [a for a in items if a.skipped_reason]
        if not reasons:
            return ""
        body = "".join(
            f'<div class="method-note"><span class="nt">{esc(a.title)} '
            f"&mdash; not applicable</span>{esc(a.skipped_reason)}</div>"
            for a in reasons[:1]
        )
        return (
            f'<section id="sec-{esc(key)}"><h2>{esc(SECTION_TITLES[key])}</h2>'
            f"{body}</section>"
        )

    metrics = [a for a in produced if a.kind == "metric"]
    notes = [a for a in items if a.kind == "note"]
    figures = [a for a in items if a.kind == "figure"]
    tables = [a for a in produced if a.kind == "table"]

    parts = [
        f'<section id="sec-{esc(key)}">',
        f"<h2>{esc(SECTION_TITLES[key])}</h2>",
        f'<div class="blurb">{esc(SECTION_BLURBS[key])}</div>',
    ]
    if metrics:
        parts.append(
            '<div class="metrics">'
            + "".join(_render_metric(a) for a in metrics)
            + "</div>"
        )
    for a in sorted(notes, key=lambda x: x.order):
        parts.append(_render_note(a))

    if figures:
        rendered = [_render_figure(a, root, out_path, cfg) for a in figures]
        rendered = [r for r in rendered if r]
        if rendered:
            parts.append(f'<div class="plot-row">{"".join(rendered)}</div>')

    if cfg.include_data_tables:
        for a in sorted(tables, key=lambda x: x.order):
            parts.append(f'<h3 class="item">{esc(a.title)}</h3>')
            if a.caption:
                parts.append(f'<div class="plot-desc">{a.caption}</div>')
            parts.append(
                _render_table(a.data.get("rows", []), a.data.get("columns"),
                              cfg.max_table_rows)
            )
    parts.append("</section>")
    return "".join(parts)


# ===========================================================================
# Public entry point
# ===========================================================================
def build_report(
    registry: Registry,
    out_path: Path,
    cfg: ReportConfig,
    title: str,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Render the report to ``out_path`` and return it."""
    root = registry.root
    by_section = registry.by_section()

    rendered: list[tuple[str, str]] = []
    for key in sorted(by_section, key=lambda k: SECTION_ORDER[k]):
        if key == "appendix":
            continue
        html_part = _render_section(key, by_section[key], root, out_path, cfg)
        if html_part:
            rendered.append((key, html_part))

    # --- appendix: notes (input state, modality detection, whitelist
    # validation, sanity checks, ...) + checklist + provenance -------------
    #
    # Every module in the pipeline calls reg.note("appendix", ...) to surface
    # exactly the kind of thing that explains a downstream failure -- "guide
    # feature names could not be resolved and are placeholders", "321 of 321
    # guides are not in the gRNA whitelist", "input matrix failed plausibility
    # checks", and so on. This section used to be rebuilt from ONLY the static
    # checklist and provenance table, silently discarding every one of those
    # notes (they were still being collected in `by_section["appendix"]`, just
    # never rendered) -- so a run could fail in an explainable way and the
    # report would show nothing about why. Notes are rendered first, in the
    # same order used everywhere else (`order`), followed by the checklist and
    # provenance table as before.
    appendix_items = by_section.get("appendix", [])
    appendix_notes = sorted(
        (a for a in appendix_items if a.kind == "note"), key=lambda x: x.order
    )
    appendix_notes_html = "".join(_render_note(a) for a in appendix_notes)

    checklist = "".join(f"<li>{esc(x)}</li>" for x in T.FINAL_CHECKLIST)
    prov_rows = [
        {"item": k, "value": v} for k, v in (provenance or {}).items()
    ]
    appendix = (
        '<section id="sec-appendix"><h2>Appendix</h2>'
        f'<div class="blurb">{esc(SECTION_BLURBS["appendix"])}</div>'
        + appendix_notes_html
        + '<h3 class="item">Before treating these results as final</h3>'
        '<div class="plot-desc">This report can tell you whether the assay worked. '
        "It cannot tell you whether the conclusion you want to draw from it is "
        "sound. These are the checks it cannot do for you.</div>"
        f'<ul class="checklist">{checklist}</ul>'
        '<h3 class="item">Run provenance</h3>'
        + _render_table(prov_rows, ["item", "value"], 200)
        + "</section>"
    )
    rendered.append(("appendix", appendix))

    nav = "".join(
        f'<li><a href="#sec-{esc(k)}">{esc(SECTION_TITLES[k])}</a></li>'
        for k, _ in rendered
    )

    n_fig = sum(
        1 for a in registry if a.kind == "figure" and a.skipped_reason is None
    )
    n_skipped = sum(1 for a in registry if a.skipped_reason is not None)
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    meta_bits = [
        f"generated {generated}",
        f"{n_fig} figures",
    ]
    if n_skipped:
        meta_bits.append(f"{n_skipped} not applicable")
    if provenance:
        for k in ("pipeline version", "cells analysed", "backend"):
            if k in provenance:
                meta_bits.append(f"{k}: {provenance[k]}")

    doc = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head><body>
<div class="wrap">
<nav><div class="navtitle">Contents</div><ol>{nav}</ol></nav>
<main>
<header class="top">
  <h1>{esc(title)}</h1>
  <div class="sub">{esc(cfg.subtitle)}</div>
  <div class="meta">{''.join(f'<span>{esc(b)}</span>' for b in meta_bits)}</div>
</header>
{''.join(part for _, part in rendered)}
<footer>
  Generated by the Perturb-seq report pipeline. Figures are embedded in this
  file, so it can be archived or emailed as a single document. Every number
  shown here is also written as CSV next to the report.
</footer>
</main>
</div>
<div id="lb"><span id="lbclose">&times;</span><img id="lbimg" alt=""><div id="lbcap"></div></div>
<script>{JS}</script>
</body></html>
"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


def build_from_artifacts(
    artifacts_path: Path, out_path: Path | None = None,
    cfg: ReportConfig | None = None, title: str | None = None,
) -> Path:
    """Rebuild the report from a saved ``artifacts.json``, without re-analysing.

    This is the useful half of the original's ``--skip-notebook``: regenerate
    the HTML after tweaking a figure, in seconds, with no risk of the analysis
    re-running.
    """
    reg = Registry.load(artifacts_path)
    cfg = cfg or ReportConfig()
    out = Path(out_path) if out_path else (reg.root / "qc_report.html")
    return build_report(reg, out, cfg, title or "Perturb-seq report", {})
