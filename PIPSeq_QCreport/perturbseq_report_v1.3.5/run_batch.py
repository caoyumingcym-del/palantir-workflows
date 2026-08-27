#!/usr/bin/env python3
"""
Run the pipeline over several sample manifests in one go.

    python run_batch.py --manifests sample_manifest_*.csv

Why not a plain shell loop
--------------------------
A bare ``for f in *.csv; do python run_perturbseq_report.py --manifest "$f"; done``
mostly works, but has four failure modes that matter on a real batch:

1. **The pipeline is explore-first**, so the first pass over a set of fresh
   manifests produces seven *explore* reports, not seven full ones. That is
   correct behaviour, but it surprises people who expected finished reports.
   This script says up front what each manifest is going to do.

2. **Two manifests can share an ``output_path``**, in which case the second run
   silently overwrites the first's report, figures and tables. Nothing in the
   single-manifest pipeline can detect this, because it only ever sees one
   manifest. This script refuses to start until the collision is resolved.

3. **One bad manifest kills the loop** unless you remember ``|| true``, and then
   you lose the exit codes and cannot tell which runs actually succeeded.

4. **Memory is not reclaimed** between experiments if you loop in-process. Each
   run here is a separate subprocess, so a large h5ad is fully released before
   the next one starts.

Every manifest is validated *before* any analysis begins, so a typo in the last
manifest fails in the first few seconds rather than an hour in.
"""
from __future__ import annotations

import argparse
import html
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from perturbseq_report.config import THRESHOLD_KEYS, decide_run_mode  # noqa: E402
from perturbseq_report.manifest import ManifestError, read_manifest   # noqa: E402
from perturbseq_report.version import __version__                     # noqa: E402

LAUNCHER = HERE / "run_perturbseq_report.py"

EXIT_LABELS = {
    0: "ok", 2: "usage error", 3: "manifest error",
    4: "pipeline error", 5: "unexpected error",
}


@dataclass
class Plan:
    """What one manifest is going to do, worked out before anything runs."""

    path: Path
    output_path: Path
    n_samples: int
    n_runs: int
    will_explore: bool
    reason: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    plan: Plan
    returncode: int
    seconds: float
    report: Path | None          # produced by THIS run
    log: Path
    stale_report: Path | None = None   # left over from an earlier run

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def status(self) -> str:
        return EXIT_LABELS.get(self.returncode, f"exit {self.returncode}")


# ===========================================================================
# Pre-flight
# ===========================================================================
def build_plans(paths: list[Path], mode: str) -> tuple[list[Plan], list[str]]:
    """Validate every manifest and work out what each will do."""
    plans: list[Plan] = []
    errors: list[str] = []

    for p in sorted(paths):
        try:
            m = read_manifest(p)
        except ManifestError as exc:
            errors.append(f"{p.name}: {exc}")
            continue

        try:
            th = m.read_thresholds()
        except ManifestError as exc:
            errors.append(f"{p.name}: {exc}")
            continue

        decision = decide_run_mode(
            manifest_thresholds=th.as_dict(),
            cli_thresholds={k: None for k in THRESHOLD_KEYS},
            explore_flag=(mode == "explore"),
            auto_flag=(mode == "auto"),
        )
        plans.append(
            Plan(
                path=p,
                output_path=m.output_path,
                n_samples=m.n_samples,
                n_runs=m.n_runs,
                will_explore=decision.explore,
                reason=decision.reason,
                warnings=list(m.warnings),
            )
        )
    return plans, errors


def find_collisions(plans: list[Plan]) -> dict[Path, list[Plan]]:
    """Manifests writing to the same output_path would overwrite each other."""
    by_out: dict[Path, list[Plan]] = defaultdict(list)
    for pl in plans:
        by_out[pl.output_path.resolve()].append(pl)
    return {k: v for k, v in by_out.items() if len(v) > 1}


# ===========================================================================
# Execution
# ===========================================================================
def run_one(plan: Plan, mode: str, log_dir: Path, extra: list[str]) -> Outcome:
    """Run one manifest in its own process, capturing output to a log file."""
    cmd = [sys.executable, str(LAUNCHER), "--manifest", str(plan.path)]
    if mode == "explore":
        cmd.append("--explore")
    elif mode == "auto":
        cmd.append("--auto-thresholds")
    cmd += extra

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{plan.path.stem}.log"

    name = "qc_explore.html" if plan.will_explore else "qc_report.html"
    report = plan.output_path / "analysis_outputs" / name
    # Note the existing report's timestamp BEFORE running. A previous run may
    # have left one here; if this run fails, that stale file must not be
    # reported as this run's output. Linking a month-old report next to a
    # FAILED status is worse than showing no link at all.
    before_mtime = report.stat().st_mtime if report.exists() else None

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE))
    seconds = time.time() - start

    produced: Path | None = None
    stale: Path | None = None
    if report.exists():
        after_mtime = report.stat().st_mtime
        if before_mtime is None or after_mtime > before_mtime:
            produced = report
        else:
            stale = report

    return Outcome(
        plan=plan, returncode=proc.returncode, seconds=seconds,
        report=produced, log=log_path, stale_report=stale,
    )


# ===========================================================================
# Summary
# ===========================================================================
def write_summary(outcomes: list[Outcome], out_dir: Path, mode: str) -> Path:
    """An index page linking every report, with status and timing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "batch_summary.html"

    def esc(x: object) -> str:
        return html.escape(str(x), quote=True)

    import os

    rows = []
    for o in sorted(outcomes, key=lambda x: x.plan.path.name):
        if o.report is not None:
            href = "/".join(
                os.path.relpath(o.report, index.parent).split(os.sep)
            )
            link = f'<a href="{esc(href)}">{esc(o.report.name)}</a>'
        elif o.stale_report is not None:
            href = "/".join(
                os.path.relpath(o.stale_report, index.parent).split(os.sep)
            )
            link = (
                f'<span class="muted">none from this run &mdash; '
                f'<a href="{esc(href)}">an older {esc(o.stale_report.name)}</a> '
                f'is still on disk</span>'
            )
        else:
            link = '<span class="muted">no report produced</span>'
        log_href = "/".join(os.path.relpath(o.log, index.parent).split(os.sep))
        cls = "ok" if o.ok else "bad"
        kind = "explore" if o.plan.will_explore else "full"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>{esc(o.plan.path.name)}</td>"
            f"<td>{esc(o.plan.n_samples)}</td>"
            f"<td><span class='pill {kind}'>{kind}</span></td>"
            f"<td>{esc(o.status)}</td>"
            f"<td class='num'>{o.seconds:,.0f}s</td>"
            f"<td>{link}</td>"
            f"<td><a href='{esc(log_href)}'>log</a></td>"
            f"</tr>"
        )

    n_ok = sum(1 for o in outcomes if o.ok)
    n_explore = sum(1 for o in outcomes if o.ok and o.plan.will_explore)
    banner = ""
    if n_explore:
        banner = (
            f"<div class='note'><strong>{n_explore} of these were QC review runs.</strong> "
            f"Nothing was filtered, clustered or quantified in them. Open each "
            f"<code>qc_explore.html</code>, adjust the five threshold columns in "
            f"its manifest if you disagree with the derived values, then re-run "
            f"this batch to produce the full reports.</div>"
        )

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Perturb-seq batch summary</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
margin:40px auto;max-width:1100px;color:#1b1f24;line-height:1.6}}
h1{{font-size:22px;margin-bottom:2px}} .sub{{color:#7a838c;font-size:13px}}
table{{border-collapse:collapse;width:100%;margin-top:18px;font-size:13.5px}}
th{{text-align:left;background:#f0f3f5;padding:8px 11px;border-bottom:1px solid #e3e7eb}}
td{{padding:7px 11px;border-bottom:1px solid #f0f3f5}}
tr.bad td{{background:#fdeeed}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.muted{{color:#9aa2a9}} a{{color:#2f5d8a}}
.pill{{font-size:11px;padding:2px 7px;border-radius:10px;font-weight:600}}
.pill.explore{{background:#fdf4e5;color:#9a6414}}
.pill.full{{background:#ecf7f0;color:#2f7d4f}}
.note{{border-left:3px solid #9a6414;background:#fdf4e5;padding:11px 14px;
margin:16px 0;font-size:13.5px;border-radius:0 6px 6px 0}}
code{{background:#f0f3f5;padding:1px 4px;border-radius:3px;font-size:.9em}}
</style></head><body>
<h1>Perturb-seq batch summary</h1>
<div class="sub">{len(outcomes)} manifest(s), {n_ok} succeeded &middot;
mode: {esc(mode)} &middot; pipeline {esc(__version__)} &middot;
{esc(time.strftime('%Y-%m-%d %H:%M'))}</div>
{banner}
<table><thead><tr><th>manifest</th><th>samples</th><th>ran</th><th>status</th>
<th>time</th><th>report</th><th>log</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>
"""
    index.write_text(doc, encoding="utf-8")

    csv_path = out_dir / "batch_summary.csv"
    lines = [
        "manifest,samples,runs,mode,status,returncode,seconds,report,"
        "stale_report_on_disk,log"
    ]
    for o in sorted(outcomes, key=lambda x: x.plan.path.name):
        lines.append(
            ",".join(
                [
                    o.plan.path.name, str(o.plan.n_samples), str(o.plan.n_runs),
                    "explore" if o.plan.will_explore else "full",
                    o.status, str(o.returncode), f"{o.seconds:.1f}",
                    str(o.report or ""), str(o.stale_report or ""), str(o.log),
                ]
            )
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


# ===========================================================================
# CLI
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="run_batch",
        description="Run the Perturb-seq pipeline over several manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "  # Pass 1 over a fresh set: QC review for each experiment\n"
            "  python run_batch.py --manifests sample_manifest_*.csv\n\n"
            "  # ...review each qc_explore.html, edit thresholds, then pass 2\n"
            "  python run_batch.py --manifests sample_manifest_*.csv\n\n"
            "  # Skip review everywhere (quick first look at all experiments)\n"
            "  python run_batch.py --manifests sample_manifest_*.csv --mode auto\n\n"
            "  # See what would happen without running anything\n"
            "  python run_batch.py --manifests sample_manifest_*.csv --dry-run\n"
        ),
    )
    p.add_argument("--manifests", nargs="+", required=True, type=Path,
                   help="manifest files (shell globs are fine)")
    p.add_argument("--mode", choices=("default", "explore", "auto"),
                   default="default",
                   help="default: let each manifest decide (explore-first); "
                        "explore: force the QC step everywhere; "
                        "auto: skip review everywhere")
    p.add_argument("--summary-dir", type=Path, default=Path("batch_output"),
                   help="where to write logs and the summary (default: batch_output)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate and print the plan, then stop")
    p.add_argument("--keep-going", action="store_true", default=True,
                   help="continue after a failing manifest (default)")
    p.add_argument("--stop-on-error", dest="keep_going", action="store_false",
                   help="abort the batch at the first failure")
    p.add_argument("--allow-shared-output", action="store_true",
                   help="permit several manifests to share an output_path "
                        "(they WILL overwrite each other)")
    p.add_argument("--", dest="_sep", nargs=argparse.REMAINDER,
                   help="arguments after -- are passed to each pipeline run")
    args, passthrough = p.parse_known_args(argv)
    passthrough = [a for a in passthrough if a != "--"]

    # Expand any globs the shell did not (Windows, mainly).
    paths: list[Path] = []
    for entry in args.manifests:
        if any(ch in str(entry) for ch in "*?["):
            paths.extend(sorted(Path().glob(str(entry))))
        else:
            paths.append(entry)
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("error: no manifest files matched", file=sys.stderr)
        return 2

    print(f"perturbseq batch — {len(paths)} manifest(s), mode={args.mode}\n")
    sys.stdout.flush()

    plans, errors = build_plans(paths, args.mode)
    if errors:
        # stdout is flushed first so this block cannot be interleaved above the
        # header when the two streams are captured together.
        print("  These manifests could not be read and will be skipped:")
        for e in errors:
            print(f"    ✗ {e}")
        print()
        sys.stdout.flush()
        if not args.keep_going:
            print("--stop-on-error: aborting.", file=sys.stderr)
            return 3

    if not plans:
        print("error: no usable manifests", file=sys.stderr)
        return 3

    collisions = find_collisions(plans)
    if collisions and not args.allow_shared_output:
        print("error: several manifests write to the same output_path.",
              file=sys.stderr)
        print("       Their reports, figures and tables would overwrite each "
              "other.\n", file=sys.stderr)
        for out, group in collisions.items():
            print(f"  {out}", file=sys.stderr)
            for pl in group:
                print(f"      <- {pl.path.name}", file=sys.stderr)
        print("\nGive each manifest its own output_path, or pass "
              "--allow-shared-output if\nthis really is intended.",
              file=sys.stderr)
        return 3

    # ------------------------------------------------------------- the plan
    width = max(len(pl.path.name) for pl in plans)
    n_explore = sum(1 for pl in plans if pl.will_explore)
    for pl in plans:
        kind = "QC review" if pl.will_explore else "FULL     "
        print(f"  {pl.path.name:<{width}}  {kind}  "
              f"{pl.n_samples} sample(s) -> {pl.output_path}")
        for w in pl.warnings:
            print(f"  {'':<{width}}    warning: {w}")
    print()

    if n_explore:
        print(f"  {n_explore} of {len(plans)} will stop after QC. Nothing will be")
        print("  filtered or quantified in those. Review each qc_explore.html,")
        print("  adjust thresholds in its manifest, then run this again.\n")

    if args.dry_run:
        print("--dry-run: stopping here.")
        return 0

    # ------------------------------------------------------------------ run
    summary_dir = args.summary_dir.resolve()
    log_dir = summary_dir / "logs"
    outcomes: list[Outcome] = []

    for i, pl in enumerate(plans, 1):
        print(f"[{i}/{len(plans)}] {pl.path.name} ... ", end="", flush=True)
        outcome = run_one(pl, args.mode, log_dir, passthrough)
        outcomes.append(outcome)
        if outcome.ok:
            print(f"{outcome.status} ({outcome.seconds:,.0f}s)")
        else:
            print(f"FAILED — {outcome.status} (see {outcome.log})")
            sys.stdout.flush()
            if not args.keep_going:
                print("\n--stop-on-error: aborting.", file=sys.stderr)
                break

    index = write_summary(outcomes, summary_dir, args.mode)

    n_ok = sum(1 for o in outcomes if o.ok)
    n_bad = len(outcomes) - n_ok
    print()
    print(f"  {n_ok} succeeded, {n_bad} failed")
    print(f"  summary: {index}")
    if n_bad:
        print("\n  Failed manifests:", file=sys.stderr)
        for o in outcomes:
            if not o.ok:
                print(f"    {o.plan.path.name}: {o.status} — {o.log}",
                      file=sys.stderr)
    return 0 if n_bad == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
