"""
Command-line interface.

The pipeline is EXPLORE-FIRST. The same command is run twice:

    # 1. QC only. Stops before anything is filtered. Fills the manifest's five
    #    threshold columns with values derived from these distributions.
    perturbseq-report --manifest sample_manifest.csv

    # 2. Review the QC panels, adjust those numbers in the manifest if you
    #    disagree, then run the identical command again for the full report.
    perturbseq-report --manifest sample_manifest.csv

The second invocation runs the full pipeline because the manifest now carries
all five thresholds. Nothing is filtered, clustered or quantified on numbers
that were not written into the manifest first.

Threshold precedence, lowest to highest:

    auto (from the data)  <  config file  <  manifest columns  <  CLI flags

Escape hatches: ``--auto-thresholds`` runs end-to-end in one pass on derived
values (for batch jobs and re-runs of near-identical experiments), and
``--explore`` forces the QC-only stop even when thresholds are already set.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .config import THRESHOLD_KEYS, ModalityConfig, build_config, decide_run_mode
from .manifest import GLOBAL_COLUMNS, ManifestError, read_manifest, write_manifest_template
from .pipeline import PipelineError, run
from .report import build_from_artifacts
from .version import __version__

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MANIFEST = 3
EXIT_PIPELINE = 4
EXIT_UNEXPECTED = 5


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="perturbseq-report",
        description=(
            "Run a Perturb-seq QC and analysis pipeline from a sample manifest "
            "and write a single self-contained HTML report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The normal workflow -- run the same command twice\n"
            "-------------------------------------------------\n"
            "  # 1. QC only. Stops before filtering; writes qc_explore.html and\n"
            "  #    fills the manifest's five threshold columns.\n"
            "  perturbseq-report --manifest sample_manifest.csv\n\n"
            "  # 2. Review qc_explore.html, adjust the thresholds in the manifest\n"
            "  #    if you disagree, then run the SAME command for qc_report.html.\n"
            "  perturbseq-report --manifest sample_manifest.csv\n\n"
            "Other modes\n"
            "-----------\n"
            "  # One pass on derived thresholds, no review step (batch jobs)\n"
            "  perturbseq-report --manifest sample_manifest.csv --auto-thresholds\n\n"
            "  # Set thresholds explicitly; runs the full pipeline immediately\n"
            "  perturbseq-report --manifest sample_manifest.csv \\\n"
            "      --min-genes 800 --max-genes 6000 --min-counts 1000 \\\n"
            "      --max-counts 30000 --max-mito 15\n\n"
            "  # Re-do the QC-only step even though thresholds are already set\n"
            "  perturbseq-report --manifest sample_manifest.csv --explore\n\n"
            "  # Rebuild the HTML from an existing run, without re-analysing\n"
            "  perturbseq-report --rebuild-report OUT/analysis_outputs/artifacts.json\n\n"
            "  # Create a blank manifest to fill in\n"
            "  perturbseq-report --init-manifest ./sample_manifest.csv\n"
        ),
    )
    p.add_argument("--manifest", type=Path, help="path to the sample manifest CSV/TSV")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    g = p.add_argument_group("alternative modes")
    g.add_argument("--init-manifest", type=Path, metavar="PATH",
                   help="write a blank manifest template and exit")
    g.add_argument("--rebuild-report", type=Path, metavar="ARTIFACTS_JSON",
                   help="rebuild the HTML from a previous run's artifacts.json "
                        "and exit (fast; does not re-analyse)")
    g.add_argument("--inspect", type=Path, metavar="H5AD",
                   help="report an .h5ad's size, sparsity and memory cost "
                        "WITHOUT loading it, then exit. Run this first if the "
                        "pipeline is being killed with no output.")

    g = p.add_argument_group("paths")
    g.add_argument("--output-path", type=Path,
                   help="override the manifest's output_path column")
    g.add_argument("--h5ad", type=Path, dest="h5ad_path",
                   help="override the manifest's h5ad_path column")
    g.add_argument("--dragen-root", type=Path, nargs="+", metavar="DIR",
                   help="one or more directories, each containing some of "
                        "the runs' DRAGEN output subfolders (each named the "
                        "same way the manifest's own dragen_path values "
                        "already end -- their basename). Every root is "
                        "tried for every run; overrides that run's "
                        "dragen_path to whichever <root>/<that basename> "
                        "actually exists. Several roots are for when runs' "
                        "DRAGEN output doesn't share one common parent "
                        "directory. For a platform that doesn't mount "
                        "project data into the task, the same reason "
                        "--h5ad exists")
    g.add_argument("--config", type=Path,
                   help="JSON or YAML file of configuration overrides")
    g.add_argument("--counts-layer", type=str, dest="counts_layer",
                   metavar="NAME",
                   help="read raw counts from adata.layers[NAME] instead of X. "
                        "Use when X holds normalised or log-transformed values.")
    g.add_argument("--subsample-cells", type=int, dest="subsample_cells",
                   metavar="N",
                   help="randomly subsample to at most N cells after loading. "
                        "Lets a very large experiment complete on a machine that "
                        "cannot hold the full matrix; recorded in the report.")

    g = p.add_argument_group("QC thresholds (omit any to derive it from the data)")
    for key, help_text in (
        ("min_genes", "minimum genes detected per cell"),
        ("max_genes", "maximum genes detected per cell"),
        ("min_counts", "minimum total UMI counts per cell"),
        ("max_counts", "maximum total UMI counts per cell"),
        ("max_mito", "maximum percent mitochondrial reads"),
    ):
        g.add_argument(f"--{key.replace('_', '-')}", type=float, help=help_text)

    g = p.add_argument_group("run mode")
    g.add_argument("--explore", action="store_true",
                   help="force the QC-only stop even if thresholds are already "
                        "set, to re-inspect the distributions")
    g.add_argument("--auto-thresholds", action="store_true",
                   help="skip the QC review step and run end-to-end on "
                        "thresholds derived from the data; the report records "
                        "that they were not reviewed")

    g = p.add_argument_group("stage control")
    g.add_argument("--conditions", nargs="*", metavar="COL",
                   help="manifest columns to use as comparison axes "
                        "(default: autodetect columns that vary)")
    g.add_argument("--doublets", action="store_true",
                   help="run scrublet doublet annotation (OFF by default: it "
                        "assumes a heterogeneous population, so its scores are "
                        "not informative for a homogeneous line, and per-batch "
                        "it is memory-hungry)")
    g.add_argument("--no-doublets", action="store_true",
                   help="skip doublet annotation (now the default; kept so "
                        "existing commands keep working)")
    g.add_argument("--no-cell-cycle", action="store_true",
                   help="skip cell-cycle scoring")
    g.add_argument("--no-regress-out", action="store_true",
                   help="skip covariate regression (now the default; kept so "
                        "existing commands keep working)")
    g.add_argument("--regress-qc", action="store_true", dest="regress_qc",
                   help="regress sequencing depth and %%mito out of the "
                        "embedding. OFF by default: it only ever affects "
                        "clusters, UMAP and E-distance (never the DE results) "
                        "and it is the longest step in the stage. Cell-cycle "
                        "regression is deliberately not offered -- many "
                        "knockouts are proliferation phenotypes, so removing "
                        "the cell-cycle scores suppresses the very "
                        "perturbations being screened for")
    g.add_argument("--skip-input-check", action="store_true",
                   dest="skip_input_check",
                   help="do not verify that the expression matrix is "
                        "plausibly gene expression (housekeeping detection "
                        "and var-statistic cross-check). Only for a matrix "
                        "you have already validated by other means.")
    g.add_argument("--force-recompute", action="store_true",
                   help="ignore QC metrics, embedding, clustering and doublet "
                        "calls already present in the h5ad and recompute "
                        "everything from the counts")
    g.add_argument("--cluster-col", type=str, dest="cluster_col", metavar="COLUMN",
                   help="name of an existing obs column holding cluster labels. "
                        "Tried before the built-in candidates (leiden, "
                        "leiden_clusters, louvain, clusters, leiden_gpu). Reusing "
                        "an existing embedding requires PCA, UMAP AND a "
                        "recognised cluster column all to be present -- if the "
                        "cluster column's name isn't one of those and isn't "
                        "given here, the pipeline cannot tell clustering was "
                        "already done and recomputes the whole embedding")
    g.add_argument("--low-memory", action="store_true",
                   help="shorthand for --no-regress-out plus not copying the "
                        "input matrix and smaller DE gene blocks")
    g.add_argument("--batch-correct", choices=("auto", "harmony", "none"),
                   help="batch correction on the embedding (default: none; "
                        "'auto' is an alias for 'none' from v1.3.0). "
                        "'harmony' corrects the PCA, which propagates to "
                        "clusters, UMAP AND the E-distance effect sizes in the "
                        "perturbation section, so small transcriptional "
                        "differences tracking the batch key are attenuated. It "
                        "is refused when the resolved batch key is also a "
                        "declared condition column")
    g.add_argument("--hvg-batch-key", type=str, dest="hvg_batch_key",
                   metavar="COL",
                   help="make highly-variable-gene selection batch-aware on "
                        "COL. Off by default: with seurat_v3 this ranks gene "
                        "variance within each level, which de-prioritises "
                        "genes that differ between levels -- do not point it "
                        "at a condition column")
    g.add_argument("--resolution", type=float, dest="leiden_resolution",
                   help="Leiden clustering resolution (default 1.0)")
    g.add_argument("--n-top-genes", type=int,
                   help="number of highly variable genes (default 5000)")
    g.add_argument("--n-marker-genes", type=int,
                   help="top marker genes per cluster shown in the marker "
                        "dot plot (default 5)")

    g = p.add_argument_group("guide calling")
    g.add_argument("--guide-min-reads", type=int, dest="min_reads",
                   help="minimum guide UMIs for a cell to be eligible (default 10)")
    g.add_argument("--guide-purity-min", type=float, dest="purity_min",
                   help="minimum top1/(top1+top2) percent to assign (default 75)")
    g.add_argument("--ntc-label", type=str,
                   help="label used for non-targeting controls (default NTC)")
    g.add_argument("--dual-guide", dest="dual_guide", action="store_true",
                   default=None, help="force dual-guide (iBAR) handling")
    g.add_argument("--single-guide", dest="dual_guide", action="store_false",
                   help="force single-guide handling")

    g = p.add_argument_group("hashtags")
    g.add_argument("--hto-threshold-mode",
                   choices=("background_quantile", "fixed", "otsu"),
                   dest="threshold_mode",
                   help="how the positivity threshold is chosen")
    g.add_argument("--hto-quantile", type=float, dest="positive_quantile",
                   help="background quantile used as the threshold (default 0.99)")

    g = p.add_argument_group("report")
    g.add_argument("--title", type=str, help="report title")
    g.add_argument("--link-figures", action="store_true",
                   help="link figures instead of embedding them (smaller HTML, "
                        "but no longer a single portable file)")
    g.add_argument("--no-tables", action="store_true",
                   help="omit data tables from the HTML")
    g.add_argument("--max-de-targets", type=int, default=40,
                   help="cap on perturbations sent to differential expression "
                        "(default 40)")
    g.add_argument("-q", "--quiet", action="store_true", help="suppress progress")
    return p


def _collect_overrides(args: argparse.Namespace) -> dict:
    """Flat override dict; build_config routes each key to its subconfig."""
    o: dict = {}
    for key in THRESHOLD_KEYS:
        v = getattr(args, key, None)
        if v is not None:
            o[key] = v
    simple = (
        "leiden_resolution", "n_top_genes", "n_marker_genes", "batch_correct", "hvg_batch_key",
        "min_reads", "purity_min", "ntc_label", "threshold_mode",
        "positive_quantile", "dual_guide", "hto_normalisation",
    )
    for key in simple:
        v = getattr(args, key, None)
        if v is not None:
            o[key] = v
    if getattr(args, "doublets", False):
        o["detect_doublets"] = True
    if args.no_doublets:
        o["detect_doublets"] = False
    if args.no_cell_cycle:
        o["score_cell_cycle"] = False
    if args.force_recompute:
        o["force_recompute"] = True
    if getattr(args, "skip_input_check", False):
        o["check_input_matrix"] = False
    # regress_out is () by default from v1.3.0, so --no-regress-out and
    # --low-memory are now no-ops for this setting and are kept only so
    # existing commands keep working. --regress-qc is the way back in, and it
    # deliberately offers depth and %mito only -- never the cell-cycle scores.
    if args.no_regress_out or args.low_memory:
        o["regress_out"] = ()
    if getattr(args, "regress_qc", False):
        o["regress_out"] = ("total_counts", "pct_counts_mt")
    if args.low_memory:
        o["copy_input"] = False
        o["de_gene_block"] = 500
    if args.title:
        o["title"] = args.title
    if args.link_figures:
        o["embed_figures"] = False
    if args.no_tables:
        o["include_data_tables"] = False
    if args.conditions is not None:
        o["condition_columns"] = tuple(args.conditions)
    if args.quiet:
        o["verbose"] = False
    if args.counts_layer:
        o["counts_layer"] = args.counts_layer
    if getattr(args, "cluster_col", None):
        defaults = ModalityConfig().cluster_col_candidates
        if args.cluster_col not in defaults:
            o["cluster_col_candidates"] = (args.cluster_col,) + defaults
    if args.subsample_cells:
        o["subsample_cells"] = args.subsample_cells
    if args.output_path:
        o["output_path"] = args.output_path
    if args.h5ad_path:
        o["h5ad_path"] = args.h5ad_path
    if args.dragen_root:
        o["dragen_root"] = args.dragen_root
    return o


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.init_manifest:
        path = write_manifest_template(args.init_manifest)
        print(f"Wrote a blank manifest template to {path}")
        print(
            "Fill in 'sample', 'h5ad_path' and 'output_path' (the last two must be "
            "identical on every row), plus any condition columns you want compared."
        )
        return EXIT_OK

    if args.inspect:
        from .inspect_h5ad import available_memory_gb, format_report, inspect
        try:
            info = inspect(args.inspect)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
        print(format_report(info, available_memory_gb()))
        return EXIT_OK

    if args.rebuild_report:
        try:
            out = build_from_artifacts(args.rebuild_report, title=args.title)
        except (OSError, ValueError, KeyError) as exc:
            print(f"error: could not rebuild the report: {exc}", file=sys.stderr)
            return EXIT_PIPELINE
        print(f"Report rebuilt: {out}")
        return EXIT_OK

    if not args.manifest:
        parser.error(
            "--manifest is required (or use --init-manifest / --rebuild-report)"
        )

    # ------------------------------------------------------------- manifest
    # A column named here is only ever used to LOCATE something (the h5ad,
    # where results go). Once a CLI flag already says where that is, the
    # manifest column is dead weight -- requiring it anyway is exactly the
    # friction a wrapper that always passes --output-path (a Nextflow/ICA
    # pipeline, say, which manages its own output layout) would otherwise
    # hit on every single manifest for a column whose value it never reads.
    required_columns = [
        c for c in GLOBAL_COLUMNS
        if not (c == "h5ad_path" and args.h5ad_path)
        and not (c == "output_path" and args.output_path)
    ]
    try:
        manifest = read_manifest(args.manifest, required_columns=required_columns)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return EXIT_MANIFEST

    try:
        cfg = build_config(_collect_overrides(args), config_file=args.config)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    cfg.manifest_path = Path(args.manifest)

    # Manifest thresholds fill any gap the CLI left; CLI wins per key.
    try:
        manifest_th = manifest.read_thresholds()
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return EXIT_MANIFEST
    cli_thresholds = {k: getattr(args, k, None) for k in THRESHOLD_KEYS}
    for key in THRESHOLD_KEYS:
        if getattr(cfg.qc, key) is None and getattr(manifest_th, key) is not None:
            setattr(cfg.qc, key, getattr(manifest_th, key))
            cfg.qc.source[key] = "manifest"
        elif cli_thresholds[key] is not None:
            cfg.qc.source[key] = "cli"

    # ----------------------------------------------------------- run mode
    mode = decide_run_mode(
        manifest_thresholds=manifest_th.as_dict(),
        cli_thresholds=cli_thresholds,
        explore_flag=args.explore,
        auto_flag=args.auto_thresholds,
    )
    cfg.explore_only = mode.explore
    cfg.auto_thresholds = args.auto_thresholds

    if not args.quiet:
        print(f"perturbseq-report {__version__}")
        print(f"  manifest : {manifest.path}")
        print(f"  samples  : {manifest.n_samples} ({manifest.n_runs} runs)")
        # cfg.h5ad_path/output_path are already resolved from --h5ad/--output-path
        # by build_config() above. Prefer them over the manifest property directly:
        # when overridden, the manifest column may not even exist (see
        # required_columns above), and manifest.h5ad_path/output_path would raise.
        print(f"  h5ad     : {cfg.h5ad_path if cfg.h5ad_path is not None else manifest.h5ad_path}")
        print(f"  output   : {cfg.output_path if cfg.output_path is not None else manifest.output_path}")
        print(f"  mode     : {'QC review (explore)' if mode.explore else 'full pipeline'}"
              f" — {mode.reason}")
        for w in manifest.warnings:
            print(f"  warning  : {w}")

    # ------------------------------------------------------------------ run
    try:
        result = run(cfg, manifest)
    except PipelineError as exc:
        print(f"\npipeline error: {exc}", file=sys.stderr)
        return EXIT_PIPELINE
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_PIPELINE
    except Exception as exc:  # noqa: BLE001
        # Unexpected failures print a full traceback. The original swallowed
        # exceptions in several places, which turned a fixable bug into a
        # silently incomplete report.
        print(f"\nunexpected error: {exc}\n", file=sys.stderr)
        traceback.print_exc()
        return EXIT_UNEXPECTED

    if not args.quiet:
        print()
        print(f"  cells    : {result.n_cells_after:,} of {result.n_cells_before:,} "
              f"passed QC")
        print(f"  report   : {result.report_path}")
        if result.warnings:
            print(f"  {len(result.warnings)} warning(s) — see the report appendix")

        if mode.explore:
            # The whole point of explore-first is that the next step is a human
            # decision, so spell it out rather than leaving them to infer it.
            th = result.thresholds.as_dict()
            print()
            print("  This was the QC review step. Nothing has been filtered,")
            print("  clustered or quantified yet.")
            print()
            print("  Thresholds written into the manifest:")
            for key in THRESHOLD_KEYS:
                value = th.get(key)
                src = result.thresholds.source.get(key, "?")
                shown = "—" if value is None else f"{value:,.1f}"
                print(f"    {key:<12} {shown:>12}   ({src})")
            print()
            print(f"  Next: open {result.report_path.name}, check the counts, genes")
            print(f"  and %mito panels, edit those five columns in")
            print(f"  {manifest.path.name} if you disagree, then run the same")
            print("  command again for the full report.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
