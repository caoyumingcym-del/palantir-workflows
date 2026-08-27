#!/usr/bin/env python3
"""
Generate a tiny synthetic Perturb-seq dataset for the `-profile test` smoke
test -- an .h5ad, a matching sample manifest, and DRAGEN-style metrics files,
all built from perturbseq_report's own ground-truth generator
(perturbseq_report/synthetic.py), so the shapes and columns are guaranteed to
match what the pipeline actually expects.

    python3 bin/make_test_data.py --outdir test_data

This only proves the Nextflow plumbing works end to end on a dataset small
enough to run in a couple of minutes -- it does not check the pipeline's
scientific correctness (that's perturbseq_report_v1.3.5/tests/test_end_to_end.py,
which checks the generator's planted ground truth is recovered).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "perturbseq_report_v1.3.5"))

from perturbseq_report import synthetic  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("test_data"),
                    help="directory to write the synthetic dataset into")
    p.add_argument("--n-cells", type=int, default=1500)
    p.add_argument("--n-genes", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    h5ad_path = outdir / "synthetic.h5ad"
    output_path = outdir / "out"  # overridden by --output-path at run time anyway
    dragen_dir = outdir / "dragen"

    try:
        _, truth = synthetic.make_h5ad(
            h5ad_path, n_cells=args.n_cells, n_genes=args.n_genes, seed=args.seed,
        )
    except ImportError as exc:
        print(f"error: anndata is required to write a synthetic .h5ad ({exc})",
              file=sys.stderr)
        return 1

    manifest_path = outdir / "sample_manifest.csv"
    synthetic.write_manifest(
        manifest_path, h5ad_path, output_path, dragen_dir=dragen_dir,
    )

    import pandas as pd
    prefixes = pd.read_csv(manifest_path)["prefix"].tolist()
    synthetic.write_dragen_metrics(dragen_dir, prefixes)

    print(f"wrote synthetic dataset to {outdir}")
    print(f"  h5ad     : {h5ad_path}")
    print(f"  manifest : {manifest_path}")
    print(f"  targets planted: {list(truth.target_knockdown)}")
    print()
    print("Run it with:")
    print(f"  nextflow run . -profile test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
