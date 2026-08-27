"""
Synthetic Perturb-seq data with known ground truth.

Purpose: exercise the whole pipeline end to end and check that it *recovers
numbers we planted*.  A pipeline that runs without crashing is not the same as
a pipeline that is correct, and the original had no way to tell the difference.

The generator plants, and ``tests/test_end_to_end.py`` checks:

* a target knockdown level per perturbation (strong, weak and zero)
* downstream differentially expressed genes per perturbation
* a guide capture rate and an ambient guide contamination level
* a hashtag doublet rate, a negative rate, and one deliberately broken hashtag
* a fraction of low-quality cells that QC should remove
* distinct cell populations tied to hashtag identity, so clustering has
  something real to find

``make_h5ad`` writes a real .h5ad when anndata is installed. ``make_bundle``
returns the same data as plain arrays, which is what lets the numeric stages be
tested in an environment without the single-cell stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GroundTruth:
    """What was planted, for the tests to check against."""

    target_knockdown: dict[str, float]        # target -> intended fraction remaining
    target_de_genes: dict[str, list[str]]     # target -> genes moved downstream
    guide_capture_rate: float
    ambient_guide_rate: float
    hto_doublet_rate: float
    hto_negative_rate: float
    broken_hashtags: list[str]
    low_quality_fraction: float
    n_cells: int
    n_genes: int
    populations: list[str]
    ntc_label: str = "NTC"

    def strong_targets(self) -> list[str]:
        return [t for t, r in self.target_knockdown.items() if r <= 0.4]

    def null_targets(self) -> list[str]:
        return [t for t, r in self.target_knockdown.items() if r >= 0.9]


@dataclass
class Bundle:
    """Synthetic dataset as plain arrays."""

    counts: np.ndarray          # cells x genes, raw counts
    gene_names: list[str]
    guide_counts: np.ndarray    # cells x guides
    guide_names: list[str]
    hto_counts: np.ndarray      # cells x hashtags
    hto_names: list[str]
    obs: pd.DataFrame
    truth: GroundTruth
    layout: str = "feature_types"


def make_bundle(
    n_cells: int = 3000,
    n_genes: int = 600,
    n_targets: int = 8,
    guides_per_target: int = 2,
    n_htos: int = 4,
    n_samples: int = 2,
    n_lanes: int = 2,
    seed: int = 0,
    guide_capture_rate: float = 0.82,
    ambient_guide_rate: float = 0.06,
    hto_doublet_rate: float = 0.08,
    hto_negative_rate: float = 0.05,
    low_quality_fraction: float = 0.12,
    broken_hashtag: bool = True,
    dual_guide: bool = False,
    layout: str = "feature_types",
) -> Bundle:
    """Generate a synthetic Perturb-seq experiment with planted effects."""
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- genes
    gene_names = [f"GENE{i:04d}" for i in range(n_genes)]
    targets = [f"TGT{i}" for i in range(n_targets)]
    for i, t in enumerate(targets):
        gene_names[i] = t                    # targets are real genes
    # Mitochondrial and ribosomal genes, so the QC and HVG-exclusion paths run.
    for j, nm in enumerate(["MT-CO1", "MT-ND1", "MT-CYB", "RPS6", "RPL13", "RPS3"]):
        gene_names[n_genes - 1 - j] = nm
    mito_idx = [i for i, g in enumerate(gene_names) if g.startswith("MT-")]

    # Knockdown strength: some strong, some partial, some null.
    remaining = {}
    for i, t in enumerate(targets):
        if i < n_targets // 2:
            remaining[t] = float(rng.uniform(0.08, 0.30))    # strong
        elif i < n_targets - 2:
            remaining[t] = float(rng.uniform(0.45, 0.65))    # partial
        else:
            remaining[t] = 1.0                               # null
    remaining["NTC"] = 1.0

    # Downstream DE genes per target (strong targets only).
    de_genes: dict[str, list[str]] = {}
    pool = [g for g in gene_names[n_targets:] if not g.startswith(("MT-", "RPS", "RPL"))]
    for i, t in enumerate(targets):
        if remaining[t] <= 0.4:
            picked = list(rng.choice(pool, size=8, replace=False))
            de_genes[t] = [str(x) for x in picked]
        else:
            de_genes[t] = []

    # ------------------------------------------------------- cell populations
    populations = [f"POP{i+1}" for i in range(n_htos)]
    pop_of_cell = rng.integers(0, n_htos, size=n_cells)

    base = rng.lognormal(mean=0.6, sigma=1.1, size=n_genes)
    base[mito_idx] *= 6.0                    # mito genes are highly expressed
    # Each population up-regulates its own block of genes, so clustering has a
    # real signal to recover and the cluster/hashtag cross-check is meaningful.
    pop_profiles = []
    block = max(8, n_genes // (n_htos * 6))
    for p in range(n_htos):
        mult = np.ones(n_genes)
        start = n_targets + 20 + p * block
        mult[start : start + block] = rng.uniform(4.0, 9.0)
        pop_profiles.append(mult)

    # ---------------------------------------------------------------- guides
    if dual_guide:
        guide_names = [
            f"{t}.iBAR.{k+1}" for t in targets for k in range(2)
        ] + ["NTC_ctrl.iBAR.1", "NTC_ctrl.iBAR.2"]
    else:
        guide_names = [
            f"{t}_ENSG{i:011d}_sg{k+1}"
            for i, t in enumerate(targets)
            for k in range(guides_per_target)
        ] + [f"NonTargetingControl_{k+1}" for k in range(4)]
    n_guides = len(guide_names)
    guide_target = []
    for g in guide_names:
        guide_target.append("NTC" if "NonTargeting" in g or "NTC" in g
                            else g.split("_")[0].split(".")[0])

    # Assign a perturbation to each cell, weighting NTC up so the control group
    # is comfortably the largest -- as in a real screen.
    weights = np.array([3.0 if gt == "NTC" else 1.0 for gt in guide_target])
    weights /= weights.sum()
    cell_guide = rng.choice(n_guides, size=n_cells, p=weights)
    cell_target = np.array([guide_target[g] for g in cell_guide])

    captured = rng.random(n_cells) < guide_capture_rate

    guide_counts = np.zeros((n_cells, n_guides))
    depth = rng.negative_binomial(4, 0.06, size=n_cells) + 3
    for i in range(n_cells):
        if captured[i]:
            guide_counts[i, cell_guide[i]] = depth[i]
        n_amb = rng.poisson(ambient_guide_rate * max(depth[i], 1))
        if n_amb:
            for j in rng.integers(0, n_guides, size=int(n_amb)):
                guide_counts[i, j] += rng.poisson(1.5) + 1

    # ------------------------------------------------------------- expression
    counts = np.zeros((n_cells, n_genes))
    lib = rng.lognormal(np.log(6000), 0.35, size=n_cells)
    gene_pos = {g: i for i, g in enumerate(gene_names)}

    for i in range(n_cells):
        mult = pop_profiles[pop_of_cell[i]].copy()
        t = cell_target[i]
        if captured[i] and t != "NTC":
            ti = gene_pos.get(t)
            if ti is not None:
                mult[ti] *= remaining[t]
            for g in de_genes.get(t, []):
                mult[gene_pos[g]] *= 3.0
        lam = base * mult
        lam = lam / lam.sum() * lib[i]
        counts[i] = rng.poisson(lam)

    # Low-quality cells: shallow, low-complexity, high mito -- what QC removes.
    n_low = int(low_quality_fraction * n_cells)
    low_idx = rng.choice(n_cells, size=n_low, replace=False)
    for i in low_idx:
        keep = rng.random(n_genes) < 0.12
        counts[i] = counts[i] * keep
        counts[i] *= rng.uniform(0.05, 0.2)
        counts[i, mito_idx] += rng.poisson(60, size=len(mito_idx))
    counts = np.rint(counts)

    # ------------------------------------------------------------- hashtags
    hto_names = [f"HTO{i+1}" for i in range(n_htos)]
    broken: list[str] = []
    if broken_hashtag and n_htos >= 2:
        hto_names[-1] = f"HTO{n_htos}_failed"
        broken.append(hto_names[-1])

    hto_counts = rng.poisson(5, size=(n_cells, n_htos)).astype(float)
    working = [j for j, nm in enumerate(hto_names) if nm not in broken]
    for i in range(n_cells):
        j = pop_of_cell[i]
        if j in working:
            hto_counts[i, j] += rng.negative_binomial(5, 0.015)
        else:
            # Cells from the broken-hashtag population get no real signal.
            hto_counts[i, j] += rng.poisson(4)

    n_dbl = int(hto_doublet_rate * n_cells)
    for i in rng.choice(n_cells, size=n_dbl, replace=False):
        j = int(rng.choice(working)) if working else 0
        hto_counts[i, j] += rng.negative_binomial(5, 0.015)
    n_neg = int(hto_negative_rate * n_cells)
    for i in rng.choice(n_cells, size=n_neg, replace=False):
        hto_counts[i, :] = rng.poisson(0.6, size=n_htos)

    # -------------------------------------------------------------------- obs
    samples = [f"sample_{i+1}" for i in range(n_samples)]
    lanes = [f"LANE{i+1}" for i in range(n_lanes)]
    obs = pd.DataFrame(
        {
            "sample": rng.choice(samples, n_cells),
            "prefix": rng.choice(lanes, n_cells),
            "true_population": [populations[p] for p in pop_of_cell],
            "true_target": cell_target,
            "true_guide_captured": captured,
        },
        index=[f"BC{i:07d}" for i in range(n_cells)],
    )

    truth = GroundTruth(
        target_knockdown=remaining,
        target_de_genes=de_genes,
        guide_capture_rate=guide_capture_rate,
        ambient_guide_rate=ambient_guide_rate,
        hto_doublet_rate=hto_doublet_rate,
        hto_negative_rate=hto_negative_rate,
        broken_hashtags=broken,
        low_quality_fraction=low_quality_fraction,
        n_cells=n_cells,
        n_genes=n_genes,
        populations=populations,
    )
    return Bundle(
        counts=counts, gene_names=gene_names, guide_counts=guide_counts,
        guide_names=guide_names, hto_counts=hto_counts, hto_names=hto_names,
        obs=obs, truth=truth, layout=layout,
    )


def bundle_to_anndata(b: Bundle):
    """Build an AnnData in the requested layout. Requires anndata."""
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "anndata is required to build a synthetic .h5ad; install it with "
            "`pip install anndata`, or use make_bundle() for array-level testing."
        ) from exc
    from scipy import sparse  # noqa: F401  (optional, only for sparse output)

    if b.layout == "feature_types":
        X = np.hstack([b.counts, b.guide_counts, b.hto_counts])
        var = pd.DataFrame(
            {
                "feature_types": (
                    ["Gene Expression"] * len(b.gene_names)
                    + ["CRISPR Guide Capture"] * len(b.guide_names)
                    + ["Antibody Capture"] * len(b.hto_names)
                )
            },
            index=b.gene_names + b.guide_names + b.hto_names,
        )
        return ad.AnnData(X=X, obs=b.obs.copy(), var=var)

    adata = ad.AnnData(
        X=b.counts,
        obs=b.obs.copy(),
        var=pd.DataFrame(index=b.gene_names),
    )
    adata.obsm["gRNA_counts"] = b.guide_counts
    adata.obsm["HTO_counts"] = b.hto_counts
    adata.uns["gRNA_features"] = np.array(b.guide_names, dtype=object)
    adata.uns["HTO_features"] = np.array(b.hto_names, dtype=object)
    return adata


def make_h5ad(path: Path, **kwargs: Any) -> tuple[Path, GroundTruth]:
    """Write a synthetic .h5ad and return its path plus the ground truth."""
    b = make_bundle(**kwargs)
    adata = bundle_to_anndata(b)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(p)
    return p, b.truth


def write_manifest(
    path: Path, h5ad_path: Path, output_path: Path,
    n_samples: int = 2, n_lanes: int = 2, hto: str = "yes",
    dragen_dir: Path | None = None,
) -> Path:
    """Write a manifest matching a synthetic dataset."""
    rows = []
    for s in range(n_samples):
        for lane in range(n_lanes):
            rows.append(
                {
                    "sample": f"sample_{s+1}",
                    "h5ad_path": str(h5ad_path),
                    "output_path": str(output_path),
                    "gRNA_method": "PCR" if s == 0 else "direct",
                    "resuspension_buffer": "CSB",
                    "fixation": "no",
                    "HTO": hto,
                    "cell_type": "HeLa/MCF10A/RPE1/A549",
                    "cell_input": 40000,
                    "prefix": f"LANE{lane+1}" if s == 0 else f"LANE{lane+1}",
                    "dragen_path": str(dragen_dir) if dragen_dir else "",
                }
            )
    # prefixes must be unique per row for metrics lookup
    for i, r in enumerate(rows):
        r["prefix"] = f"LANE{i+1}"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def write_dragen_metrics(directory: Path, prefixes: list[str]) -> Path:
    """Write plausible DRAGEN scRNA_metrics.csv files for the given prefixes."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for pref in prefixes:
        total = int(rng.integers(8e8, 1.4e9))
        cells = int(rng.integers(9000, 15000))
        lines = [
            f"RNA,{pref},Total input reads,{total},",
            f"RNA,{pref},Total barcoded reads,{int(total * 0.96)},96.00",
            f"RNA,{pref},Reads with valid molecular identifier sequences,"
            f"{int(total * 0.94)},94.00",
            f"RNA,{pref},Mapped reads,{int(total * 0.91)},91.00",
            f"RNA,{pref},Fraction of reads in passing cells,,"
            f"{rng.uniform(60, 85):.2f}",
            f"RNA,{pref},Estimated number of cells,{cells},",
            f"RNA,{pref},Median genes per cell,{int(rng.integers(1500, 2600))},",
            f"RNA,{pref},Sequencing saturation,,{rng.uniform(40, 70):.2f}",
            f"RNA,{pref},CRISPR Number of reads,{int(total * 0.1)},",
            f"RNA,{pref},CRISPR Estimated number of cells,{int(cells * 0.97)},",
            f"RNA,{pref},CRISPR fraction valid reads in cells,,"
            f"{rng.uniform(55, 80):.2f}",
        ]
        (d / f"{pref}.scRNA_metrics.csv").write_text("\n".join(lines) + "\n")
    return d
