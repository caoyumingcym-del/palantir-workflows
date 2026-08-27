"""
Transcriptome processing: normalisation, feature selection, embedding,
clustering, cell-cycle scoring, doublet annotation and marker genes.

Pipeline order (this is the part that must not be rearranged):

    raw counts (already QC-filtered)
      -> normalise to fixed depth
      -> log1p
      -> HVG selection, EXCLUDING mitochondrial and ribosomal genes
      -> regress out depth / %mito / cell-cycle scores
      -> scale
      -> PCA
      -> optional batch correction (harmony)
      -> neighbours -> UMAP -> Leiden
      -> merge tiny clusters
      -> marker genes, again excluding MT/ribo

Three things here come from the collaborator's pipeline and were absent from
the original:

1. **MT/ribosomal exclusion from HVG selection and from the marker test.**
   Without it, clusters are frequently defined by stress and translation
   programmes and their "markers" are a list of ribosomal proteins, which tells
   you nothing about cell identity.
2. **Regressing out depth, %mito and cell-cycle scores.**
3. **Merging tiny Leiden fragments into their nearest cluster in PCA space**
   rather than leaving 12-cell specks that clutter every downstream panel.

One claim from the original is corrected: it described its embedding as
"batch-aware", but only ever passed ``batch_key`` to HVG selection -- the PCA,
neighbours, UMAP and Leiden steps were entirely uncorrected. Here batch
correction is either actually applied (harmony) or explicitly reported as not
applied.

Also: doublets are **annotated and retained**, never removed, and that is
stated in the report. Silently dropping them would change every downstream cell
count relative to the QC section.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import embedcache as EMBCACHE
from . import plotting as P
from . import text as T
from .artifacts import Registry
from .config import EmbeddingConfig, ModalityConfig, PipelineConfig
from .provenance import LOG1P, NORMALISED, RAW_COUNTS
from .stats import (
    benjamini_hochberg, col_variances, mannwhitney_u,
    mannwhitney_u_columns, mannwhitney_u_sparse_columns, nbytes_dense,
    normalize_rows, row_sums, sparse_log1p, take_columns, to_csc,
)

try:  # pragma: no cover
    import scanpy as sc
    _HAVE_SCANPY = True
except Exception:  # pragma: no cover
    sc = None
    _HAVE_SCANPY = False


# Standard human cell-cycle marker sets (Tirosh et al. 2016), used when the
# caller does not supply their own. Matched case-insensitively so they work
# against mouse-cased symbols too.
S_GENES = (
    "MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 "
    "CENPU HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 "
    "ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN "
    "POLA1 CHAF1B BRIP1 E2F8"
).split()
G2M_GENES = (
    "HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO "
    "CENPF TACC3 PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B "
    "GTSE1 KIF20B HJURP CDCA3 JPT1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 "
    "CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 "
    "GAS2L3 CBX5 CENPA"
).split()


@dataclass
class EmbeddingResult:
    """Everything the downstream stages and the report need."""

    X_log: Any                        # normalised, log1p, all genes (SPARSE)
    var_names: list[str]
    obs: pd.DataFrame                 # clusters, phase, doublets, scores
    pca: np.ndarray | None
    umap: np.ndarray | None
    hvg: list[str] = field(default_factory=list)
    markers: pd.DataFrame | None = None
    cluster_summary: pd.DataFrame | None = None
    backend: str = "scanpy"
    batch_corrected: str = "none"
    notes: list[str] = field(default_factory=list)


# ===========================================================================
# Helpers
# ===========================================================================
def _prefix_mask(names: Sequence[str], prefixes: Sequence[str]) -> np.ndarray:
    return np.array([any(str(n).startswith(p) for p in prefixes) for n in names])


def excluded_feature_mask(
    var_names: Sequence[str], mcfg: ModalityConfig, ecfg: EmbeddingConfig
) -> np.ndarray:
    """Genes to keep OUT of HVG selection and the marker test."""
    mask = np.zeros(len(var_names), dtype=bool)
    if ecfg.exclude_mito_from_hvg:
        mask |= _prefix_mask(var_names, mcfg.mito_prefixes)
    if ecfg.exclude_ribo_from_hvg:
        mask |= _prefix_mask(var_names, mcfg.ribo_prefixes)
    return mask


def pick_batch_key(
    obs: pd.DataFrame,
    candidates: Sequence[str],
    dominance_max: float = 0.90,
) -> tuple[str | None, list[str]]:
    """First candidate column with 2+ levels and no single level dominating.

    Returns ``(key, reasons)``, where ``reasons`` explains what each rejected
    candidate failed on so the report can state the outcome rather than imply
    it.

    Up to v1.2.5 this function's docstring made exactly the claim above and the
    code implemented only half of it: ``vc`` was computed and then only
    ``len(vc)`` was tested, so a 99%/1% split counted as batch structure. The
    dominance test is now real -- a column where one level holds
    ``dominance_max`` or more of the cells is one big group with a rounding
    error attached, not a batch.
    """
    reasons: list[str] = []
    for c in candidates:
        if c not in obs.columns:
            reasons.append(f"{c!r}: not present in obs")
            continue
        vc = obs[c].astype(str).value_counts()
        if len(vc) < 2:
            reasons.append(f"{c!r}: only one level ({vc.index[0]!r})" if len(vc)
                           else f"{c!r}: no values")
            continue
        top_frac = float(vc.iloc[0]) / float(vc.sum())
        if top_frac >= dominance_max:
            reasons.append(
                f"{c!r}: level {str(vc.index[0])!r} holds "
                f"{100.0 * top_frac:.1f}% of cells (>= "
                f"{100.0 * dominance_max:.0f}% dominance limit)"
            )
            continue
        reasons.append(
            f"{c!r}: accepted -- {len(vc)} levels, largest "
            f"{100.0 * top_frac:.1f}% of cells"
        )
        return c, reasons
    return None, reasons


DOUBLET_FALLBACK_NOTE = (
    "Doublet annotation used the built-in synthetic-doublet fallback: "
    "{n:,} of {total:,} cells ({pct:.1f}%) flagged. Flagged cells are RETAINED "
    "in the embedding and in every downstream analysis. Read the rate with "
    "care -- the fallback's threshold is max(90th percentile of scores, "
    "1.5x expected), so it flags a near-fixed 5-10% of cells whatever the "
    "data look like, and the number is largely a property of that rule rather "
    "than a measurement. Scrublet is preferable where it is available."
)


def small_cluster_notes(
    labels: np.ndarray, ecfg: EmbeddingConfig
) -> list[str]:
    """Report cluster fragmentation instead of silently repairing it.

    From v1.3.0 ``min_cluster_frac`` defaults to 0, so nothing is merged. That
    makes fragmentation visible, which means it has to be *stated* -- otherwise
    the change just moves a problem from "silently absorbed" to "silently
    present".
    """
    n = int(np.asarray(labels).size)
    if n == 0:
        return []
    vc = pd.Series(labels).value_counts()
    frac = float(ecfg.small_cluster_report_frac)
    cutoff = max(int(np.floor(frac * n)), 1)
    small = vc[vc < cutoff]
    if small.empty:
        return [
            f"All {len(vc)} clusters hold at least {frac * 100:.1f}% of cells "
            f"({cutoff:,}), so there is no cluster fragmentation to report."
        ]
    return [
        f"{len(small)} of {len(vc)} clusters hold fewer than "
        f"{frac * 100:.1f}% of cells ({cutoff:,} cells), covering "
        f"{int(small.sum()):,} cells in total: "
        + ", ".join(f"cluster {k} ({int(v):,})" for k, v in
                    list(small.items())[:8])
        + ("; ..." if len(small) > 8 else "")
        + ". These are reported, NOT merged. Up to v1.2.5 any cluster below "
          "0.5% of cells was absorbed into its nearest neighbour by PCA "
          "centroid distance -- which in a 187k-cell experiment meant "
          "discarding populations of ~935 cells on the strength of a Euclidean "
          "distance between centroids, with no method behind the rule. In a "
          "screen a rare population may be the interesting one, so the "
          "fragmentation is now shown and left for you to judge."
    ]


def merge_tiny_clusters(
    labels: np.ndarray, pca: np.ndarray, min_frac: float
) -> tuple[np.ndarray, list[str]]:
    """Merge clusters below ``min_frac`` into their nearest neighbour by centroid.

    Merging rather than dropping means the cell count is stable across every
    section of the report. Iterative, because merging one speck can leave
    another still below threshold.
    """
    labels = np.asarray(labels).astype(object).copy()
    notes: list[str] = []
    n = labels.size
    if n == 0 or pca is None:
        return labels, notes
    min_cells = max(int(np.floor(min_frac * n)), 1)

    for _ in range(50):
        vc = pd.Series(labels).value_counts()
        if len(vc) <= 1:
            break
        small = [k for k, v in vc.items() if v < min_cells]
        if not small:
            break
        big = [k for k, v in vc.items() if v >= min_cells]
        if not big:
            break
        centroids = {
            k: pca[labels == k].mean(axis=0) for k in vc.index
        }
        victim = min(small, key=lambda k: vc[k])
        target = min(
            big,
            key=lambda k: float(
                np.sum((centroids[victim] - centroids[k]) ** 2)
            ),
        )
        notes.append(
            f"merged cluster {victim} ({int(vc[victim])} cells) into "
            f"cluster {target}"
        )
        labels[labels == victim] = target

    if notes:
        notes = [
            f"{len(notes)} Leiden cluster(s) smaller than {min_frac * 100:.1f}% of "
            f"cells ({min_cells} cells) were merged into their nearest cluster in "
            f"PCA space rather than dropped, so no cells are lost: "
            + "; ".join(notes[:6])
            + ("; ..." if len(notes) > 6 else "")
        ]
    # Relabel to contiguous integers ordered by descending size.
    vc = pd.Series(labels).value_counts()
    remap = {old: str(i) for i, old in enumerate(vc.index)}
    return np.array([remap[x] for x in labels], dtype=object), notes


# ===========================================================================
# Fallback (no scanpy) implementations
# ===========================================================================
def _fallback_pca(X: np.ndarray, n_pcs: int, random_state: int = 0) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    k = int(min(n_pcs, min(Xc.shape) - 1))
    if k < 1:
        return np.zeros((X.shape[0], 1))
    try:
        rng = np.random.default_rng(random_state)
        # Randomised SVD: full SVD on a large matrix is wasteful and slow.
        Omega = rng.normal(size=(Xc.shape[1], k + 10))
        Y = Xc @ Omega
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ Xc
        Ub, S, _ = np.linalg.svd(B, full_matrices=False)
        U = Q @ Ub
        return (U[:, :k] * S[:k])
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.zeros((X.shape[0], k))


def _fallback_cluster(
    pca: np.ndarray, resolution: float, random_state: int = 0
) -> np.ndarray:
    """k-means style clustering, used only when scanpy/leidenalg is absent.

    This is explicitly NOT Leiden and the report says so. It exists so the
    pipeline can be exercised end-to-end (and produce a report) in an
    environment without the single-cell stack; it is not a substitute for
    graph-based clustering on real data.
    """
    n = pca.shape[0]
    k = int(np.clip(round(4 * resolution), 2, max(2, min(20, n // 30 or 2))))
    rng = np.random.default_rng(random_state)
    centres = pca[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    p_sq = np.einsum("ij,ij->i", pca, pca)[:, None]
    for _ in range(60):
        d = (
            p_sq
            - 2.0 * (pca @ centres.T)
            + np.einsum("ij,ij->i", centres, centres)[None, :]
        )
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                centres[j] = pca[labels == j].mean(axis=0)
    return labels.astype(object).astype(str)


def _score_genes(
    X_log: np.ndarray, var_names: Sequence[str], gene_set: Sequence[str]
) -> np.ndarray:
    """Mean expression of a gene set, centred on a random background.

    A simplified version of scanpy's ``score_genes``: the set's mean minus the
    mean of all genes, which is enough to separate cycling from non-cycling
    cells for the purpose of a QC panel.
    """
    upper = {str(v).upper(): i for i, v in enumerate(var_names)}
    idx = [upper[g] for g in (x.upper() for x in gene_set) if g in upper]
    if not idx:
        return np.full(X_log.shape[0], np.nan)
    # Only the marker set is densified; the background mean is a sparse row
    # aggregate over the whole matrix.
    set_mean = take_columns(X_log, idx).mean(axis=1)
    n_genes = int(X_log.shape[1])
    background = row_sums(X_log) / max(n_genes, 1)
    return set_mean - background


def _detect_doublets_fallback(
    pca: np.ndarray, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate synthetic doublets and score cells by their similarity to them.

    The Scrublet idea in miniature: build artificial doublets by averaging
    random cell pairs, then score each real cell by how many of its nearest
    neighbours are synthetic. Used when scrublet is not installed.

    Everything happens in PCA space. That is not an approximation: PCA is an
    affine projection, so the projection of an averaged expression profile
    equals the average of the two projections. The previous version instead
    recovered a projection basis with ``np.linalg.pinv`` on the full
    cells-by-genes matrix, which on a real experiment is both enormous and
    unnecessary.
    """
    if pca is None or np.asarray(pca).size == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    pca = np.asarray(pca, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    n = pca.shape[0]
    if n < 4:
        return np.zeros(n), np.zeros(n, dtype=bool)

    n_sim = max(int(0.5 * n), 50)
    a = rng.integers(0, n, n_sim)
    b = rng.integers(0, n, n_sim)
    sim_pca = (pca[a] + pca[b]) / 2.0

    combined = np.vstack([pca, sim_pca])
    is_sim = np.zeros(combined.shape[0], dtype=bool)
    is_sim[pca.shape[0]:] = True

    k = int(min(30, combined.shape[0] - 1))
    scores = np.zeros(n)
    # Squared distances via ||a||^2 - 2a.b + ||b||^2 and one matrix product,
    # rather than a 3-D broadcast. The broadcast materialises an
    # (chunk x m x n_pcs) array, which for a typical run is gigabytes and
    # dominated the whole pipeline's runtime.
    comb_sq = np.einsum("ij,ij->i", combined, combined)
    chunk = max(256, int(4e6 // max(combined.shape[0], 1)))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = pca[start:stop]
        d = (
            np.einsum("ij,ij->i", block, block)[:, None]
            - 2.0 * (block @ combined.T)
            + comb_sq[None, :]
        )
        nn = np.argpartition(d, k, axis=1)[:, : k + 1]
        scores[start:stop] = is_sim[nn].mean(axis=1)
    expected = n_sim / (n + n_sim)
    thr = float(np.quantile(scores, 0.90))
    thr = max(thr, expected * 1.5)
    return scores, scores > thr


# ===========================================================================
# Main
# ===========================================================================
def _reuse_embedding(
    gex: Any, cfg: PipelineConfig, prior: Any, notes: list[str], step=None
) -> EmbeddingResult:
    """Wrap an embedding that is already in the object, computing nothing.

    On an already-analysed h5ad this replaces normalisation, HVG selection,
    scaling, PCA, batch correction, UMAP and Leiden with a handful of array
    lookups. For 300,000 cells that is the difference between tens of GB and
    essentially none, and it also avoids presenting a second, different
    clustering of cells that were already clustered upstream.
    """
    ecfg = cfg.embedding
    if step:
        step(f"reusing existing embedding (obsm['{prior.pca_key}'], "
             f"obsm['{prior.umap_key}'], obs['{prior.cluster_column}'])")

    obs = gex.obs.copy()
    pca = np.asarray(gex.obsm[prior.pca_key], dtype=np.float64)
    umap = np.asarray(gex.obsm[prior.umap_key], dtype=np.float64)

    labels = obs[prior.cluster_column].astype(str).to_numpy()
    obs["cluster"] = pd.Categorical(labels)

    if "total_counts" not in obs.columns:
        obs["total_counts"] = row_sums(gex.X)

    if prior.doublet_call_column and "predicted_doublet" not in obs.columns:
        obs["predicted_doublet"] = (
            obs[prior.doublet_call_column].astype(str)
            .str.lower().isin(("true", "1", "doublet", "yes"))
        )
    if prior.doublet_score_column and "doublet_score" not in obs.columns:
        obs["doublet_score"] = pd.to_numeric(
            obs[prior.doublet_score_column], errors="coerce")

    hvg: list[str] = []
    if prior.hvg_column and prior.hvg_column in gex.var.columns:
        flags = gex.var[prior.hvg_column].astype(bool).to_numpy()
        hvg = [str(v) for v, f in zip(gex.var.index, flags) if f]

    # X may already be log-normalised; if so leave it alone. But "normalised,
    # not log-transformed" is a THIRD state, distinct from both -- and it used
    # to be folded into the same branch as "already log1p", which meant X was
    # passed downstream unlogged and labelled X_log. Every consumer of X_log
    # (percent_knockdown, differential_expression, perturbation_score) assumes
    # log_input=True and calls expm1() on it; expm1() of an already-linear
    # normalised value for a highly-expressed gene (tens) explodes into the
    # 1e10-1e23 range, corrupting every knockdown/DE number built on it while
    # looking superficially plausible for lowly-expressed genes. Each of the
    # three detected states now gets the transform it actually needs.
    if prior.x_state == RAW_COUNTS:
        if step:
            step("normalising the reused object (its X is raw counts)")
        X_log = sparse_log1p(normalize_rows(gex.X, ecfg.target_sum))
    elif prior.x_state == NORMALISED:
        if step:
            step("log1p-transforming the reused object (its X is normalised "
                 "but not log-transformed)")
        X_log = sparse_log1p(gex.X)
        notes.append(
            f"X held {prior.x_state}: it was NOT re-normalised (that was "
            f"already done upstream), but log1p WAS applied, because "
            f"downstream expression analysis (knockdown, differential "
            f"expression) un-logs its input and would otherwise silently "
            f"treat already-linear values as if they were log-scale, "
            f"producing nonsensical mean-expression values for "
            f"highly-expressed genes."
        )
    else:
        if step:
            step(f"leaving the reused object's X as-is (already {prior.x_state})")
        X_log = gex.X
        notes.append(
            f"X was left as-is for downstream expression analysis because it "
            f"already holds {prior.x_state}. Nothing was normalised or "
            f"log-transformed a second time."
        )

    n_clusters = int(pd.Series(labels).nunique())
    notes.append(
        f"Embedding, clustering ({n_clusters} clusters) and marker-gene inputs "
        f"were taken from the input object rather than recomputed."
    )
    return EmbeddingResult(
        X_log=X_log, var_names=[str(v) for v in gex.var.index], obs=obs,
        pca=pca, umap=umap, hvg=hvg,
        backend=f"reused from input (obsm['{prior.pca_key}'])",
        batch_corrected=(
            "as provided in the input"
            if "harmony" in str(prior.pca_key).lower() else "unknown (from input)"
        ),
    )


def resolve_batch_correction(
    obs: pd.DataFrame,
    ecfg: EmbeddingConfig,
    condition_columns: Sequence[str] | None,
    notes: list[str],
) -> tuple[str | None, bool]:
    """Resolve the batch key, and decide whether correction may run.

    Returns ``(batch_key, may_correct)``. The key is resolved and REPORTED
    whether or not anything consumes it, because "no batch key was found" and
    "a batch key was found and deliberately not used" are different facts about
    a run and the report should not conflate them.

    Correction is refused -- the correction only, not the run -- when the
    resolved key overlaps a declared condition column. Integrating away your
    own comparison should not be reachable by accident.
    """
    batch_key, reasons = pick_batch_key(
        obs, ecfg.batch_key_candidates, ecfg.batch_dominance_max
    )
    want = str(ecfg.batch_correct or "none").lower()
    # "auto" used to mean "harmony whenever possible" and was the default.
    # It is retained as an alias for "none" so an old config file cannot
    # silently reacquire correction.
    if want == "auto":
        want = "none"
        notes.append(
            "batch_correct='auto' is read as 'none' from v1.3.0. Batch "
            "correction is now opt-in via --batch-correct harmony."
        )

    if batch_key is None:
        notes.append(
            "No batch-like column was found among "
            f"{list(ecfg.batch_key_candidates)}, so there is nothing to "
            "correct for and nothing to check. Considered: "
            + "; ".join(reasons) + "."
        )
        return None, False

    notes.append(
        f"Batch key resolved to {batch_key!r}. Considered: "
        + "; ".join(reasons) + "."
    )

    conds = [str(c) for c in (condition_columns or [])]
    overlap = [c for c in conds if c == batch_key]
    if want == "harmony" and overlap:
        notes.append(
            f"REFUSED: batch correction was requested on {batch_key!r}, but "
            f"{batch_key!r} is also a declared condition column. Correcting on "
            f"it would integrate away the comparison this experiment is "
            f"running, so the correction was skipped and the rest of the run "
            f"continued. The embedding below is UNCORRECTED. If the overlap is "
            f"intentional, remove {batch_key!r} from the manifest's condition "
            f"columns or choose a different batch key."
        )
        return batch_key, False

    if want != "harmony":
        notes.append(
            f"No batch correction was applied, which is the default from "
            f"v1.3.0. {batch_key!r} has multiple levels, so use the "
            f"{batch_key}-coloured UMAP panel to judge whether it matters: "
            f"clusters that map one-to-one onto a level are a batch effect. "
            f"This report measures batch structure rather than removing it -- "
            f"a QC report that silently corrects an effect cannot also tell "
            f"you whether the effect was there. Integration, if wanted, "
            f"belongs downstream of this report."
        )
        return batch_key, False

    return batch_key, True


def process_transcriptome(
    gex: Any,
    cfg: PipelineConfig,
    reg: Registry,
    step=None,
    prior: Any = None,
    condition_columns: Sequence[str] | None = None,
) -> EmbeddingResult:
    """Normalise, embed, cluster and annotate the gene-expression matrix."""
    ecfg, mcfg = cfg.embedding, cfg.modality
    notes: list[str] = []
    var_names = [str(v) for v in gex.var.index]
    obs = gex.obs.copy()

    excl = excluded_feature_mask(var_names, mcfg, ecfg)
    if excl.any():
        notes.append(
            f"{int(excl.sum())} mitochondrial/ribosomal genes were excluded from "
            f"highly-variable-gene selection and from the marker-gene test, so "
            f"clusters are defined by cell identity rather than by stress and "
            f"translation programmes. They remain in the matrix and are still "
            f"available for differential expression."
        )

    batch_key, may_correct = resolve_batch_correction(
        obs, ecfg, condition_columns, notes
    )

    # Reuse an embedding that is already present, unless told otherwise.
    if (
        prior is not None
        and getattr(prior, "can_skip_embedding", False)
        and not cfg.force_recompute
    ):
        result = _reuse_embedding(gex, cfg, prior, notes, step)
        result.notes = notes
        return result

    # Cached embedding from a previous identical run. The key covers the input
    # file, the retained barcodes and every embedding setting, so a cache hit
    # means the answer would have been the same -- and a miss says why.
    cache_key = None
    if cfg.use_checkpoints and not cfg.force_recompute:
        obs_names = [str(x) for x in gex.obs.index]
        var_names_all = [str(v) for v in gex.var.index]
        cache_key = EMBCACHE.embedding_key(cfg.h5ad_path, obs_names, cfg)
        payload, reason = EMBCACHE.load(cfg, cache_key, obs_names, var_names_all)
        if payload is not None:
            step("reusing the cached embedding (PCA, UMAP, clusters)")
            X_log = sparse_log1p(normalize_rows(gex.X, ecfg.target_sum))
            cached_obs = EMBCACHE.apply(payload, gex.obs)
            notes.append(
                "PCA, UMAP and clusters were loaded from this run directory's "
                "checkpoint rather than recomputed: the input file, the "
                "retained cells and every embedding setting are unchanged since "
                "the run that produced them. Pass --force-recompute to rebuild."
            )
            return EmbeddingResult(
                X_log=X_log, var_names=var_names_all, obs=cached_obs,
                pca=payload["pca"], umap=payload["umap"], hvg=payload["hvg"],
                backend=payload["backend"] + " (cached)",
                batch_corrected=payload["batch_corrected"], notes=notes,
            )
        notes.append(f"Embedding not reused: {reason}.")

    if _HAVE_SCANPY:
        result = _process_with_scanpy(
            gex, cfg, excl, batch_key, notes, step, may_correct
        )
    else:
        notes.append(
            "scanpy is not installed in this environment, so the transcriptome "
            "stage ran on a built-in numpy fallback: PCA by randomised SVD, "
            "k-means-style clustering instead of Leiden, and the first two "
            "principal components in place of UMAP. The clustering and embedding "
            "in this report are therefore NOT graph-based and should not be "
            "interpreted as Leiden clusters. Install scanpy (and leidenalg) for "
            "the intended behaviour."
        )
        result = _process_fallback(gex, cfg, excl, batch_key, notes)

    result.notes = notes

    # Cache the expensive outputs so a re-run of the same input with the same
    # settings skips HVG, regress_out, scale, PCA, harmony, UMAP and Leiden.
    if cache_key is not None:
        saved = EMBCACHE.save(result, cfg, cache_key)
        if saved is not None:
            step(f"cached the embedding for re-runs ({saved.name})")
            notes.append(
                f"PCA, UMAP and clusters were cached to "
                f"{saved.parent.name}/{saved.name}. A later run with the same "
                f"input and settings will reuse them instead of recomputing; "
                f"--force-recompute overrides."
            )

    return result


def _regress_and_scale(
    Xc: np.ndarray, covariates: np.ndarray | None, max_value: float | None
) -> np.ndarray:
    """Least-squares regression of covariates out of each gene, then z-scale.

    ``max_value=None`` skips the outlier clip.
    """
    if covariates is not None and covariates.size:
        C = np.column_stack([np.ones(Xc.shape[0]), covariates])
        keep = np.isfinite(C).all(axis=0)
        C = C[:, keep]
        try:
            beta, *_ = np.linalg.lstsq(C, Xc, rcond=None)
            Xc = Xc - C @ beta
        except np.linalg.LinAlgError:  # pragma: no cover
            pass
    mu = Xc.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (Xc - mu) / sd
    if max_value is None:
        return Z
    return np.clip(Z, -max_value, max_value)


def _covariate_matrix(
    obs: pd.DataFrame, which: Sequence[str]
) -> tuple[np.ndarray | None, list[str]]:
    cols, used = [], []
    for name in which:
        if name in obs.columns:
            v = pd.to_numeric(obs[name], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(v).any():
                cols.append(np.nan_to_num(v, nan=float(np.nanmean(v))))
                used.append(name)
    if not cols:
        return None, []
    return np.column_stack(cols), used


def _process_fallback(
    gex: Any, cfg: PipelineConfig, excl: np.ndarray, batch_key: str | None,
    notes: list[str],
) -> EmbeddingResult:
    ecfg = cfg.embedding
    var_names = [str(v) for v in gex.var.index]
    obs = gex.obs.copy()

    # Kept sparse throughout: normalise_rows and log1p both preserve sparsity
    # (log1p(0) == 0), so a 200k x 25k matrix stays at a few GB instead of 40.
    X_log = sparse_log1p(normalize_rows(gex.X, ecfg.target_sum))

    # HVG by variance of the log data, excluding MT/ribo.
    var = col_variances(X_log)
    var_masked = var.copy()
    var_masked[excl] = -np.inf
    n_top = int(min(ecfg.n_top_genes, int((~excl).sum())))
    hvg_idx = np.argsort(var_masked)[::-1][:n_top]
    hvg = [var_names[i] for i in sorted(hvg_idx)]

    obs["total_counts"] = row_sums(gex.X)
    if ecfg.score_cell_cycle:
        obs["S_score"] = _score_genes(X_log, var_names, S_GENES)
        obs["G2M_score"] = _score_genes(X_log, var_names, G2M_GENES)
        phase = np.where(
            (obs["S_score"] < 0) & (obs["G2M_score"] < 0), "G1",
            np.where(obs["S_score"] >= obs["G2M_score"], "S", "G2M"),
        )
        obs["phase"] = np.where(
            obs["S_score"].isna().to_numpy(), "unknown", phase
        )

    cov, used = _covariate_matrix(obs, ecfg.regress_out)
    if used:
        notes.append(
            f"Regressed out of the embedding: {', '.join(used)}. Affects "
            f"clusters and the 2-D projection only; X_log is untouched."
        )
    elif not list(ecfg.regress_out or ()):
        notes.append(
            "No covariates were regressed out (the default from v1.3.0)."
        )
    if ecfg.scale_max_value is not None:
        notes.append(
            f"The scaled highly-variable block was clipped at "
            f"{ecfg.scale_max_value:g} standard deviations -- a convention "
            f"inherited from Seurat's ScaleData(scale.max = 10) via the scanpy "
            f"tutorial, not a derived threshold. It affects the embedding only."
        )
    # Only the HVGs are densified for scaling/PCA -- a few thousand columns,
    # not the whole transcriptome.
    Xs = _regress_and_scale(
        take_columns(X_log, sorted(hvg_idx)), cov, ecfg.scale_max_value
    )

    pca = _fallback_pca(Xs, ecfg.n_pcs, ecfg.random_state)
    labels = _fallback_cluster(pca, ecfg.leiden_resolution, ecfg.random_state)
    labels, merge_notes = merge_tiny_clusters(labels, pca, ecfg.min_cluster_frac)
    notes.extend(merge_notes)
    notes.extend(small_cluster_notes(labels, ecfg))
    obs["cluster"] = pd.Categorical(labels)

    if ecfg.detect_doublets:
        score, call = _detect_doublets_fallback(pca, ecfg.random_state)
        obs["doublet_score"] = score
        obs["predicted_doublet"] = call
        notes.append(DOUBLET_FALLBACK_NOTE.format(
            n=int(call.sum()), total=len(call), pct=100.0 * float(call.mean()),
        ))

    umap = pca[:, :2] if pca is not None and pca.shape[1] >= 2 else None

    return EmbeddingResult(
        X_log=X_log, var_names=var_names, obs=obs, pca=pca, umap=umap, hvg=hvg,
        backend="numpy-fallback", batch_corrected="none",
    )


def _process_with_scanpy(  # pragma: no cover - requires scanpy
    gex: Any, cfg: PipelineConfig, excl: np.ndarray, batch_key: str | None,
    notes: list[str], step=None, may_correct: bool = False,
) -> EmbeddingResult:
    def _s(label: str) -> None:
        if step is not None:
            step(label)
    ecfg = cfg.embedding

    # scanpy parallelises regress_out and parts of neighbors over
    # settings.n_jobs. PipelineConfig.n_jobs existed from v1.1.0 and was never
    # assigned to it, so both ran single-threaded on however many cores the
    # machine had. One line, and it is the second-cheapest speedup available.
    try:
        sc.settings.n_jobs = int(cfg.n_jobs)
    except Exception:
        pass

    # copy_input=False since v1.3.0: this is the largest single allocation in
    # the stage and the pipeline never reads the input object again. Library
    # callers who still need their AnnData should set it back to True.
    _s("copy input matrix" if ecfg.copy_input else "using input matrix in place")
    ad = gex.copy() if ecfg.copy_input else gex
    var_names = [str(v) for v in ad.var.index]
    ad.var["excluded_from_hvg"] = excl

    n_top = int(min(ecfg.n_top_genes, ad.n_vars))
    hvg_done = False

    # HVG pairing: seurat_v3 expects RAW COUNTS, the other flavours expect LOG
    # data. Getting this backwards is a classic silent error.
    #
    # For seurat_v3 we therefore select features BEFORE normalising, while X is
    # still counts. The previous version instead stashed a full copy of the
    # counts in ``layers["counts"]`` and selected afterwards -- correct, but it
    # doubled peak memory for no benefit. Also removed: ``ad.raw = ad``, a third
    # full copy that nothing in this pipeline ever read.
    #
    # HVG selection is NOT batch-aware by default from v1.3.0. Until then the
    # resolved batch_key was passed straight through, so with seurat_v3 gene
    # variance was ranked within each batch -- correct when batch is a
    # nuisance, and wrong when "batch" is `sample` and sample is the condition,
    # because it then de-prioritises exactly the genes that differ between
    # conditions. Set embedding.hvg_batch_key explicitly to opt back in.
    hvg_batch = ecfg.hvg_batch_key or None
    if hvg_batch is not None and hvg_batch not in ad.obs.columns:
        notes.append(
            f"embedding.hvg_batch_key={hvg_batch!r} is not a column in obs; "
            f"highly-variable-gene selection ran without a batch key."
        )
        hvg_batch = None
    if hvg_batch is not None:
        notes.append(
            f"Highly-variable-gene selection was batch-aware on "
            f"{hvg_batch!r} (explicitly requested). Gene variance is ranked "
            f"within each level and the ranks combined, so genes that differ "
            f"only between levels are de-prioritised. Do not set this to a "
            f"condition column."
        )
    try:
        if ecfg.hvg_flavor == "seurat_v3":
            _s(f"HVG selection on raw counts (seurat_v3, top {n_top}, "
               f"batch_key={hvg_batch!r})")
            sc.pp.highly_variable_genes(
                ad, n_top_genes=n_top, flavor="seurat_v3",
                batch_key=hvg_batch,
            )
            hvg_done = True
    except Exception as exc:
        notes.append(
            f"HVG selection on raw counts with flavor='seurat_v3' failed "
            f"({exc}); falling back to flavor='seurat' on the log data."
        )

    if ecfg.detect_doublets:
        # Scrublet expects RAW COUNTS and normalises internally. Running it
        # after normalize_total + log1p double-transformed the data, which is
        # also where the repeated "adata.X seems to be already log-transformed"
        # warnings came from -- one per batch, because batch_key makes it
        # re-normalise each batch separately.
        _s(f"doublet detection (scrublet, batch_key={batch_key!r})")
        called = False
        try:
            sc.pp.scrublet(ad, batch_key=batch_key)
            called = True
        except Exception as exc:
            notes.append(
                f"scrublet is unavailable or failed ({exc}); using the built-in "
                f"synthetic-doublet fallback instead."
            )
        if called:
            n_d = int(ad.obs.get("predicted_doublet", pd.Series(dtype=bool)).sum())
            notes.append(
                f"Scrublet flagged {n_d:,} of {ad.n_obs:,} cells "
                f"({100.0 * n_d / max(ad.n_obs, 1):.1f}%) as predicted doublets. "
                f"They are RETAINED in the embedding and in every downstream "
                f"analysis; the panel exists to show the rate and location of the "
                f"calls."
            )


    _s("normalize_total")
    sc.pp.normalize_total(ad, target_sum=ecfg.target_sum)
    _s("log1p")
    sc.pp.log1p(ad)

    try:
        if not hvg_done:
            sc.pp.highly_variable_genes(
                ad, n_top_genes=n_top,
                flavor=("seurat" if ecfg.hvg_flavor == "seurat_v3"
                        else ecfg.hvg_flavor),
                batch_key=hvg_batch,
            )
    except Exception as exc:
        notes.append(
            f"HVG selection with flavor={ecfg.hvg_flavor!r} failed ({exc}); fell "
            f"back to flavor='seurat' on the log-normalised data."
        )
        sc.pp.highly_variable_genes(ad, n_top_genes=int(min(ecfg.n_top_genes,
                                                           ad.n_vars)))

    # Exclusion applied AFTER selection so the ranking is unaffected, then the
    # excluded genes are simply not used as features.
    ad.var["highly_variable"] = ad.var["highly_variable"].to_numpy() & ~excl
    hvg = [v for v, f in zip(var_names, ad.var["highly_variable"]) if f]

    if ecfg.score_cell_cycle:
        upper = {v.upper(): v for v in var_names}
        s_genes = [upper[g] for g in (x.upper() for x in S_GENES) if g in upper]
        g2m_genes = [upper[g] for g in (x.upper() for x in G2M_GENES) if g in upper]
        if s_genes and g2m_genes:
            _s("cell-cycle scoring")
            try:
                sc.tl.score_genes_cell_cycle(ad, s_genes=s_genes,
                                             g2m_genes=g2m_genes)
            except Exception as exc:
                notes.append(f"Cell-cycle scoring failed: {exc}")
        else:
            notes.append(
                "Cell-cycle scoring skipped: none of the standard human S/G2M "
                "marker genes are present in this reference."
            )

    _s(f"subset to {int(ad.var['highly_variable'].sum()):,} highly variable genes")
    ad_hvg = ad[:, ad.var["highly_variable"]].copy()
    cov, used = _covariate_matrix(ad_hvg.obs, ecfg.regress_out)
    if used:
        try:
            _s(
                f"regress_out {list(used)} on {ad_hvg.n_obs:,} x "
                f"{ad_hvg.n_vars:,} -- the most memory- and time-hungry step; "
                f"set embedding.regress_out=[] to skip it"
            )
            sc.pp.regress_out(ad_hvg, list(used))
            notes.append(
                f"Regressed out of the embedding: {', '.join(used)}. This "
                f"affects clusters, UMAP and E-distance only -- X_log, and "
                f"therefore every differential-expression result, is "
                f"untouched."
            )
        except Exception as exc:
            notes.append(f"regress_out({used}) failed: {exc}; continuing unregressed.")
    else:
        wanted = list(ecfg.regress_out or ())
        if wanted:
            notes.append(
                f"Covariate regression was configured for {wanted} but none of "
                f"those columns were present on obs, so NOTHING was regressed "
                f"out. This is stated because a caption claiming otherwise is "
                f"how v1.1.0 through v1.2.4 hid the fact that depth and %mito "
                f"were never being removed."
            )
        else:
            notes.append(
                "No covariates were regressed out, which is the default from "
                "v1.3.0. The step only ever touched the HVG block used for the "
                "embedding -- never X_log -- so it could not change a fold "
                "change, and it was the longest step in the stage. Cell-cycle "
                "regression in particular is withheld deliberately: many "
                "knockouts are proliferation phenotypes, so removing "
                "S_score/G2M_score suppresses exactly the perturbations being "
                "screened for. Scores are still computed and shown in the "
                "phase panel. Pass --regress-qc to remove depth and %mito."
            )

    _s(f"scale ({ad_hvg.n_obs:,} x {ad_hvg.n_vars:,}) -- this densifies the HVG block")
    if ecfg.scale_max_value is None:
        sc.pp.scale(ad_hvg)
        notes.append(
            "The scaled HVG block was not clipped (scale_max_value=None)."
        )
    else:
        sc.pp.scale(ad_hvg, max_value=ecfg.scale_max_value)
        # How many entries the clip actually bit on. If this is large,
        # something upstream is wrong and it is worth knowing rather than
        # having it quietly absorbed.
        try:
            n_at = int(np.sum(np.asarray(ad_hvg.X) >= ecfg.scale_max_value))
            frac = 100.0 * n_at / max(ad_hvg.n_obs * ad_hvg.n_vars, 1)
            clip_detail = f" {n_at:,} entries ({frac:.4f}% of the block) sit at the cap."
        except Exception:
            clip_detail = ""
        notes.append(
            f"The scaled highly-variable block was clipped at "
            f"{ecfg.scale_max_value:g} standard deviations.{clip_detail} This is "
            f"a convention inherited from Seurat's ScaleData(scale.max = 10) via "
            f"the scanpy tutorial, not a derived threshold: it stops a handful "
            f"of extreme cells dominating a gene's contribution to PC1. It "
            f"affects the embedding only."
        )

    _s("PCA")
    # svd_solver='randomized' rather than scanpy's arpack default: the HVG
    # block here is wide and dense, which is the case randomized SVD is for.
    try:
        sc.tl.pca(ad_hvg, n_comps=int(min(ecfg.n_pcs, min(ad_hvg.shape) - 1)),
                  random_state=ecfg.random_state, svd_solver="randomized")
    except (TypeError, ValueError):
        sc.tl.pca(ad_hvg, n_comps=int(min(ecfg.n_pcs, min(ad_hvg.shape) - 1)),
                  random_state=ecfg.random_state)

    rep = "X_pca"
    corrected = "none"
    if may_correct and batch_key:
        try:
            _s(f"harmony batch correction on {batch_key!r}")
            sc.external.pp.harmony_integrate(ad_hvg, key=batch_key)
            rep = "X_pca_harmony"
            corrected = f"harmony on {batch_key!r}"
            notes.append(
                f"Batch correction: harmony applied to the PCA embedding using "
                f"{batch_key!r}, explicitly requested via --batch-correct "
                f"harmony. Neighbours, UMAP and Leiden all run on the corrected "
                f"representation, and so does E-distance in the perturbation "
                f"section, because it is computed on this same PCA array. Small "
                f"transcriptional differences that track {batch_key!r} will be "
                f"attenuated."
            )
        except Exception as exc:
            notes.append(
                f"Batch correction was requested but harmony is unavailable or "
                f"failed ({exc}). The embedding and clustering below are "
                f"UNCORRECTED for {batch_key!r} -- use the {batch_key}-coloured "
                f"UMAP panel before interpreting clusters as biology."
            )

    _s("nearest-neighbour graph")
    sc.pp.neighbors(ad_hvg, n_neighbors=ecfg.n_neighbors, use_rep=rep,
                    random_state=ecfg.random_state)
    _s("UMAP")
    sc.tl.umap(ad_hvg, random_state=ecfg.random_state)
    try:
        _s("Leiden clustering")
        sc.tl.leiden(ad_hvg, resolution=ecfg.leiden_resolution,
                     key_added="cluster", random_state=ecfg.random_state,
                     flavor="igraph", n_iterations=2, directed=False)
    except TypeError:
        sc.tl.leiden(ad_hvg, resolution=ecfg.leiden_resolution,
                     key_added="cluster", random_state=ecfg.random_state)

    pca = np.asarray(ad_hvg.obsm[rep])
    labels, merge_notes = merge_tiny_clusters(
        ad_hvg.obs["cluster"].astype(str).to_numpy(), pca, ecfg.min_cluster_frac
    )
    notes.extend(merge_notes)
    notes.extend(small_cluster_notes(labels, ecfg))

    obs = ad.obs.copy()
    obs["cluster"] = pd.Categorical(labels)
    # GATED since v1.3.0. Up to v1.2.5 this ran on the condition
    # `"predicted_doublet" not in obs.columns` alone, which is true precisely
    # when detection was switched off -- so turning doublets off swapped
    # scrublet for this fallback instead of skipping the step, silently and
    # with no note.
    if ecfg.detect_doublets and "predicted_doublet" not in obs.columns:
        score, call = _detect_doublets_fallback(pca, ecfg.random_state)
        obs["doublet_score"] = score
        obs["predicted_doublet"] = call
        notes.append(DOUBLET_FALLBACK_NOTE.format(
            n=int(call.sum()), total=len(call),
            pct=100.0 * float(call.mean()),
        ))

    return EmbeddingResult(
        X_log=ad.X, var_names=var_names, obs=obs, pca=pca,
        umap=np.asarray(ad_hvg.obsm["X_umap"]), hvg=hvg,
        backend="scanpy", batch_corrected=corrected,
    )


# ===========================================================================
# Marker genes
# ===========================================================================
def _marker_pvalues(
    Xk: Any, in_rows: np.ndarray, out_rows: np.ndarray, block: int = 512
) -> np.ndarray:
    """Two-sided Mann-Whitney p-value for every column of ``Xk``.

    Blocked over columns so peak memory is bounded by ``block`` genes rather
    than by the gene count. Columns carrying no information come back NaN, so
    ``benjamini_hochberg`` excludes them from the family instead of counting
    them as p = 1.
    """
    k = int(Xk.shape[1])
    pv = np.full(k, np.nan)
    n1, n2 = int(in_rows.size), int(out_rows.size)
    if n1 == 0 or n2 == 0 or k == 0:
        return pv

    use_sparse = hasattr(Xk, "tocsc")
    Ain = Xk[in_rows]
    Aout = Xk[out_rows]
    if use_sparse:
        # One conversion; column slices of a CSC are then cheap.
        Ain = Ain.tocsc()
        Aout = Aout.tocsc()
        # The sparse ranking treats zeros as one tie group at the bottom, which
        # is only valid for non-negative data. Checked rather than assumed: a
        # scaled or residual matrix would silently break it.
        gmin = float(Ain.data.min()) if Ain.data.size else 0.0
        rmin = float(Aout.data.min()) if Aout.data.size else 0.0
        if gmin < 0.0 or rmin < 0.0:
            use_sparse = False

    for start in range(0, k, block):
        stop = min(start + block, k)
        g, r = Ain[:, start:stop], Aout[:, start:stop]
        if use_sparse:
            _u, p = mannwhitney_u_sparse_columns(g, r, n1, n2)
        else:
            g = g.toarray() if hasattr(g, "toarray") else np.asarray(g)
            r = r.toarray() if hasattr(r, "toarray") else np.asarray(r)
            _u, p = mannwhitney_u_columns(
                g.astype(np.float64), r.astype(np.float64)
            )
        pv[start:stop] = p
    return pv


def compute_markers(
    X_log: np.ndarray,
    var_names: Sequence[str],
    clusters: pd.Series,
    excluded: np.ndarray,
    n_genes: int = 10,
    max_genes_tested: int = 4000,
) -> pd.DataFrame:
    """One-vs-rest marker genes per cluster, excluding MT/ribosomal genes.

    Ranked by effect size with a Mann-Whitney p-value, restricted to the most
    variable genes for tractability. Excluding MT/ribo here is the difference
    between a marker list you can interpret and a list of ribosomal proteins.

    ``max_genes_tested`` defines the multiple-testing family: every one of
    those genes is tested and BH is applied across all of them, so ``padj``
    means what it says. Only enriched genes (log2FC > 0) are returned, but that
    selection happens *after* the correction, which does not affect it.
    """
    labels = pd.Series(clusters).astype(str).to_numpy()
    keep = ~np.asarray(excluded)
    idx_all = np.flatnonzero(keep)
    if idx_all.size > max_genes_tested:
        var = col_variances(X_log)
        var_keep = np.full(var.shape, -np.inf)
        var_keep[idx_all] = var[idx_all]
        idx_all = np.argsort(var_keep)[::-1][:max_genes_tested]
    idx_all = np.sort(idx_all)
    names = [var_names[i] for i in idx_all]

    # Per-cluster means are computed with sparse aggregates -- no densification.
    # expm1 preserves sparsity (expm1(0) == 0), so the linear-scale matrix is
    # just the same sparsity pattern with transformed data.
    Xk = to_csc(X_log)[:, idx_all] if hasattr(X_log, "tocsc") else \
        np.asarray(X_log, dtype=np.float64)[:, idx_all]
    if hasattr(Xk, "tocsr"):
        Xk = Xk.tocsr()
        Xlin = Xk.copy()
        Xlin.data = np.expm1(Xlin.data)
    else:
        Xlin = np.expm1(Xk)

    def _means(M, mask):
        sub = M[np.flatnonzero(mask)]
        if hasattr(sub, "mean"):
            out = sub.mean(axis=0)
            return np.asarray(out).ravel().astype(np.float64)
        return np.asarray(sub, dtype=np.float64).mean(axis=0)

    def _frac_nonzero(M, mask):
        sub = M[np.flatnonzero(mask)]
        n = max(int(mask.sum()), 1)
        if hasattr(sub, "getnnz"):
            return np.asarray((sub != 0).sum(axis=0)).ravel() / n
        return (np.asarray(sub) != 0).mean(axis=0)

    rows = []
    for cl in sorted(pd.unique(labels)):
        m = labels == cl
        if m.sum() < 3 or (~m).sum() < 3:
            continue
        mean_in = _means(Xlin, m)
        mean_out = _means(Xlin, ~m)
        frac_in = _frac_nonzero(Xk, m)
        log2fc = np.log2((mean_in + 1e-9) / (mean_out + 1e-9))

        # Test every screened gene and apply BH across all of them.
        #
        # The original pre-ranked by fold change, tested only the top
        # max(n_genes * 6, 60), and then applied BH with n = that subset. Two
        # compounding errors, both pushing padj down: the denominator was the
        # selected subset rather than the family screened (up to
        # ``max_genes_tested``, i.e. understated by up to ~67x), and the subset
        # had been chosen using the same data as the test.
        #
        # Testing everything is affordable because the column-wise ranking in
        # stats.py does a whole block in one pass. The per-gene scipy loop this
        # replaces is what made testing all genes look expensive.
        in_rows = np.flatnonzero(m)
        out_rows = np.flatnonzero(~m)
        pvals = _marker_pvalues(Xk, in_rows, out_rows)
        padj = benjamini_hochberg(pvals)
        sub = pd.DataFrame(
            {
                "cluster": str(cl),
                "gene": list(names),
                "log2fc": log2fc,
                "mean_in_cluster": mean_in,
                "mean_out_cluster": mean_out,
                "frac_expressing": frac_in,
                "pvalue": pvals,
                "padj": padj,
            }
        )
        # Enriched genes only, and drop the untestable ones: a NaN padj means
        # "carried no information", not "not significant". Both filters are
        # applied after the correction, so neither changes it.
        sub = sub[np.isfinite(sub["padj"].to_numpy()) & (sub["log2fc"] > 0.0)]
        sub = sub.sort_values(["padj", "log2fc"], ascending=[True, False])
        rows.append(sub.head(n_genes))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarise_clusters(
    obs: pd.DataFrame, markers: pd.DataFrame, sample_col: str | None
) -> pd.DataFrame:
    """Per-cluster size, markers, phase mix and a generated description."""
    labels = obs["cluster"].astype(str)
    total = len(labels)
    rows = []
    for cl in sorted(labels.unique(), key=lambda x: (len(x), x)):
        m = (labels == cl).to_numpy()
        n = int(m.sum())
        top = (
            markers.loc[markers["cluster"] == cl, "gene"].tolist()
            if not markers.empty else []
        )
        phase_frac = None
        if "phase" in obs.columns:
            phase_frac = (
                obs.loc[m, "phase"].astype(str).value_counts(normalize=True).to_dict()
            )
        top_sample = None
        if sample_col and sample_col in obs.columns:
            vc = obs.loc[m, sample_col].astype(str).value_counts(normalize=True)
            if len(vc):
                top_sample = (str(vc.index[0]), float(vc.iloc[0]))
        dfrac = (
            float(obs.loc[m, "predicted_doublet"].astype(bool).mean())
            if "predicted_doublet" in obs.columns else None
        )
        rows.append(
            {
                "cluster": cl,
                "n_cells": n,
                "pct_cells": 100.0 * n / total if total else np.nan,
                "top_markers": ", ".join(top[:6]),
                "dominant_phase": (
                    max(phase_frac, key=lambda k: phase_frac[k])
                    if phase_frac else ""
                ),
                "pct_predicted_doublet": 100.0 * dfrac if dfrac is not None else np.nan,
                "description": T.describe_cluster(
                    cl, n, n / total if total else 0.0, top, phase_frac,
                    top_sample, dfrac,
                ),
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# Figures
# ===========================================================================
def plot_umaps(
    res: EmbeddingResult, cfg: PipelineConfig, path: Path,
    extra_colors: dict[str, pd.Series] | None = None,
) -> Path:
    fcfg = cfg.figures
    panels: list[tuple[str, pd.Series, dict]] = [
        ("Leiden cluster" if res.backend == "scanpy" else "cluster (fallback)",
         res.obs["cluster"].astype(str), {"categorical": True,
                                          "label_clusters": True}),
    ]
    if "phase" in res.obs.columns:
        panels.append(("cell-cycle phase", res.obs["phase"].astype(str),
                       {"categorical": True}))
    if "predicted_doublet" in res.obs.columns:
        panels.append(
            ("predicted doublet (retained)",
             res.obs["predicted_doublet"].map({True: "doublet", False: "singlet"})
             .astype(str), {"categorical": True})
        )
    if "total_counts" in res.obs.columns:
        panels.append(("total UMI counts", res.obs["total_counts"],
                       {"categorical": False, "cap_percentile": 99}))
    for name, series in (extra_colors or {}).items():
        panels.append((name, series.astype(str), {"categorical": True}))

    nrows, ncols = P.grid_dims(len(panels), max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows),
                             squeeze=False)
    for ax, (title, values, kw) in zip(axes.ravel(), panels):
        P.scatter_embedding(ax, res.umap, values, fcfg, title=title, **kw)
    P.blank_unused_axes(axes, len(panels))
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_marker_dotplot(
    res: EmbeddingResult, markers: pd.DataFrame, cfg: PipelineConfig, path: Path
) -> Path:
    if markers.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        P.annotate_empty(ax, "no marker genes passed the test")
        return P.save_figure(fig, path, cfg.figures)

    clusters = sorted(markers["cluster"].unique(), key=lambda x: (len(x), x))
    columns, blocks = [], []
    for cl in clusters:
        genes = [g for g in markers.loc[markers["cluster"] == cl, "gene"]
                 if g not in columns]
        start = len(columns)
        columns.extend(genes)
        blocks.append((cl, start, len(columns)))

    name_to_idx = {v: i for i, v in enumerate(res.var_names)}
    idx = [name_to_idx[g] for g in columns if g in name_to_idx]
    columns = [g for g in columns if g in name_to_idx]
    labels = res.obs["cluster"].astype(str).to_numpy()
    # One dense block of just the marker genes, reused for every cluster.
    marker_block = take_columns(res.X_log, idx)

    size = pd.DataFrame(index=clusters, columns=columns, dtype=float)
    color = pd.DataFrame(index=clusters, columns=columns, dtype=float)
    for cl in clusters:
        m = labels == cl
        sub = marker_block[np.flatnonzero(m)]
        size.loc[cl] = (sub > 0).mean(axis=0)
        color.loc[cl] = sub.mean(axis=0)

    width = max(8.0, 0.20 * len(columns) + 3.0)
    height = max(3.0, 0.34 * len(clusters) + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))
    P.dotplot(
        ax, size, color, cfg.figures,
        size_label="fraction expressing", color_label="mean log-expression",
        cmap=cfg.figures.continuous_cmap, column_blocks=blocks,
    )
    ax.set_title("marker genes per cluster (MT/ribosomal genes excluded)",
                 fontsize=9)
    fig.tight_layout()
    return P.save_figure(fig, path, cfg.figures)


def plot_cluster_composition(
    res: EmbeddingResult, group: pd.Series, group_name: str,
    cfg: PipelineConfig, path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    labels = res.obs["cluster"].astype(str)
    g = group.astype(str).reindex(res.obs.index)

    frac = pd.crosstab(g, labels, normalize="index")
    P.stacked_fraction_bars(axes[0], frac, cfg.figures, legend_title="cluster")
    axes[0].set_title(f"cluster composition of each {group_name}")

    frac2 = pd.crosstab(labels, g, normalize="index")
    P.stacked_fraction_bars(axes[1], frac2, cfg.figures, legend_title=group_name)
    axes[1].set_xlabel("cluster")
    axes[1].set_title(f"{group_name} composition of each cluster")
    fig.tight_layout()
    return P.save_figure(fig, path, cfg.figures)


# ===========================================================================
# Stage driver
# ===========================================================================
def run_transcriptome_stage(
    gex: Any,
    cfg: PipelineConfig,
    reg: Registry,
    group_columns: dict[str, pd.Series],
    sample_col: str | None = None,
    step=None,
    prior: Any = None,
    condition_columns: Sequence[str] | None = None,
) -> EmbeddingResult:
    res = process_transcriptome(
        gex, cfg, reg, step, prior,
        condition_columns=(
            list(condition_columns) if condition_columns is not None
            else list(group_columns.keys())
        ),
    )
    fig_dir, table_dir = cfg.fig_dir, cfg.table_dir

    reg.note("transcriptome", "method", "What was done", T.EMBEDDING_DESC, order=1)
    for i, note in enumerate(res.notes):
        level = (
            "warn"
            if any(k in note for k in ("UNCORRECTED", "NOT graph-based", "failed"))
            else "info"
        )
        reg.note("transcriptome", f"note_{i}", "Processing note", note,
                 level=level, order=5 + i)

    reg.metric("summary", "n_clusters", "Clusters",
               int(res.obs["cluster"].nunique()), order=50)
    if "predicted_doublet" in res.obs.columns:
        reg.metric(
            "summary", "pct_doublets", "Predicted doublets (retained)",
            round(100.0 * float(res.obs["predicted_doublet"].astype(bool).mean()), 1),
            unit="%", order=51,
        )

    if "predicted_doublet" not in res.obs.columns:
        reg.skipped(
            "transcriptome", "doublets", "Predicted doublets",
            "Doublet detection is off by default (embedding.detect_doublets), "
            "so no doublet scores were computed and no rate is reported. "
            "Scrublet assumes a heterogeneous population and is not "
            "informative on a homogeneous Perturb-seq line; the built-in "
            "fallback is brute-force all-pairs and costs roughly ten minutes "
            "at 187k cells for a rate that its own threshold rule pins near "
            "5-10% regardless of the data. Pass --doublets to turn it back on. "
            "Note that in v1.1.0 through v1.2.5 this switch was only half "
            "effective: turning detection off substituted the fallback for "
            "scrublet rather than skipping the step, so earlier reports may "
            "show a doublet panel that was never asked for.",
        )

    # The sample axis is added UNCONDITIONALLY, not just when it happens to be
    # among the resolved condition axes. Several notes tell the reader to check
    # the sample-coloured UMAP; up to v1.2.5 `extra` was exactly
    # `group_columns`, and `sample` only entered that as a fallback for
    # single-condition experiments -- so in any multi-condition run the panel
    # those notes pointed at did not exist.
    extra = {k: v for k, v in group_columns.items()}
    if sample_col and sample_col in res.obs.columns:
        s = res.obs[sample_col].astype(str)
        if s.nunique() >= 2 and sample_col not in extra:
            extra[sample_col] = s
    # Both of these used to run with no step() call in between, so their cost
    # (rendering every panel for the full cell count, then one-vs-rest
    # Mann-Whitney testing across every cluster) was silently folded into
    # whatever the LAST step() label happened to be -- typically one of the
    # fast, one-off actions inside process_transcriptome/_reuse_embedding, so
    # a run reusing an existing embedding could show something like "33m46s
    # for: normalising the reused object" when normalisation itself is a
    # seconds-long vectorised operation and the real cost was here, several
    # steps downstream and unrelated. Each gets its own label so the log
    # attributes time to the work that actually took it.
    if step:
        step(f"plotting UMAP panels ({res.obs.shape[0]:,} cells)")
    reg.figure(
        "transcriptome", "umaps", "Embedding",
        plot_umaps(res, cfg, fig_dir / "umap_panels.png", extra),
        caption=" ".join(
            [T.UMAP_LEIDEN_DESC, T.UMAP_PHASE_DESC, T.UMAP_DOUBLET_DESC,
             T.UMAP_TOTAL_COUNTS_DESC, T.UMAP_BATCH_DESC]
        ),
        order=10, width="full",
    )

    excl = excluded_feature_mask(res.var_names, cfg.modality, cfg.embedding)
    if step:
        step(
            f"computing cluster marker genes "
            f"({int(res.obs['cluster'].nunique())} clusters)"
        )
    markers = compute_markers(
        res.X_log, res.var_names, res.obs["cluster"], excl,
        cfg.embedding.n_marker_genes,
    )
    res.markers = markers
    if step:
        step("rendering marker dotplot and cluster summary")
    if not markers.empty:
        markers.to_csv(table_dir / "cluster_markers.csv", index=False)
        reg.figure(
            "transcriptome", "markers", "Cluster marker genes",
            plot_marker_dotplot(res, markers, cfg, fig_dir / "cluster_markers.png"),
            caption=T.MARKERS_DESC, order=20, width="full",
        )
    else:
        reg.skipped("transcriptome", "markers", "Cluster marker genes",
                    "No gene reached significance in any one-vs-rest test.")

    summary = summarise_clusters(res.obs, markers, sample_col)
    res.cluster_summary = summary
    summary.to_csv(table_dir / "cluster_summary.csv", index=False)
    reg.table(
        "transcriptome", "cluster_summary", "Clusters",
        path=table_dir / "cluster_summary.csv",
        inline=summary[["cluster", "n_cells", "pct_cells", "description"]]
        .round(2).to_dict("records"),
        columns=["cluster", "n_cells", "pct_cells", "description"],
        caption=T.CLUSTER_INTERP_NOTE, order=30,
    )

    if step and group_columns:
        step(f"plotting cluster composition by condition ({len(group_columns)} axis/es)")
    for axis_name, series in group_columns.items():
        if series.astype(str).nunique() < 2:
            continue
        reg.figure(
            "transcriptome", f"composition_{axis_name}",
            f"Cluster composition by {axis_name}",
            plot_cluster_composition(res, series, axis_name, cfg,
                                     fig_dir / f"cluster_composition_{axis_name}.png"),
            caption=(
                "Left: which clusters each condition contains. Right: which "
                "conditions each cluster is made of. A cluster drawn almost "
                "entirely from one sample is the clearest signature of a batch "
                "effect being reported as a cell state."
            ),
            order=40, width="full",
        )

    pd.DataFrame({"hvg": res.hvg}).to_csv(table_dir / "hvg.csv", index=False)
    return res
