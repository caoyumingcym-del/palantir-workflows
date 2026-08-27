# v1.2.5

Three changes, one of which is a correctness bug that affects the run you are
about to start.

---

## 1. `regress_out` was silently applying only half its covariates (BUG)

`EmbeddingConfig.regress_out` is
`("total_counts", "pct_counts_mt", "S_score", "G2M_score")`, and the report's
caption said all four were regressed out of the scaled matrix.

They were not. `_covariate_matrix()` looks for those columns on `obs`, but the
QC metrics live in `qc_table`, which was never joined onto `gex.obs`. Only
`S_score` and `G2M_score` — computed inside the embedding stage itself — were
ever found. The log said so, quietly:

```
    regress_out ['S_score', 'G2M_score']
```

So sequencing depth and %mitochondrial were **not** removed before PCA, in
every version from v1.1.0 through v1.2.4. Depth in particular is usually the
largest single source of variance in PC1, which means clusters were partly depth
clusters.

Fixed by attaching the QC metrics to `obs` after the QC mask is applied:

```python
gex_f = split.gex[keep].copy()
_qc_kept = qc_table.loc[keep]
for _c in ("total_counts", "n_genes_by_counts", "pct_counts_mt",
           "pct_counts_ribo", "log10_genes_per_umi"):
    if _c in _qc_kept.columns and _c not in gex_f.obs.columns:
        gex_f.obs[_c] = _qc_kept[_c].to_numpy()
```

The log line now reads `regress_out ['total_counts', 'pct_counts_mt',
'S_score', 'G2M_score']`. If it does not, the covariates are still missing and
that is worth stopping for.

**Consequence:** clusters, UMAP and cluster markers will differ from the
v1.2.4 run. That is the fix working, not a regression.

---

## 2. Clustering is now actually checkpointed

`use_checkpoints` and `checkpoint_dir` existed since v1.1.0. The directory was
created. Nothing ever read or wrote it. Every re-run repeated HVG selection,
`regress_out`, scaling, PCA, harmony, neighbours, UMAP and Leiden — on a 187k
cell object, the bulk of the wall clock, for no new information.

New module `perturbseq_report/embedcache.py`. After a successful embedding the
stage writes `analysis_outputs/checkpoints/embedding_<key>.npz` plus a small
JSON sidecar, holding:

- PCA coordinates (float32)
- UMAP coordinates (float32)
- the HVG list
- `leiden`, `phase`, `S_score`, `G2M_score`, `doublet_score`,
  `predicted_doublet`
- the barcodes, so a partial match cannot pass

`X_log` is deliberately **not** cached: it is a `normalize_total` + `log1p` away
from the counts, which is seconds, and it is the largest array in the stage.
Marker genes are also not cached — they now run on the fast Wilcoxon path from
v1.2.4 and are cheap.

### Invalidation

The cache key is a content hash, not a timestamp. It covers:

- the input h5ad path, size and mtime
- a SHA-1 of the retained barcodes (two different QC threshold sets can retain
  the same *number* of cells)
- 14 embedding settings: `target_sum`, `n_top_genes`, the MT/ribo exclusion
  flags, `hvg_flavor`, `regress_out`, `scale_max_value`, `n_pcs`,
  `n_neighbors`, `leiden_resolution`, `batch_correct`, `score_cell_cycle`,
  `detect_doublets`, `random_state`

Change any of them and the cache **misses** and the report says why, rather
than returning a stale embedding. A stale embedding is precisely the failure
mode that hid a broken doublet step for three versions, so the load path also
refuses a different cell set, a different gene set, and an unreadable file —
a corrupt checkpoint misses, it does not raise.

No action needed on the existing run directory: because caching never actually
happened before v1.2.5, there are no stale checkpoints to invalidate. The first
v1.2.5 run recomputes and writes one; the second reuses it. (`CACHE_VERSION` is
part of the key, so if the cached contents ever change meaning, old files miss
rather than mislead.)

The report states in the transcriptome notes whether the embedding was
recomputed or reused, and reused runs are marked `scanpy (cached)` as the
backend so a cached figure can never be mistaken for a fresh one.

---

## 3. Guide purity per sample and per condition

The pooled purity panels answer "did guide calling work?". They cannot answer
"did it work *equally*?", which is the question that matters when the experiment
is a comparison — an assignment rate that differs by condition is a systematic
difference in guide capture, and it propagates into every per-condition number
downstream without ever looking like a bug.

New `plot_purity_by_group()`, registered once per axis: `sample` first, then
each condition axis the manifest nominates. Two rows, one column per level:

- **assignment criterion** — the `top1 / (top1 + top2)` histogram with the
  cut-off drawn on, titled with that level's cell count and the percentage
  above the cut-off, so the levels are directly comparable
- **purity triangle** — `top1/total` against `(top1+top2)/total`, with both
  gates drawn

Axes with fewer than two levels are skipped rather than rendered as a single
pointless column.

**Rename:** the third pooled panel, previously "strict secondary gate", is now
"purity triangle" everywhere.

---

## Tests

`tests/test_v121_changes.py` grows to ~66 checks. New:

- `test_embedding_cache` — key determinism; key changes on resolution and on
  cell set; miss-before-write; save/load round trip for PCA, UMAP, HVG, cluster
  labels and cell-cycle scores; refusal on a different cell set, a different
  gene set and a corrupt file; and that `gex.py` actually calls both
  `EMBCACHE.load` and `EMBCACHE.save` (a cache nothing calls is what v1.1.0
  already had)
- `test_purity_by_group` — the rename, the new `sample` parameter, the pipeline
  wiring, rendering with two levels and with one
- `test_regress_out_covariates_are_present` — the QC columns are attached, and
  the config still asks for them
