
# v1.3.2 — hashtag recovery from DRAGEN, and a knockdown-corrupting bug

## stats.py — `normalize_rows` was quietly slow on large matrices

A real run appeared to hang for 20+ minutes on "normalising the reused
object" for a 361,762 x 38,292 matrix, right after the `cluster_col_candidates`
fix let it correctly take the embedding-reuse path. It was not hung: the
sparse branch of `normalize_rows` computed `diags(scale) @ X` -- building a
full n x n sparse diagonal matrix and running a general sparse-sparse matrix
multiply to accomplish "scale each row's nonzero values by that row's factor".
Mathematically correct, but at hundreds of millions of nonzeros the overhead
of that approach is the difference between seconds and tens of minutes. This
is exactly the operation scanpy's `normalize_total` does internally via direct
CSR data scaling, which is why the *fresh*-computation path (which calls
scanpy directly) was never slow on the same-sized matrix, while every caller
of this function was.

Fixed to scale `X.data` directly (`X.data *= np.repeat(scale, row_nnz)`)
instead of going through a matrix multiply. Benefits all three callers in
`gex.py`: the raw-counts reuse branch, the embedding-cache reuse branch, and
the numpy-fallback fresh-computation path. Verified the scaling arithmetic
against a manual CSR reconstruction in pure numpy (this sandbox has no scipy
to test the real sparse path against) -- please confirm the reuse path
actually runs quickly now on ICA.

## config.py / cli.py — `cluster_col_candidates` and `--cluster-col`

A real run on MDL1856's re-processed h5ad (`..._gex370_dim30_res1.h5ad`) had
PCA and UMAP already present but got a full HVG/scale/PCA recompute anyway --
correctly, as it turned out: reusing an existing embedding requires PCA, UMAP,
*and* a recognised cluster column all present (`can_skip_embedding` in
`provenance.py`), and this object's cluster column wasn't named `leiden`,
`leiden_clusters`, `louvain` or `clusters`. It was `leiden_gpu` (RAPIDS/GPU
Leiden). Added to the built-in candidate list.

Since a differently-named cluster column will keep showing up across
datasets, also added `--cluster-col COLUMN` to try a given obs column before
the built-in list, so this doesn't need a code change (and a new pipeline
version) every time. Flat CLI overrides are already routed to whichever
subconfig declares the field by `config.build_config`, so this needed no new
plumbing beyond the argparse flag itself.

## pipeline.py — h5ad load no longer dies on an unreadable `uns` value

Real report: `ad.read_h5ad()` raised `IORegistryError: No read method
registered for IOSpec(encoding_type='null', ...)` while reading
`uns['solo_bycelltype_main']['A549']['threshold']`, killing the run before
anything else happened. **This is unrelated to the `NORMALISED`/`X_log` fix
above or anything from the DRAGEN hashtag work** -- it is a version-skew
problem between the anndata that WROTE this h5ad (new enough to encode a
`None` value with the `'null'` IOSpec) and the anndata INSTALLED in the run
environment (`/data/.local/...`, old enough to not know how to read that
encoding back). It surfaced now because this is evidently the first time this
particular h5ad -- with Solo doublet-detection metadata carrying a `None`
threshold for a cell type with too few cells to compute one -- was run through
this pipeline.

This pipeline never reads `adata.uns` from a real input file (only
`synthetic.py`'s test-fixture generator writes to it), so a value in `uns`
that the installed anndata can't decode should not be able to block the
entire run over metadata nothing downstream uses.

`_load_h5ad` now tries `ad.read_h5ad()` first as before; if it fails with this
specific IOSpec/registry error, it falls back to `_load_h5ad_skipping_uns`,
which reads `X`, `obs`, `var`, `obsm`, `varm`, `obsp`, `varp`, `layers` and
`raw` normally (through anndata's own `read_elem`, not a reimplementation) and
skips `uns` entirely. A registry note records that this happened and that
anything read FROM `uns` in this run (Solo metadata, etc.) will be absent from
the in-memory object, with `pip install -U anndata` named as the durable fix.
Any other read failure re-raises the original error unchanged.

**Not verified against a real anndata version-skew file** -- this sandbox has
no network access to install anndata/scipy, so this was checked for correct
error-detection and for not breaking the primary (working) read path, but not
against an actual mismatched-version h5ad. Please confirm this actually
resolves the load in the ICA environment before relying on it, and report
back if `_load_h5ad_skipping_uns` itself errors (the traceback will name which
top-level group failed).

## gex.py — the fix that changes reported numbers

`_reuse_embedding()`'s "is X already log-transformed?" check only branched on
two cases (`"raw counts"` vs. everything else), but `provenance.detect_x_state`
distinguishes **three**: `RAW_COUNTS`, `NORMALISED` (linear-scale, size-factor
or CPM-normalised, but not logged), and `LOG1P`. `NORMALISED` was falling into
the same "leave X alone" branch as `LOG1P`, so on an already-analysed h5ad
whose `X` held normalised-but-unlogged values, those values were passed
downstream labelled `X_log` without ever being log-transformed.

Every consumer of `X_log` — `percent_knockdown`, `differential_expression`,
`perturbation_score` — assumes `log_input=True` and calls `np.expm1()` on its
input to get back to linear scale before averaging. `expm1()` of an
already-linear value explodes: a highly-expressed housekeeping gene with a
normalised value in the tens (typical for VIM, TMSB10, and similar genes)
becomes a "mean expression" of 1e10–1e23. `pct_knockdown` itself stays bounded
inside its own formula (`(1 - mean_perturbed / mean_control) * 100` cannot
exceed 100 on the knockdown side), which is exactly why the corruption went
unnoticed for genes with small values but was visible as absurd
`mean_perturbed`/`mean_control` columns for the most highly-expressed targets
in a real run.

Fix: the reuse path now has three branches. `RAW_COUNTS` normalises and
log1p's as before. `LOG1P` passes through untouched as before. `NORMALISED`
is new: it applies `log1p` only, without re-normalising (normalisation was
already done upstream), and adds a registry note stating that this correction
fired. Verified against fake `RAW_COUNTS`/`NORMALISED`/`LOG1P` inputs directly
exercising `_reuse_embedding`.

## stats.py / hto.py — CLR docstring correction

`clr_by_feature`'s default (per-hashtag, across cells — Seurat's `margin = 2`
axis) was justified in the docstring by "what the WNN tutorial uses for ADT"
and implied this is also what the canonical `HTODemux` hashing vignette runs.
It is not: that vignette's `NormalizeData(..., normalization.method = "CLR")`
call does not set `margin`, so it uses Seurat's default of `margin = 1`
(per-cell). `margin = 2` is a separate convention Seurat recommends for
ADT/protein *visualisation* in the multimodal/WNN vignette, not for hashtag
demultiplexing.

The axis itself was not wrong — it matches the ORIGINAL Cell Hashing
definition (Stoeckius et al. 2018, *Genome Biology*: "counts were divided by
the geometric mean of an HTO across cells") — but the docstrings in both
`stats.py` (`clr_by_feature`, `clr_true_seurat`) and `hto.py`'s module
docstring cited the wrong source for it. Corrected to cite the original paper
and to state plainly that the hashing vignette's actual default is
`margin = 1`, so a reader comparing against "what Seurat runs" picks the
vignette that matches their purpose rather than assuming one margin is
universal.

## hto_dragen.py (new) — hashtag recovery from DRAGEN output

Some experiments' h5ads arrive with no hashtag/HTO matrix detected, even
though the raw counts exist in DRAGEN's per-run
`<prefix>.scRNA.cellhashing.tsv` output. This mirrors functionality the
pipeline's predecessor had and v1.3.1 lacked.

* `find_cellhashing_file` / `load_cellhashing_file` locate and parse the TSV,
  detecting orientation (barcodes as rows vs. columns) and dropping
  non-numeric columns with a report of what was dropped.
* `best_barcode_transform` tries several barcode transforms (as-is, strip/add
  `-N` suffix, prefix/suffix splits, case) against the target h5ad's
  `obs_names` and picks the one with the highest match fraction, rather than
  assuming barcodes will match as-is.
* `build_hto_modality_from_dragen` combines multiple DRAGEN runs for one
  sample either by summing counts (when the runs re-sequence the *same*
  cells/cDNA) or by keeping the strongest run per barcode (when runs are
  independent pools that happen to draw from the same finite barcode
  whitelist). Which mode applies is **not guessed** — see below.
* `build_hto_modality_multi_sample` orchestrates this across every sample in
  the manifest, refusing outright (with an actionable note, never a silent
  guess) when a sample has multiple runs and hasn't declared how they relate.

Wired into `pipeline.py`: when `split.hto` is not present but the manifest
declares hashtags and has DRAGEN run paths, this fallback runs before falling
back to "no hashtag data" and reports what it found (or why it couldn't) in
the appendix.

## manifest.py — `dragen_runs_share_cells`

New per-sample manifest column. When a sample has more than one DRAGEN run,
this declares whether those runs sequence the same cells/cDNA (`yes` — HTO
counts are summed across runs) or independent pools that happen to share a
finite barcode whitelist (`no` — the strongest-signal run wins per barcode).
Undeclared + multi-run is refused with a note rather than defaulted, because
guessing wrong here silently corrupts hashtag calling for every affected
sample. `Manifest.dragen_runs_share_cells(sample)` validates per-sample
consistency and rejects conflicting values across a sample's rows. Documented
in `examples/WHITELISTS.md`; `diagnose_cellhashing_tsv.py`'s Section G is the
empirical way to decide it for a given experiment.

## diagnose_cellhashing_tsv.py (new, repo root)

Standalone, read-only diagnostic — independent of whatever pipeline version is
actually deployed, since it imports only `perturbseq_report.manifest`. Takes a
sample manifest and checks, per sample: cellhashing.tsv structure and
orientation, barcode-format sanity, barcode match rate against the h5ad
(trying multiple transforms), and — when a sample has multiple runs — whether
the same barcode appears in more than one run's file with correlated counts
(evidence for "same cells, sum") or uncorrelated/exclusive counts (evidence
for "independent pools, keep strongest"). Defaults to checking only the first
run per sample; `--all-runs` checks every run.

---

Full test suite: `tests/test_hto_dragen.py` (9 tests, 20 checks, new),
`tests/test_units.py` (3 new manifest tests for
`dragen_runs_share_cells`), plus the existing v1.2.0/v1.2.1/v1.3.0/v1.3.1
suites — no regressions. `pytest`/`scipy`/`anndata` were unavailable in the
sandbox used for this round of changes (no network access to install them);
the `gex.py` fix was verified with a targeted regression script exercising
`_reuse_embedding` directly against fake `RAW_COUNTS`/`NORMALISED`/`LOG1P`
inputs. **Run the full suite in the ICA environment before trusting this in
production**, per the standing note in `PENDING_WORK.md` about this sandbox's
limits.
