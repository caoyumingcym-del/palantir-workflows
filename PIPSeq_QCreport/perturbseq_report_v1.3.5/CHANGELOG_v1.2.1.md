# perturbseq_report v1.2.1

Prepares v1.2.0 for a first run on a clean h5ad. Seven changes: one guard, one
performance fix, three usability fixes, two small ones. No change to any
statistic — the DE speedup is asserted to return identical numbers.

See `CHANGELOG_v1.2.0.md` for the correctness work this builds on.

---

## 1. Input plausibility gate — the guard that was missing

Testing v1.2.0 on `MDL1856_analyzed_full_depth.h5ad` produced two complete,
confident, entirely wrong reports before anyone noticed. That object's
`var_names` are **permuted relative to the columns of `X`**: nonzeros-per-cell
matched `obs['n_genes_by_counts']` exactly (1,417), but ACTB was detected in
0.02% of cells, EEF1A1 in none, and a CRISPR guide feature in 99.2%.

Guide calling, hashtag calling and QC gating were unaffected — they read `obs`
and `obsm`. Every gene-level result was reading a different gene than the one
it named.

New `sanity.py` runs two checks on the GEX matrix *after* guides and hashtags
are removed, so it tests exactly what feeds HVG selection and every gene
statistic:

- **Housekeeping detection.** Median detection across whichever of 15
  housekeeping genes are present must exceed 30%. Real matrices give ~90%; the
  failure above gave 0.02%. The threshold is deliberately loose so a shallow or
  heavily-filtered experiment is not rejected.
- **`var['n_cells_by_counts']` cross-check.** When scanpy wrote that column the
  labels matched the columns. If it now disagrees with the matrix by more than
  ~30%, they have since diverged.

On failure the pipeline **refuses to run** and prints the per-probe detection
table. `--skip-input-check` overrides. Neither check can prove a matrix is
correct; both can prove one is wrong, which is the useful direction.

## 2. Differential expression is ~20x faster

`differential_expression` called `mannwhitney_u` once per gene per target:
40 targets × 38,000 genes ≈ **1.5 million scipy calls**, each ranking ~30,000
values. It dominated every run.

**The obvious fix did not work.** A dense column-wise ranking — the standard
vectorisation, and what scanpy does — measured **1.0x**. Ranking a
21,000 × 500 block is dominated by `argsort` and memory traffic, and that cost
does not disappear because the loop moved into numpy.

What works is exploiting the zeros. Single-cell expression is ~96% zeros, and
log1p leaves them at zero, so they form one enormous tie group at the bottom of
every column. Their average rank is `(n_zero + 1) / 2` by inspection, and only
the ~4% nonzero values need sorting. The loop over columns is then cheap
because each iteration sorts a few hundred values instead of tens of thousands.
This is the trick presto uses. Measured **20–23x**.

The DE path no longer densifies at all: means and detection fractions are
computed from the sparse data directly (`expm1(0) == 0`, so the inverse
transform applies to the stored values alone), which also removes the ~400 MB
per-block allocation.

**Nothing about the statistic changed.** Wilcoxon's rank sum and Mann-Whitney's
U are the same test, related by `U1 = W1 - n1(n1+1)/2`. Tie correction and the
continuity correction match scipy's defaults, and the test suite asserts
`max|ΔU| == 0` and `max|Δp| == 0` against the per-gene implementation, plus
agreement of every DE table column to 1e-12 between the sparse and dense paths.
The speedup itself is asserted (≥5x) so a future change cannot quietly undo it.

Incidental fix: the no-scipy fallback in `mannwhitney_u` did not apply the
continuity correction while the scipy branch did, so p-values differed
depending on whether scipy was installed. Both now apply it.

## 3. Family suffix suppressed when there is only one family

My regression from v1.2.0. Running without a gRNA whitelist put every label at
`ABT1_unassigned_v1.0s2` and every control group at `NTC_unassigned`, where
v1.1.0 said `ABT1` and `NTC`. The suffix now appears only when two or more
families are present — which is the only time it carries information.

| families | label | target_key |
|---|---|---|
| one (or none declared) | `ABT1_v1.0s2` | `ABT1` |
| two or more | `ABT1_A_v1.0s2` | `ABT1_A` |

## 4. Plot readability

- **Per-target bar charts** (knockdown, E-distance) now scale figure height
  with the number of targets and shrink tick labels within bounds. A fixed
  height is legible at 20 targets and overprints at 80.
- **DEG dot plot** is capped at 120 genes and 30 perturbations. At 40
  perturbations × 10 DEGs the x-axis reached ~400 columns and the figure was
  metres wide. The title states what was capped, and the full tables are still
  written to CSV.

## 5. `condition_columns` in the sample manifest

```csv
sample,prefix,h5ad_path,output_path,fixation,buffer,condition_columns
s1,p1,data.h5ad,out,fresh,CSB,fixation|buffer
```

Nominates which comparisons matter. Precedence: `--conditions` → manifest
`condition_columns` → autodetection. `|`, `,` and `;` all separate. A name that
is not a column in the manifest **raises**, rather than being skipped — a typo
here would silently change which comparisons the report makes.

When more than three axes are available, declared columns keep the first three
*in the order given* (the author's priority) instead of the previous
fewest-levels heuristic, and the run logs what it dropped either way.

## 6 & 7. Small

- Ten CRISPR/Feature sequencing metrics added to `seqmetrics.METRIC_ALIASES` —
  they were among the 49 the MDL-1856 run silently dropped, and they are
  precisely the ones that say whether guide and hashtag capture worked.
- `n_cells_with_target` removed from the per-target table: it was identical to
  `n_cells_assigned` in every run.

---

## Verification

```
python tests/test_v121_changes.py     # 23 checks
python tests/test_v120_changes.py     # 57 checks
```

Both pass, as do the 45 runnable v1.1.0 tests.

## Still outstanding

Deferred deliberately until a clean report exists to build on: per-condition
perturbation comparisons, crossed condition axes, pseudobulk GEX/gRNA
comparability, fallback controls for families with no NTCs (needs
depth-matching — unassigned cells run at roughly half the depth of assigned
ones), and a warning when a family's control pool rests on a single guide.
