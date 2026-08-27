# v1.3.1 — figure legibility, and the bugs behind it

## plotting.py

* Typography derives from one base through named roles (`fs(cfg, role)`), with
  an 11 pt floor. Every size in this module used to be an absolute number
  written inline (5.5, 6, 6.5, 7, 8), so raising `FigureConfig.font_size` did
  nothing to the labels that were actually too small to read. Titles, axis
  labels and ticks are bold via `rcParams`. Lifts every figure in the report.
* `heatmap(mask_diagonal=True)` removes self-comparison cells from both the
  colour scaling and the annotation. Including the diagonal pinned `vmax` at
  1.0 and pushed a healthy 0.977-0.993 correlation spread into the black end of
  viridis.
* `heatmap` annotation colour is chosen per cell from the colour underneath it
  (WCAG relative-luminance crossover) instead of a fixed `#222` that was
  invisible on dark cells.
* `heatmap(robust=True)` scales to the 2nd/98th percentile so one outlier cell
  cannot flatten the rest. `cbar_fmt` controls colourbar tick formatting.
* `tick_layout()` sets long categorical labels at 40 degrees with right
  alignment instead of a flat 90; `shorten()` middle-elides over-long ones.
* `triangular_similarity` scales its Jaccard triangle to the observed spread
  (`upper_vmax`) instead of a fixed 0..1, which rendered the whole triangle one
  shade of pale yellow.
* `annotate_empty` takes an optional `cfg` so its message scales too.

## pseudobulk.py

* `plot_comparability` rebuilt: masked diagonal with the colour range stated in
  the title, per-cell annotation contrast, shared level order across both
  heatmaps, 20 guides instead of 40, value-labelled drift bars, an
  aspect-locked scatter with a labelled `y = x` line, and a figure size derived
  from the number of levels and the longest label.
* Both heatmaps now use one sorted level order. `pseudobulk_by_group` used
  first-appearance order while `gRNA_composition` got sorted order from
  `pd.crosstab`, so the same conditions were listed differently in panels
  directly above one another.
* `astype(str)` no longer creates a phantom `"nan"` condition level. In
  `gRNA_composition` the guard was `v.notna()` *after* `astype(str)`, which is
  always True.
* Axes with fewer than two levels are registered as skipped instead of dropped
  with a bare `continue`. A guide/annotation length mismatch is reported as a
  warning instead of silently emptying half the figure.
* The drift table is written to `grna_drift_<axis>.csv`; it was the only data in
  this figure with no CSV beside it.
* `min_cells` is read from config rather than left at the function default.
* Registry keys are derived once, before anything can fail, via a
  collision-proof `_safe_key`. The failure path used the raw axis name and the
  success path the sanitised one; the sanitiser was also lossy, so two axes
  could collide into a duplicate-key `ValueError` reported to the reader as
  "could not be computed".
* `comparability_findings()` returns `(level, text)`. Severity used to be
  recovered by grepping the rendered sentence, and one of the three patterns
  contained a newline the message never had, so that branch never fired.
  `comparability_notes()` still returns text only.

## guide.py

* The efficiency table is titled `"... (table)"`. `report.py` emits an `<h3>`
  from the figure path and again from the table path, so a figure and table
  sharing one title printed the same heading twice — visible in the rendered
  report for all four condition axes.
* Efficiency figures/tables get `order=50 + i` / `55 + i`. A shared `order=50`
  made the sort fall back to its alphabetical key tiebreak, so these panels ran
  in a different axis order from the purity panels above them.
* Removed a second, identical write of `guide_target_mapping.csv`.

Full test suite passes: 51 unit, 25 end-to-end, plus the v1.2.0/v1.2.1/v1.3.0
suites.

---

# Round 2 — multiple-testing corrections

These change reported numbers, not just their presentation.

## stats.py — untested genes no longer enter the BH family

`mannwhitney_u_sparse_columns` initialised `pv = np.ones(k)` and `continue`d on
all-zero columns, so a gene detected in no cell emitted **p = 1.0**.
`benjamini_hochberg` excludes NaN from the ranking but counts 1.0 — its own
docstring names this as the thing it exists to avoid — so every other gene's
adjusted p was inflated by the ratio of screened to testable genes. The dense
path did this correctly via its `informative` pre-filter, so the two paths
disagreed. Both now return NaN, for all-zero columns and for zero-variance
columns alike: untestable is not the same as not significant.

Measured effect: on a matrix with 40% of genes detected in no cell, adjusted
p-values were inflated by up to 1.667× (= n_screened / n_testable). For the
reference object cited in `stats.py` — 11,504 of 38,402 genes detected in no
cell — the factor is 1.43×. The direction is conservative, so the practical
consequence was **lost** DEGs: `select_degs` thresholds on `padj < de_padj_max`,
so this propagated into per-perturbation DEG counts, the "Perturbations with
≥1 DEG" metric, and the DEG sets feeding the Jaccard similarity matrix.

Note that `test_de_sparse_equals_dense` could not catch this: it compared the
paths with `np.nanmax`, so the single column where they disagreed was masked by
the very NaN that marked the disagreement. `tests/test_v131_changes.py` now
compares NaN patterns explicitly.

## gex.py — marker padj is corrected over the family actually screened

`compute_markers` pre-ranked genes by fold change, tested only the top
`max(n_genes * 6, 60)`, then applied BH with that subset as *n*. Two compounding
errors, both pushing padj down: the denominator was the selected subset rather
than the screened family (up to `max_genes_tested`, i.e. understated by up to
~67×), and the subset had been chosen using the same data as the test.

Every screened gene is now tested and BH is applied across all of them. Only
enriched genes are returned, but that selection happens after the correction,
which does not affect it. Added `_marker_pvalues`, which blocks over columns and
routes through the vectorised ranking in `stats.py`.

Cost: essentially nothing on sparse data. Measured at this experiment's scale
(187,503 cells, 7 clusters, 6% density), the sparse column path runs at 0.54 ms
per gene against 28 ms for the per-gene loop it replaces — 52× faster per gene —
so testing 4,000 genes instead of 60 moves the marker step from about 11.8 s to
15.2 s. On a *dense* input the same change costs ~10× more, because dense inputs
take the slower branch; `max_genes_tested` is the dial if that matters, and it
now honestly defines the correction family rather than hiding a bias.

## guide.py — two metrics that did not mean what they said

* `pct_above_min_reads` was computed as `guide_total_umis > 0`, i.e. the
  percentage of cells with *any* guide UMI, under a name promising the
  assignment threshold. Both quantities are now emitted under accurate names,
  with the threshold threaded from `cfg.guide.min_reads`.
  **This changes the schema of `guide_efficiency_by_*.csv`:** the new column is
  `pct_with_any_guide_umi`, and `pct_above_min_reads` now means what it says.
* "Distinct targets assigned" counted the non-targeting pool as a target, so
  the Summary metric was high by one.

Added `tests/test_v131_changes.py` (19 checks). Full suite after these changes:
51 unit, 25 end-to-end, plus the v1.2.0 / v1.2.1 / v1.3.0 / v1.3.1 suites — no
failures.
