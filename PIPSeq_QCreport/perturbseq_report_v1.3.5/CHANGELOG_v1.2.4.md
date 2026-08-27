# perturbseq_report v1.2.2

Fixes found by running v1.2.1 on MDL1898 (`all_samples_GEX_gRNA.h5ad`), plus
the downsampling comparison. Four of the five are my bugs from v1.2.1.

## 1. `condition = fixation` was ignored  (my bug)

The manifest column is `condition`; v1.2.1 looked only for `condition_columns`.
So the nomination was invisible, autodetection found four qualifying columns,
the cap kept three, and — every column having exactly 2 levels — the
fewest-levels tie-break dropped `fixation`, the one column the author had named.

Now: `condition`, `conditions`, `condition_column`, `compare_columns` and
`priority_conditions` are all accepted. `condition` is ambiguous (it could hold
values like `untreated`/`TGFb`), so it counts as nominating only when its
values name real columns of the manifest; otherwise it stays ordinary metadata.
The canonical `condition_columns` still raises on an unknown name, since a typo
there can only be a typo.

## 2. Nominating a column no longer discards the others  (my bug)

v1.2.1 treated nomination as *restrictive*: declaring `condition = fixation`
would have analysed fixation and thrown away gRNA_method, acoh and
resuspension_buffer. You asked for "prioritise". Nominated columns now come
first and autodetected ones fill the remaining slots, so the named comparison
can never be the one dropped.

The axis cap is also configurable (`max_condition_axes`) and defaults to **4**,
because a 4-factor design needs 4 and silently dropping one is worse than a
longer report.

## 3. "no matching cell_input in manifest"  (my bug)

The end-to-end yield panel compared the retention summary's groups against
`cell_input`. But the retention summary is grouped by the *first condition
axis* (`CSU`/`IVT`) while `cell_input` is indexed by *sample*
(`MDL1898_1`…) — so nothing ever matched, and the panel reported a missing
`cell_input` on a manifest that had one for every row.

The yield panel now gets its own sample-grouped summary. When it genuinely
cannot match, it prints both sets of names instead of implying the manifest is
at fault.

## 4. Scattered numbers on the sequencing-metrics panel  (my bug)

Two compounding faults:

- `ax.bar(labels, ...)` treated the prefixes as categories, so **duplicate
  prefixes** — legitimate when a CSU and an IVT library share a lane, as four
  of yours do — collapsed into one bar while the value labels were still placed
  at integer index `i`. Labels ended up floating away from any bar. Now plotted
  at explicit numeric positions, with the sample name added to the tick when
  prefixes repeat.
- Rate metrics arrive from DRAGEN as fractions (`pct_reads_in_cells = 0.607`)
  and every value was formatted `"{:,.0f}"`, printing "1" above a bar of height
  0.607 and "0" above one of 0.42. Rates are now converted to percent and shown
  to one decimal; counts keep a thousands separator. Bars also get headroom so
  labels are not clipped.

## 5. Downsampling: before vs after

The DRAGEN section describes the upstream run **before** downsampling, and said
nothing about it — inviting the reader to attribute that depth to the object
being analysed. It is now labelled "(upstream, pre-downsampling)" in the panel
title, the table title and the caption.

New **"Depth before vs after downsampling"** panel and table: per-sample mean
GEX and guide depth measured from the h5ad itself, beside the DRAGEN figures.

One thing deliberately not done: the two are **not** divided to produce a
"downsampling factor". DRAGEN counts reads; an h5ad holds deduplicated UMIs;
downsampling reads reduces UMIs sub-linearly because of saturation. The ratio
would look authoritative and mean little. What is directly interpretable is the
**spread across samples**, quoted on every panel as a CV: downsampling to a
common depth should collapse it. A CV that is unchanged means the downsampling
did not take effect on this object — which is the failure worth catching.

## Also

`_parse_nominating` built its error message by calling `metadata_columns()`,
which calls `nominating_column()`, which calls back into it — infinite
recursion on a manifest with a typo'd `condition_columns`. Caught by the test
suite before shipping.

## Verification

```
python tests/test_v121_changes.py     # 35 checks
python tests/test_v120_changes.py     # 57 checks
```

Both pass, as do the 45 runnable v1.1.0 tests.

## 6. Doublet detection is now opt-in

It was never new code. Any input carrying `obsm['X_pca']` takes the
`_reuse_embedding()` path, which returns early and only *adopts* doublet
annotations already present in `obs`. `sc.pp.scrublet` is called solely from
`_process_with_scanpy()`. MDL1856 had a precomputed PCA and no doublet columns,
so nothing ever ran and nothing ever appeared in a report. MDL1898 was the
first object without a precomputed embedding — the first time scrublet had ever
executed — and it killed the run.

`detect_doublets` now defaults to **False**. `--doublets` opts in;
`--no-doublets` still works so existing commands are unaffected. Doublets were
never dropped in either case, so nothing downstream changes.

Three reasons for the default, in order of weight:

1. Scrublet identifies doublets as cells resembling mixtures of
   transcriptionally distinct types. In a homogeneous line there are no
   distinct types to mix, so the scores are not informative — which matches
   what you found independently.
2. With `batch_key` it runs per batch, simulating doublets and building a kNN
   graph for each. On 187k cells across 8 batches it exhausted memory; the run
   was already at 15.7 GB on entry.
3. It was effectively untested, per the above.

Two fixes so it is correct if switched on:

- **It ran on log-transformed data.** Scrublet expects raw counts and
  normalises internally, so it was double-transforming. The block now runs
  before `normalize_total`, where `X` is still counts. This was also the source
  of the repeated "adata.X seems to be already log-transformed" warnings — one
  per batch, from scrublet re-normalising each batch.
- **No step label.** Cell-cycle scoring and doublet detection printed nothing,
  so the log went silent exactly where it died and appeared to stop at log1p.
  Both are now labelled. Second time an unlabelled step has cost a diagnostic
  round.

---

# v1.2.4 — crash fix

## `IndexError: index N out-of-bounds in add.reduceat`

The sparse DE path added in v1.2.1 computed per-column sums with
`np.add.reduceat(data, indptr[:-1])`. When a block's **last column is empty**,
`indptr[k-1] == len(data)`, which reduceat rejects. The run died at the
perturbation stage after clustering, guide assignment and everything else had
already completed.

My tests used uniformly dense synthetic blocks, so no column was ever empty and
this never fired — despite **11,504 of the object's 38,402 genes being detected
in no cell**, which made such a block close to inevitable. The dead code that
was supposed to handle empty columns sat *after* the reduceat, so it never ran.

Replaced with `np.bincount` over repeated column indices: an empty column
contributes nothing and lands at zero, with no edge case. Verified identical to
the dense path (max |diff| 2.8e-16) with the empty column first, last, in a
trailing run, scattered, and with every column empty — all five now asserted in
the test suite.
