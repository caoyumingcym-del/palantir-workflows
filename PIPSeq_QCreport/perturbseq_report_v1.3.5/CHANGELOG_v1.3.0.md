# v1.3.0

Implements the 20 Aug review in full: the seven A-items and Part B (B1–B5, B8).

The A-items share a theme. Every one of them was a case of the code and its own
description disagreeing — a default that was never chosen, a setting wired to
nothing, a docstring claiming a check that did not exist. None of them were
subtle in retrospect and all of them read plausibly in review, which is why the
B8 sweep at the end matters more than any individual fix.

---

## A1/A4 — batch_key and harmony

**Harmony is off, behind an explicit flag, and refused when it would matter.**

`batch_correct` now defaults to `"none"`, and `"auto"` is an **alias for
`"none"`** so an existing config file cannot silently reacquire correction.
`--batch-correct harmony` turns it back on for a genuinely multi-batch,
non-confounded experiment.

Where harmony reached, restated because it is the substantive part: `rep`
becomes `X_pca_harmony`, `EmbeddingResult.pca` is that array, and the pipeline
passes it to `run_perturbation_stage`, where it drives `edistance_table`. So
E-distance — a quantitative effect size, not a visualisation — was computed in
corrected space. Fold changes, DEG counts, target knockdown and the resampling
test all read `X_log` and were never affected.

**Refusal is per-correction, not per-run.** When the resolved batch key is also
a declared condition column, the correction is skipped, the run continues, and
the report says `REFUSED: ...` with the reason. Integrating away your own
comparison should not be reachable by accident, but it also should not cost you
a long run.

**`pick_batch_key` now does what it said.** It promised "2+ levels and no single
level dominating" and implemented only the first half — `vc` was computed and
then only `len(vc)` was tested, so a 99%/1% split counted as batch structure.
The dominance limit is real (`batch_dominance_max = 0.90`) and the function
returns `(key, reasons)` so every candidate's accept/reject is stated rather
than implied.

**HVG selection is no longer batch-aware by accident.** The resolved batch key
used to go straight into `highly_variable_genes`. With `seurat_v3` that ranks
each gene's variance *within* each level and combines ranks — right when batch
is a nuisance, wrong when "batch" is `sample` and sample is the condition,
because it then de-prioritises exactly the genes that differ between
conditions. New `embedding.hvg_batch_key` (`--hvg-batch-key`), `None` by
default.

**The resolved key is always reported**, even when nothing consumes it. "No
batch-like column was found" and "one was found and deliberately not used" are
different facts about a run.

**The sample-coloured UMAP is now rendered unconditionally.** Several notes
tell the reader to go and check it. The panel list was `extra = group_columns`,
and `sample` only entered that as a fallback for single-condition experiments —
so in any multi-condition run, including MDL1898, the figure those notes pointed
at did not exist.

**E-distance states its space.** A new note says whether the PCA was corrected,
and that everything else in the section is computed on the log-normalised matrix
and unaffected by any embedding choice.

## A2 — covariate regression

`regress_out` defaults to `()`. `--regress-qc` opts back in to **depth and
%mito only**; cell-cycle regression is deliberately not offered.

Two reasons. Scope: the step only ever touched the HVG block used for the
embedding, never `X_log`, so it could not move a fold change — and it was the
longest step in the stage. And for cell cycle specifically it is actively wrong
here: many knockouts *are* proliferation phenotypes, so removing
`S_score`/`G2M_score` suppresses the perturbations being screened for. Scores
are still computed and shown in the phase panel; they are just not removed.

The v1.2.5 fix that attached the QC metrics to `obs` still matters — it is what
makes `--regress-qc` work at all when someone asks for it.

Every case now reports itself, including "nothing was regressed out" and
"covariates were configured but not present on obs". A caption claiming
otherwise is how v1.1.0–v1.2.4 hid the fact that depth and %mito were never
being removed.

## A3 — the z-score clip

Kept at 10, now explained: a convention from Seurat's
`ScaleData(scale.max = 10)` via the scanpy tutorial, embedding-only, and a round
number rather than a derived one. The report also states **how many entries
actually sat at the cap** — if that number is large, something upstream is wrong
and it should not be quietly absorbed. `scale_max_value = None` disables it.

## A5 — Leiden and performance

**Clustering itself was fine.** `flavor="igraph", n_iterations=2,
directed=False` is scanpy's current recommendation, `n_neighbors=15` is the
default, and Leiden is one of the *faster* steps.

**Cluster merging is off** (`min_cluster_frac = 0.0`). There was no method
behind "merge anything below 0.5% into its nearest cluster by PCA centroid
distance": Euclidean distance between centroids in 30 dimensions is a poor test
of whether two populations are the same thing, and 0.5% of 187k cells is ~935
cells — a real population, and in a screen quite possibly the interesting one.
Fragmentation is now counted and reported instead
(`small_cluster_report_frac = 0.01`). The merge function remains for anyone who
passes an explicit threshold.

**Performance, in order of payoff:**

1. The ungated doublet detector is gone (A6) — measured 1.5 s at 10k cells,
   6.0 s at 20k, 26.2 s at 40k, quadratic, extrapolating to **8–11 minutes at
   187k**. It was the largest avoidable cost in the pipeline and it was running
   with detection switched off.
2. **`sc.settings.n_jobs` is now assigned from `cfg.n_jobs`, defaulting to
   `-1`.** The field existed from v1.1.0 and nothing read it, so scanpy's
   `regress_out` and `neighbors` ran single-threaded regardless.
3. `regress_out = ()` removes the next-largest step outright.
4. `copy_input = False` by default — the largest single allocation, and the
   pipeline never reads the input object again. *Library callers who still need
   their AnnData should set it back to `True`: with `False` the stage modifies
   the object it was handed.*
5. `_step()` reports the **previous** step's duration. Announcing before running
   is what makes a crash diagnosable, but it meant no line could carry its own
   time, so "which step is longest?" was unanswerable from a log — including for
   me, which is why the figures above are extrapolated rather than measured.
6. `sc.tl.pca(svd_solver="randomized")`, with a fallback, for a wide dense block.

## A6 — off means off

The gate was missing on the scanpy path:

```python
if "predicted_doublet" not in obs.columns:      # <- no detect_doublets check
    score, call = _detect_doublets_fallback(pca, ecfg.random_state)
```

That condition is true **precisely when detection was switched off**, so turning
doublets off substituted the built-in synthetic-doublet detector for scrublet
rather than skipping the step — and unlike the scrublet branch it emitted no
note, so the report showed a doublet panel nobody asked for. The `_process_fallback`
path was correctly gated, which is why the test suite missed it: the tests run
without scanpy.

Now gated. With detection off the panel is registered as **skipped, with a
reason**, so its absence is explicit. Both paths share one note template, and
that note states that the fallback's threshold —
`max(quantile(scores, 0.90), expected × 1.5)` — pins the reported rate near
5–10% whatever the data look like, so it is a property of the rule rather than a
measurement. Measured across four input sizes it sat at 4.8% every time.

`remove_doublets` deleted (dead config).

## A7 — HTO normalisation

**The axis was right; the claim was wrong.** Per-feature normalisation is
Seurat's `margin = 2` and the correct choice for ADT/HTO. But the transform is
not Seurat's CLR, and both `hto.py` and `clr_by_feature` said it was. Seurat
computes `log1p(x / exp(sum(log1p(x[x>0])) / length(x)))` — division on the raw
scale, inside the log, with the geometric mean summing over nonzero entries
while dividing by the count of all of them. We compute
`log1p(x) − mean(log1p(x))`. Seurat's output is ≥ 0 with zeros mapping to
exactly 0; ours maps zeros to `−mu`.

Renaming the function to `clr_by_feature` and then restating the overclaim in
the comment beside it defeated the point of the rename. Fixed.

`stats.clr_true_seurat` implements the Seurat formula (including its
sum-over-nonzero / divide-by-all asymmetry, faithfully — the comparison should
be against what people actually run). `HTOConfig.normalisation` selects between
`mean_centred_log1p` (default, unchanged), `seurat_clr` and `compositional`.

**And the argument is now a number.** `compare_normalisations` repeats the whole
threshold fit under each transform and the report tabulates it. On synthetic
data the two per-feature transforms agree on call rate to within 0.33 percentage
points, confirming that all three are monotone in the raw count so only the
cut-off moves. The same table found something sharper: on a hashtag that never
fired, both per-feature transforms correctly call **0%** positive while the
compositional CLR calls **50%** — a concrete reason not to use it for
demultiplexing, measured rather than argued.

Unchanged and deliberate: per-tag positivity plus declared-combination
matching. `HTODemux` assumes one tag per cell in its classification step, which
makes it the wrong tool for combinatorial hashing.

---

# Part B

## B1 — depth-matched fallback controls

New `controls.py`. Where a family has no non-targeting guides (RPE1), its
control pool is built from guide-**unassigned** cells of the same family,
depth-matched.

The naive version of this is what makes it dangerous: unassigned cells run at
roughly half the depth of assigned ones (median 2,674 vs 5,446 UMIs), so
substituting them raw puts a systematic depth difference into every fold change
in the family, which reads as a global transcriptional effect and is not one.

Matching is stratified on quantile bins of `log1p(depth)` — quantile bins put
the resolution where the cells are, and log space because depth is roughly
log-normal. Two details that took a second pass to get right:

- **Common support.** A stratum the pool cannot supply is not a sampling
  shortfall, it is a depth range where no control exists. Taking "as many as
  available" from every bin — the obvious implementation, and my first one —
  silently reshapes the pool towards wherever controls are plentiful, i.e.
  towards shallow cells, which is the exact bias the function exists to remove.
  Those target cells are excluded and the exclusion is reported.
- **One common ratio across strata**, set by the scarcest stratum, so the
  target's depth distribution is reproduced rather than approximated. On the
  real depth profile this moves the median gap from 2,696 UMIs to 5 — a ratio of
  1.001 — where the first implementation reached only 0.78.

Sampling is without replacement: a pool containing the same cell twice would
understate its own variance, and estimating the spread of unperturbed expression
is the entire job.

Every affected row carries `control_is_fallback`, and the report states the
achieved match, the fraction of target cells excluded for want of a
depth-matched control, and the caveat matching cannot fix — unassigned cells are
not verified unperturbed, since some carry a guide that was simply not called.
Where no usable pool exists, the family is skipped rather than compared against
an unmatched one.

### B1a — cross-family control borrowing was happening silently

Found while verifying that depth matching only applies in fallback cases. It
does — but the fallback was unreachable on the very data it was built for.

`target_annotations` did not receive the config, so it **inferred** pooling:

```python
if len(ntc_keys) == 1:        # "pool_ntc_across_families=True", said the comment
    for fam in set(family_by_key.values()):
        ntc_key_by_family.setdefault(fam, only)
```

A single NTC key does mean pooling was requested when
`pool_ntc_across_families=True`, because `guide.target_key` collapses every
control to `cfg.ntc_label`. But it is **also** true when pooling was not
requested and only one family happens to carry controls — which is exactly the
mixed experiment where one library has NTCs and another does not. In that case
every family without its own controls silently borrowed the family that had
them, `n_ntc_by_family` came out non-zero, and the B1 fallback was skipped.

So on MDL1898, RPE1 guides would have been compared against the other cell
line's NTCs, with `control_is_fallback = False` and nothing in the report saying
so. That is the family-scoping guarantee from v1.2.0 failing in the one case it
most needed to hold.

`pool_across_families` is now an explicit parameter threaded from
`cfg.guide.pool_ntc_across_families` through all five call sites. With it off
(the default), a family with no controls either gets a depth-matched fallback
pool or is excluded with a reason — never another family's controls.

Verified three ways in `test_fallback_is_only_used_when_needed` and
`test_no_silent_cross_family_borrowing`: a normal dataset is **byte-identical**
with and without the fallback machinery supplied; a mixed dataset fires the
fallback for the deficient family only; and with the machinery withheld the
deficient family is excluded rather than borrowed.

## B2/B4 — control-pool composition and consistency

`summarise_control_pool` reports per-family pool composition and flags
`single_guide` (family C: 1 guide, 974 cells) — every fold change in such a
family is measured against one reagent, so that guide's off-target profile is
indistinguishable from the biology being compared to it. Also flags
`concentrated`: three guides where one supplies 80%+ of the cells behaves closer
to a single-guide pool than the count suggests.

`leave_one_out_consistency` answers the question as it was actually asked — *are
these controls interchangeable?* Each control guide is held out, the pool mean
recomputed from a running total, and the largest shift reported. A guide that
moves the pool is contributing something of its own, and pooling flattens that
into the baseline every perturbation in the family is measured against.

## B3 — pseudobulk comparability

New `pseudobulk.py` and a new report section, **Condition comparability**, which
sits before the cross-checks and asks whether the conditions are comparable at
all before any per-cell difference between them is believed.

Two profiles per condition level: mean log expression per gene (Spearman across
levels — "do these rank genes the same way?", and not dragged around by a handful
of high-expression genes), and gRNA composition as a percentage of cells.

The distinction is the point: a transcriptome difference between conditions may
be real biology, but **the library was pooled once, so a guide-composition
difference almost never is**. It means cells were lost non-randomly —
differential dropout, a failed sort, uneven recovery — and every per-guide effect
size in that arm inherits the bias.

## B5 — per-condition perturbation comparisons

`per_condition_knockdown` recomputes target-gene knockdown *within* each
condition level, with **controls drawn from the same level as well as the same
family**. Comparing one condition's perturbed cells against controls pooled
across all conditions would fold the condition effect into every knockdown
estimate, which is the mistake the function exists to avoid.

`condition_effect_spread` reports `range_pp`, the gap between a target's best
and worst arm. That is the number that says whether pooling was hiding
something: 80% in one arm and 5% in the other pools to ~45%, a figure that
describes neither.

## B8 — dead-config and docstring sweep

Done as a test rather than a one-off pass: `test_no_dead_config` walks every
field of every config dataclass and fails if nothing in the package references
it. It found **ten**:

| Field | Action |
|---|---|
| `PerturbConfig.similarity_method` | deleted — only Spearman over the fixed union DEG set is implemented, and that is a methodological choice, not a knob |
| `EmbeddingConfig.marker_method` | deleted — only the Mann-Whitney path exists |
| `EmbeddingConfig.remove_doublets` | deleted (A6) |
| `ModalityConfig.hto_call_col_candidates` | deleted |
| `ModalityConfig.cell_type_col_candidates` | deleted |
| `ModalityConfig.hemo_prefixes` | deleted — plausible feature, never implemented |
| `GuideConfig.family_max_len` | deleted |
| `HTOConfig.require_whitelist_for_demux` | deleted |
| `FigureConfig.max_width_px` | deleted |
| `ReportConfig.use_web_fonts` | deleted — actively contradicts the no-network-resources design |
| `ReportConfig.include_method_notes` | deleted |

`n_jobs` was the eleventh and is now wired. `use_checkpoints` was the twelfth
and was fixed in v1.2.5. The two settings that looked most like statistical
knobs — `similarity_method` and `marker_method` — were the two most misleading,
since both named a method that could not be changed.

Docstring claims corrected in the same spirit: `pick_batch_key`'s dominance
promise (now implemented) and `clr_by_feature`'s Seurat claim (now accurate).

**Still latent, noted not fixed:** `_coerce` in `config.py` tests
`target_type in (int, float, str, bool)`, but with `from __future__ import
annotations` every `f.type` is a *string*, so those branches never fire and the
function is close to a no-op. Harmless today because the CLI passes correctly
typed values, but a JSON config supplying `["a","b"]` for a `tuple[str, ...]`
field would not be converted. Added to `PENDING_WORK.md`.

---

## Verification

- `tests/test_v130_changes.py` — **new, 135 checks** across nineteen groups
- `tests/test_units.py` — 51 passed
- `tests/test_v120_changes.py` — all passed
- `tests/test_v121_changes.py` — all passed, with
  `test_regress_out_covariates_are_present` rewritten for the inverted default
- `tests/test_end_to_end.py` — **25/25 passed**
- Two real pipeline runs in one output directory: the new sections render, the
  embedding cache still hits on the second run, and the doublet panel is
  registered as skipped rather than populated

Several v1.3.0 checks are deliberately *source* assertions rather than
behavioural ones. Every bug in this release had the same shape — a setting wired
to nothing, or a docstring claiming a check that did not exist — and no
behavioural test can catch "nobody calls this".

**One test cannot run here and must run in ICA:** the doublet gate on the scanpy
path. This environment has no scanpy, so it exercises `_process_fallback`, which
was already correctly gated — that is exactly why the bug survived three
versions. Please run `python tests/test_v130_changes.py` once in the ICA
environment.

## What will look different in your next report

- Clusters, UMAP and markers will change: no harmony, no covariate regression,
  no cluster merging. Expect **more clusters**, some of them small, listed
  rather than absorbed.
- **E-distance values will change** — they were harmony-corrected. Nothing else
  in the perturbation section moves for families that have their own controls.
- **RPE1 (and any family without NTCs) will change substantively.** Previously
  it was silently compared against another family's controls; now it gets a
  depth-matched fallback pool built from its own unassigned cells, flagged as
  such on every row, or is excluded with a reason. Do not compare those numbers
  against the v1.2.x run.
- No doublet panel; a skip note explaining why instead.
- New: Condition comparability section, per-condition knockdown tables, control
  pool composition, control consistency, normalisation comparison, a
  sample-coloured UMAP panel, and per-step timings in the log.
- **Nothing to delete.** `CACHE_VERSION` is bumped to 3, so any checkpoint from
  a v1.2.5 run misses rather than misleads. The key also covers `batch_correct`
  and `regress_out`, both of which changed, so it would have missed regardless —
  the version bump just makes that guaranteed rather than incidental.
