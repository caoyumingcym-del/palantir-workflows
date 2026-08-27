# Pending work and open decisions

Current shipped build: **v1.3.2**.

The 20 Aug review (A1–A7) and Part B items B1–B5 and B8 are **implemented** —
see `CHANGELOG_v1.3.0.md` for what each one turned out to be. v1.3.1 and
v1.3.2 changes are in their own changelogs. This file now tracks only what is
left.

---

## Open, ranked by what would change a conclusion

### 1. Batch-mixing statistic (was A1 item 7)

Harmony is gone and the sample-coloured UMAP is now rendered unconditionally, so
batch structure is visible. It is not yet *measured*.

The argument for measuring it rather than correcting it: a QC report exists to
tell you whether your samples are comparable, and a pipeline that can silently
make them comparable cannot also tell you whether they were. So the replacement
for harmony is a number — per-cluster sample composition against expected, or a
LISI-style local mixing score, reported and thresholded.

Deferred rather than dropped because the threshold needs calibrating against a
dataset where you already know the answer. The pseudobulk comparability section
(B3, shipped) covers the coarse version of this question; this is the per-cell
version.

### 2. MDL1856 `var_names` / `X` column permutation

Unrepaired input defect, deferred by you. The v1.2.1 sanity gate now refuses the
object rather than producing a third confident wrong report, so this is safe to
leave — but the dataset remains unusable until it is fixed upstream.

### 3. Hashtag ambiguity rate on real data

You flagged an unusually high ambiguous rate and it has not been chased to a
root cause. v1.3.0 adds the tool for it: the normalisation-comparison table
shows whether the call rate is being decided by the transform or by the data. If
the spread across transforms is small, the ambiguity is real and the next step is
a caller comparison (below).

### 4. Alternative hashtag callers

Worth evaluating against the current per-tag threshold rule on a dataset where
the design is known: **deMULTIplex2** (negative-binomial GLM, models ambient
contamination explicitly), **GMM-Demux**, **demuxEM**, **HashSolo**. `cellhashR`
wraps several for comparison.

Not a recommendation to switch — the current structure (per-tag positivity plus
declared-combination matching) is correct for combinatorial designs in a way
`HTODemux` is not. This is about whether a better background model reduces the
ambiguous fraction.

### 5. Crossed condition axes

B5 shipped per-condition comparisons along each axis independently. Crossed axes
(fixation × gRNA method, say) are not handled — a target that works only in one
*cell* of the cross will not show up as a large `range_pp` on either axis alone.
Needs a decision about how many cells are enough per crossed level before it is
worth reporting.

### 6. DE-based cluster merging (optional)

Merging is off by default now, which is the right default. If the small-cluster
count turns out to be annoying in practice, the defensible version is a DE-based
criterion — merge two adjacent clusters only if fewer than *N* genes are
differentially expressed between them — rather than the centroid-distance rule
that was removed. The fast Wilcoxon path from v1.2.4 makes this cheap. Only
worth building if the reports actually become cluttered.

### 7. PCA variance-ratio panel

`n_pcs = 30` is never justified from the data because there is no elbow plot.
Small addition; makes an invisible choice visible.

---

## Known latent issues, noted not fixed

### `_coerce` is close to a no-op

`config.py`'s `_coerce` tests `target_type in (int, float, str, bool)` and
`origin in (tuple, list)`. With `from __future__ import annotations`, every
`f.type` is a **string** like `"tuple[str, ...]"`, so none of those branches ever
fire and the function returns its input unchanged.

Harmless today: the CLI passes correctly typed values, and the dataclasses are
never validated against their annotations anyway. It matters the first time
someone writes a JSON/YAML config — a list supplied for a `tuple[str, ...]`
field stays a list, and a string `"30"` for an `int` field stays a string.

Fix is `typing.get_type_hints()` on the dataclass, or matching on the annotation
string. Low priority, non-zero risk of surprising someone.

### The doublet-gate test cannot run outside ICA

`test_v130_changes.py` asserts the gate is present in source, which catches
regression. It cannot assert the *behaviour* — that `predicted_doublet` is
absent after a scanpy-path run with `detect_doublets=False` — because this
environment has no scanpy and therefore exercises `_process_fallback`, which was
always correctly gated. That asymmetry is exactly why the bug survived three
versions.

**Please run `python tests/test_v130_changes.py` once in the ICA environment.**

---

## The pattern worth remembering

Every A-item in v1.3.0 was the same failure: code and its own description
disagreeing.

- `use_checkpoints` — a directory created and never written (fixed v1.2.5)
- `n_jobs` — a field never assigned to `sc.settings.n_jobs`
- `remove_doublets` — declared, read nowhere
- `similarity_method`, `marker_method` — named a method that could not be changed
- seven more config fields wired to nothing
- `pick_batch_key` — a dominance check promised in the docstring, half
  implemented in the code
- `clr_by_feature` — renamed to stop overclaiming, then the overclaim restated
  in the comment beside it
- the doublet gate — a condition that read as "don't redo work" and actually
  meant "run this whenever it was switched off"
- `regress_out` — a caption asserting four covariates where two were applied
- the "check the sample-coloured UMAP" note — pointing at a figure that was
  never rendered in a multi-condition run

None were subtle in retrospect and all read plausibly in review. `test_no_dead_config`
now fails the build on the first category, which is the only one of these that
can be caught mechanically. The rest need someone to ask "does the code do what
this sentence says?" — so it is worth doing that pass deliberately after any
release that adds config or captions, rather than finding them one at a time.
