# v1.3.4 — a metadata flag that outranked the data, and two places that forgot the whitelist changes hto_class

## provenance.py — `detect_x_state` trusted `uns['log1p']` over the data itself

Real run, MDL-1856 (`qc_report (8).html`, visible only after the v1.3.3
report.py appendix fix): the newly-restored "How the input state was
determined" note read

    sampled 20,000 non-zero values: min 1, max 94, all integers;
    uns['log1p'] is present, which scanpy writes after log1p
    -> Input object: X holds normalised and log-transformed

20,000 sampled non-zero values, every one an exact integer, is unambiguous:
real `log1p` output on scRNA-seq data is continuous and does not land on exact
integers at this sample size. `X` is raw (or otherwise untransformed) count
data. But `uns['log1p']` was also present — stale, almost certainly left over
from an earlier processing pass whose output was later replaced with raw
counts, an upstream data-hygiene problem this pipeline can detect but not fix.
The previous version of `detect_x_state` checked the flag as the "strongest"
signal and returned `LOG1P` the moment it saw it, without re-confirming
against integrality. `gex.py`'s `_reuse_embedding` — already correctly
handling three `x_state` cases since the earlier NORMALISED/LOG1P fix — then
took the "already logged, leave X alone" branch on data that was actually raw
counts. Every consumer that un-logs with `expm1()` (`percent_knockdown`,
`differential_expression`, `perturbation_score`) exploded raw integer counts
(`expm1(94)` ~ 10^40) into the same absurd 1e13-1e23 mean-expression values
seen before — a different root cause than the earlier fix, identical visible
symptom, which is why it looked like a regression of already-fixed code when
it wasn't: `_reuse_embedding` itself was never touched.

**Why this surfaced now and not in the previous report:** the v1.3.3 fix to
`_load_h5ad_skipping_uns` (dropping only the individual `uns` keys that fail
to decode, instead of the whole group) was necessary to stop guide/hashtag
names from being lost. But it also means the stale `uns['log1p']` flag now
survives the load, where it previously would have been wiped out along with
everything else in `uns` — which had been *accidentally* giving the right
`x_state` answer for the wrong reason, while simultaneously breaking guide
naming. Fixing one exposed the other; neither `detect_x_state` nor
`_reuse_embedding` needed to be touched to introduce this, and neither was
reverted.

Fixed: integrality is now checked first and is authoritative. All sampled
non-zero values being exact integers means raw counts, regardless of what
`uns['log1p']` claims. When the two disagree, the disagreement is recorded
explicitly as a `CONTRADICTION` in the evidence list (visible in the "How the
input state was determined" note) rather than being silently resolved in the
flag's favour. The flag is now only consulted to help classify non-integer
data (`NORMALISED` vs `LOG1P`), which is the case it can actually speak to.

Verified with four synthetic cases directly against `detect_x_state` (stale
flag + raw integers -> `RAW_COUNTS` with contradiction noted; raw integers, no
flag -> `RAW_COUNTS`; genuine small-magnitude non-integer data + flag ->
`LOG1P`; large-magnitude non-integer data, no flag -> `NORMALISED`). No real
h5ad available in this sandbox to test end-to-end — **please confirm against
the actual MDL-1856 file in ICA** that this now reports `RAW_COUNTS` (or
whatever the true state turns out to be) and that the knockdown table's
`mean_control`/`mean_perturbed` columns land in a physically plausible range.

## hto.py — three places still keyed on `"Singlet"` after `_apply_design` overwrites it

`_apply_design` replaces `per_cell["hto_class"]` with `RESOLVED`/`AMBIGUOUS`/
`NEGATIVE` whenever a hashtag whitelist is declared, discarding the
count-based `"Singlet"`/`"Multiplet"` labels entirely (contrary to this
function's own docstring, which claims the two are "reported side by side,
never merged"). `run_hto_stage`'s Summary-metric code and the pipeline log
already correctly branch on `calls.design_declared` to avoid reading the
now-absent labels; three other places did not:

* `efficiency_by_group()` (feeds "Hashtag performance by <condition>")
  unconditionally read `vc.get("Singlet", 0)` / `vc.get("Multiplet", 0)`,
  which are always 0 once a whitelist is declared. Now emits
  `pct_resolved`/`pct_ambiguous` when `design_declared`, `pct_singlet`/
  `pct_multiplet` otherwise.
* `plot_efficiency()` unconditionally plotted the `pct_singlet`/
  `pct_multiplet`/`pct_negative` columns, which would KeyError or silently
  plot zeros against the new column set. Now picks whichever pair
  `efficiency_by_group` actually produced.
* `plot_composition()` (the "singlet composition by <condition>" panel)
  filtered `hto_class == "Singlet"`, always empty under a declared design, so
  it always rendered "no singlets to compose" regardless of the actual
  resolved rate. Now filters on `RESOLVED` when a design is declared. Also
  switched the composition category itself from `hto_call` to `hto_demux_id`
  in that case: `hto_call` is the naive per-tag label (a hashtag name for
  exactly one positive tag, else `"Multiplet"`), so for a Resolved cell
  carrying a declared two-or-more-tag combination it was always
  `"Multiplet"` -- composition by resolved *sample* is what this panel is
  actually asking about once a design is declared.

**Why this didn't show up in the MDL-1856 reports reviewed so far:** the
sample manifest's `condition` column is blank, so no condition/group axis was
ever resolved for this experiment (the report's own "Condition comparability -
not applicable: No condition axes were resolved" note), and the per-condition
loop that calls all three of these functions never ran. This is a latent bug,
not one visible in the reports so far -- it will fire the moment a run has
both a hashtag whitelist and a resolved condition column, an entirely
ordinary combination.

Verified `efficiency_by_group` directly against synthetic `design_declared`
True/False cases (checked the correct column pair appears and is non-zero in
each). `plot_efficiency`/`plot_composition` were checked by inspection and by
parsing (no matplotlib in this sandbox to render the actual figures) --
**please spot-check the "Hashtag performance by <condition>" and "singlet/
resolved composition by <condition>" panels on a run that has both a
hashtag whitelist and a condition column.**

---

**What will look different in your next report:** if MDL-1856's `X` is
genuinely raw counts (which the evidence strongly indicates), the knockdown
table's `mean_control`/`mean_perturbed` values should drop from the
1e13-1e23 range to physically plausible values (single to low-thousands), and
`pct_knockdown`/`log2fc` will change accordingly -- treat the current
knockdown table as unreliable until re-run. Any run with both a condition
column and a hashtag whitelist will show real `pct_resolved`/`pct_ambiguous`
numbers in "Hashtag performance by <condition>" instead of all-zero
`pct_singlet`/`pct_multiplet`, and the composition panel will show actual
sample composition instead of "no singlets to compose".
