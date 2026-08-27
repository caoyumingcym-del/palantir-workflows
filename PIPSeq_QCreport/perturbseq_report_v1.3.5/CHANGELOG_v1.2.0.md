# perturbseq_report v1.2.0

Six changes, five of them fixes to bugs found by running v1.1.0 on
`MDL1856_analyzed_full_depth.h5ad` (339,063 cells × 38,613 vars, 321 guides,
8 hashtags). Every number below is measured, not estimated.

v1.1.0 is untouched in the parent directory. This is a sibling copy.

---

## 1. Guide-ID parsing — 168 of 321 guides were unparsed

**Cause.** None of the five patterns in `GuideConfig.guide_id_regexes` could
match the library's dominant naming scheme. The closest required the Ensembl ID
as the second underscore-delimited token; another excluded `_` from the gene
character class.

**Consequence, all silent.** Each unparsed guide became its own single-guide
"target" (186 target groups instead of 77). `perturb.resolve_gene()` looks up
the symbol then the ENSG — but the raw guide ID matched nothing and
`target_ensg` was `None` for every guide in the library, so the Ensembl ID
*present inside the guide ID* was never used. The report's "84 targets >50%
knockdown / 42% median" was computed on the ~55 guides that happened to parse.

**Fix.** Two patterns restored from the previous-generation script, ahead of
the existing five. Two details are load-bearing and both were wrong in my first
attempt:

- `_spacer_target_+` — one or *more* underscores. Control IDs carry a doubled
  underscore (`..._spacer_target__INTERGENIC_CONTROL_...`) and a single `_`
  misses every one.
- The control-token alternative must allow underscores on both sides of
  `INTERGENIC`. `ONE_INTERGENIC_SITE` is a real token, and a class of
  `[A-Z0-9]*` before it silently dropped all 26 such guides into a looser
  pattern that discarded the target field.

## 2. Two names per guide, and which one wins

Many IDs carry both a gene-symbol prefix and a `spacer_target` field, and they
disagree exactly where it matters:

```
ABT1_target_version_1.0_..._spacer_target_ENSG00000146109_...      agree
ABT1_target_version_1.1_..._spacer_target_ONE_INTERGENIC_SITE_...  DISAGREE
```

The `.1` versions are per-gene **matched intergenic controls**: same backbone,
same prefix, spacer redirected. New `parse_target_from` column (per guide,
default `second`) selects the authoritative name. Defaulting to the spacer
target classifies matched controls correctly with no special case, and records
`control_of = ABT1` so the pairing survives.

NTC detection is now **structured-first**: `ntc_regex` runs only when no
structured target field was captured. v1.1.0 ran it over the whole ID
unconditionally, which was right on this library by luck and would misclassify
a gene named `CTRL1`.

## 3. Guide features were contaminating the transcriptome

**Not previously reported, and the most serious of the five.**

In the v1.1.0 report, clusters 0, 1, 2, 3 and 8 list guide IDs as their marker
genes. `modalities.split_modalities()` only populates `guide_mask` in the
feature-types branch and the guide-ID-shaped-var_names branch. This object has
no feature-type column, so the guide matrix resolves from `obsm['gRNA_counts']`
— a branch that never sets the mask. `gex_mask` came out all-True and all 321
guide features stayed in the GEX matrix, feeding HVG selection, PCA, Leiden and
the marker test. Clusters were partly defined by which guide a cell carried.
Cluster/hashtag agreement was 49.5%.

**Fix.** Back-fill both masks by feature *name* after any resolution branch, so
every path is covered including ones added later. Plus a name-recovery step:
when `obsm` names are unresolvable the modality gets placeholders
(`guide_0`, `guide_1`, …) and there is nothing to match on, so names are
recovered from guide-ID-shaped `var_names` when the counts agree.

## 4. Hashtag separability was inverted

v1.1.0 scored separability as the k-means mode gap over the *unweighted* mean
of the two cluster SDs, against a hard cut of 2.5. On this data it flagged the
one good hashtag and cleared the two bad ones:

| hashtag | v1.1.0 | truth |
|---|---|---|
| prot:hash.C | **flagged** (2.587) | deep trough — fine |
| prot:hash.F | passed (2.838) | **no trough at all** |
| prot:hash.D | passed (7.432) | **98.4% zero UMIs, bg_sd 0.005** |

The metric measures how far apart the modes are, which is not what bimodality
means. **Fix:** the primary statistic is now the depth of the trough between
the two modal peaks, relative to the smaller peak; the standardised gap is
retained as a secondary with a properly size-weighted pooled SD; the verdict is
graded `clean / shallow / unimodal / degenerate` rather than boolean.

Two traps, both caught by tests rather than by inspection:

- Reading the density *at the cut* on a raw histogram scores a unimodal
  distribution as perfectly separated whenever the threshold lands in an empty
  bin out in the tail — which is exactly where a 3-SD cut on a unimodal hashtag
  lands. Measure between the peaks instead.
- The smoothing bandwidth must come from the data's spread (Silverman with
  `min(sd, IQR/1.349)`), not from the bin count. A low-abundance hashtag is a
  comb of discrete spikes after log1p, and a fixed-width smoother reads the
  empty bins between them as a trough. This one regressed the existing
  `test_broken_hashtag_is_flagged` before it was fixed.

Also now stated in the report: on this data **every** threshold was set by the
3-SD safety floor, not by the configured `background_quantile` rule. The
caption previously described a rule that never applied.

## 5. Combinatorial hashtagging is now expressible

`call_hashtags()` hardcoded "1 positive = singlet, ≥2 = multiplet", and the
summary graded anything under 45% singlets as `poor`. This experiment reads
39.4% / 57.3%, partly by design.

Cells are now classified **Resolved / Ambiguous / Negative** against
combinations declared in a hashtag whitelist, with `pct_hto_resolved` replacing
`pct_hto_singlet` as the graded metric. Unexpected sets are reported verbatim
in their own table and never rescued by subset matching — a cell positive for
one tag of a declared pair may be a dropout, but it may equally be a cell from
another sample with one spurious tag, and guessing would put it silently into a
comparison group.

`demux_id` may repeat across rows, so one biological sample split into aliquots
tagged differently — one combinatorial, one single — resolves to one sample,
with `hto_aliquot` retained on obs so the aliquots can be compared before the
pool is trusted.

A sample **intentionally not hashtag labelled** is declared with `untagged` in
`hashtag_set`. Its cells then resolve to that sample with its family and
metadata, instead of landing in `Negative` where they are indistinguishable
from capture failure and carry no family — which would drop them out of the
family-scoped comparisons entirely. A blank `hashtag_set` is rejected rather
than assumed to mean this: deliberate and forgotten must not look the same.
Only one untagged sample per pool, since no-hashtag cells cannot be told apart.
The report states the unavoidable caveat: that group also contains every
tagged-sample cell whose capture failed, and only the guide-family cross-check
can separate them.

## 6. Controls are family-scoped

An experiment with four cell lines and four libraries has **four distinct NTC
populations**. v1.1.0 pooled all 60 NTC guides (26,056 cells) into one baseline
backing every knockdown, E-distance and DE comparison.

Family is declared per guide in the gRNA whitelist, never inferred —
`NTC_10_ACGT...` carries no library information and two libraries can
independently use `NTC_10`. Comparison groups are now
`target_key = "{gene}_{family}"`, giving `NTC_A`, `NTC_B`, `NTC_C`, `NTC_D` as
separate baselines. `pool_ntc_across_families` (default `False`) collapses them
where that is genuinely correct.

Family is an opaque token, not a cell line: one cell line with two libraries is
two families, two cell lines with one library is also two families.
`cell_line` and `library` are ordinary metadata columns.

**New cross-check.** Guide family against hashtag family — two independent
assays on the same cell. Off-diagonal cells are doublets or index hops, giving
a contamination estimate independent of any doublet caller. Those cells are
flagged `family_conflict` and excluded from knockdown, E-distance and DE.

This required reordering the pipeline: hashtags now run **before**
perturbation, because the flag has to exist before the comparisons are
computed.

## 7. Plot labels

`ABT1_A_v1.0s2`, `CDH1_B_g1`, `CD55_C_sg`, `ABT1ic_A_v1.1s2`. Family always
shown. `version` keeps its literal value — `1.0` and `1.1` are different
reagents, and stripping the `.0` collapsed six distinct matched controls onto
one label. Any labels still colliding get a 4-character spacer-derived suffix,
so uniqueness never depends on the library's naming discipline.

A guide-ID → target mapping table is now always emitted. v1.1.0 surfaced only a
warning counting the failures, which is why 168 mis-parsed guides were
invisible unless you knew to look at the target list.

---

## Files changed

| file | change |
|---|---|
| `config.py` | 2 new guide regexes; `parse_target_from`; family settings; graded separability params |
| `whitelists.py` | **new** — both whitelist loaders and all validation |
| `manifest.py` | `grna_whitelist` / `hashtag_whitelist` columns; generic `read_table` |
| `guide.py` | two-name precedence; families; `control_of`; labels + uniqueness; `target_key`; mapping table |
| `modalities.py` | mask back-fill by name; placeholder-name recovery |
| `hto.py` | valley-depth separability; degenerate class; design matching; family cross-check |
| `perturb.py` | `TargetAnnotations`; family-scoped controls throughout; conflict exclusion |
| `pipeline.py` | whitelist loading; hashtags before perturbation |
| `tests/test_v120_changes.py` | **new** — 56 checks, one per behaviour above |
| `verify_on_real_data.py` | **new** — applies the new code to the real h5ad |

## Verification

```
python tests/test_v120_changes.py     # 56 checks, all pass
python verify_on_real_data.py /data/.../MDL1856_analyzed_full_depth.h5ad
```

The existing v1.1.0 suite still passes (45 runnable tests; the rest need pytest
fixtures or scipy). `test_guide_parser_targets_and_controls` and
`test_broken_hashtag_is_flagged` both still hold, which is the backwards
compatibility that matters — the new parser handles every ID the old one did,
and the new separability metric still catches the planted broken hashtag.

## Not done

The `--force-recompute` path and the report's Appendix rendering were not
touched. Neither was implicated in anything found here.
