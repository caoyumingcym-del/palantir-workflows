# Whitelist specification (v1.2.0)

Two optional CSVs, both pointed to from the sample manifest. They replace
guesswork with declaration: the gRNA whitelist says which **family** each guide
belongs to, and the hashtag whitelist says which hashtag combination identifies
which family and condition.

Together they let a guide-positive cell be compared **only against NTC cells
from its own family**, which is the requirement that motivated them.

---

## What "family" means

A family is an **opaque identifier for one guide population** — the set of
guides whose NTCs form a valid control group for each other. Nothing more.

It is deliberately *not* "cell line", because family is not always cell line:

- one cell line screened with two different CRISPR libraries → **two families**
- two cell lines screened with the same library → **two families**
- the same library in the same cell line at two timepoints → **one family**

So family IDs are short arbitrary tokens — `A`, `B`, `C`, `D`, or `lib1`,
`lib2` — and the descriptive attributes (`cell_line`, `library`, whatever else)
live in their own columns as metadata. The example files show families `A` and
`B` as the *same* cell line (A375) with two different libraries, and families
`B` and `C` as the *same* library in two different cell lines.

Requirements: unique, short (they appear in every plot label), and matching
`[A-Za-z0-9_.-]{1,8}`. Validated at load.

---

## Why family must be declared, not inferred

In an experiment with four guide populations there are four distinct NTC
populations. An `NTC` guide from family A is not a valid control for a cell
carrying a family B guide — different genetic background, different library
prep, or both.

v1.1.0 pooled all 60 NTC guides (26,056 cells) into one baseline and measured
every knockdown, E-distance and DE comparison against it. Family cannot be
recovered from the guide ID alone: `NTC_10_ACGT…` carries no library
information, and two libraries can independently use `NTC_10`.

---

## Sample manifest — two new columns

```csv
sample,prefix,h5ad_path,output_path,grna_whitelist,hashtag_whitelist,...
sample_1,PH20260504_1_1,...,...,whitelists/MDL1856_grna_whitelist.csv,whitelists/MDL1856_hashtag_whitelist.csv,...
```

- Paths resolve **relative to the manifest's own directory**, matching the
  existing behaviour of `h5ad_path` / `output_path`.
- Validated as global columns: identical (or blank) on every row, since they
  describe the experiment rather than a sequencing run.
- Both optional. Absent → v1.1.0 behaviour: no family scoping, hashtag calling
  as plain singlet/multiplet. The report states which mode was used.

---

## 1. gRNA whitelist

One row per guide in the library.

### Minimal form — two columns

Only `guide_id` and `family` are mandatory. Everything else is derived from the
guide ID by the parser, which covers 321/321 guides in this library:

```csv
guide_id,family
ABT1_target_version_1.0_spacer_number_2_spacer_target_ENSG00000146109_TCCATGTTGACTGACACGAG,A
ABT1_target_version_1.1_spacer_number_2_spacer_target_ONE_INTERGENIC_SITE_GATGTTACTCACAACCAACC,A
CDH1_1_TGAACCACCAGGGTATACGT,B
NTC_10_ACGTTGACCATGCTAAGGCA,B
CD55_singleguide,C
```

From those two columns alone the second row resolves to `target_gene = NTC`,
`role = matched_control`, `control_of = ABT1`, label `ABT1ic_A_v1.1s2`.

The optional columns exist for three situations only:

1. the parser cannot read the ID (unusual naming, no structure) — supply
   `target_gene` / `target_ensg`;
2. the derived answer is wrong for that library — override `role`,
   `control_of`, or flip `parse_target_from` to `first`;
3. you want a specific plot label — set `short_label`.

Supplying a column for some rows and leaving it blank for others is fine:
blank means "derive it". The report lists which fields were declared and which
were derived, so a whitelist is never silently doing more than you intended.

### Full form

```csv
guide_id,family,parse_target_from,target_gene,target_ensg,role,control_of,short_label
ABT1_target_version_1.0_spacer_number_2_spacer_target_ENSG00000146109_TCCATGTTGACTGACACGAG,A,second,ABT1,ENSG00000146109,targeting,,
ABT1_target_version_1.1_spacer_number_2_spacer_target_ONE_INTERGENIC_SITE_GATGTTACTCACAACCAACC,A,second,NTC,,matched_control,ABT1,
INTERGENIC_CONTROL_target_version_7.0_spacer_number_1_spacer_target__INTERGENIC_CONTROL_TAAATACGGTCGTTAATCCC,A,second,NTC,,ntc,,
CDH1_1_TGAACCACCAGGGTATACGT,B,,CDH1,ENSG00000039068,targeting,,
NTC_10_ACGTTGACCATGCTAAGGCA,B,,NTC,,ntc,,
CD55_singleguide,C,,CD55,ENSG00000196352,targeting,,CD55_sg
```

### Columns

| column | required | meaning |
|---|---|---|
| `guide_id` | **yes** | Full-length ID, byte-identical to the feature name in `var` / `obsm`. Join key. Must be unique. |
| `family` | **yes** | Guide-population ID. Scopes the NTC pool and every per-target comparison. Short token; appears in plot labels. |
| `parse_target_from` | no | `first` \| `second`. Which of the two names in the ID is authoritative. **Derived: `second`.** See below. |
| `target_gene` | no | Gene symbol; `NTC` for controls. **Derived:** the authoritative name, with the first name kept as the symbol when the second is an ENSG. |
| `target_ensg` | no | Ensembl gene ID, used by `resolve_gene()` when the symbol is not in `var_names`. **Derived:** the `spacer_target` field when it matches `ENS[A-Z]*G\d+`. |
| `role` | no | `targeting` \| `ntc` \| `matched_control` \| `safe_harbour`. **Derived:** `ntc` when the authoritative name is a control/intergenic token, else `targeting`; upgraded to `matched_control` when the first name is a gene that appears as a target in the same family. |
| `control_of` | no | The gene whose construct a matched control mirrors. **Derived:** the first name, for guides resolved to `matched_control`. |
| `short_label` | no | Plot label. **Derived:** from the captured fields plus the family suffix. |

### `parse_target_from` — which name wins

Many guide IDs carry **two** target names: a gene-symbol prefix and a
`spacer_target` field. They do not always agree, and when they disagree the
disagreement is meaningful:

| guide ID | first name | second name | agree? |
|---|---|---|---|
| `ABT1_target_version_1.0_..._spacer_target_ENSG00000146109_...` | `ABT1` | `ENSG00000146109` | yes — same gene |
| `ABT1_target_version_1.1_..._spacer_target_ONE_INTERGENIC_SITE_...` | `ABT1` | `ONE_INTERGENIC_SITE` | **no** |
| `INTERGENIC_CONTROL_target_version_7.0_..._spacer_target__INTERGENIC_CONTROL_...` | `INTERGENIC_CONTROL` | `INTERGENIC_CONTROL` | yes |

The prefix records what the construct was *designed from*; the `spacer_target`
records what the spacer actually cuts. For the `version_X.1` matched controls
the prefix is stale — the construct keeps the `ABT1` name but the spacer was
redirected to an intergenic site.

- **`second` (default)** — target identity comes from the `spacer_target`
  field. An ENSG makes the guide targeting; an intergenic/control token makes
  it an NTC. This classifies the matched controls correctly with no special
  case, and it is the biologically true answer.
- **`first`** — target identity comes from the gene-symbol prefix. Use when a
  library's `spacer_target` field is unreliable or absent, or when you
  deliberately want `version_X.1` guides analysed under their design gene.

Notes:

- When the ID carries only one name (`CDH1_1_TGAACC…`, `CD55_singleguide`),
  the column is ignored — leave it blank, as in the example rows above.
- When `second` is an ENSG, the symbol from `first` is kept as the display
  name, so labels stay readable while `target_ensg` drives `resolve_gene()`.
- The setting is per row, so one file can mix libraries with different
  conventions. `GuideConfig.parse_target_from` sets the default when the
  column is absent entirely; it ships as `"second"`.
- Whenever the two names disagree and the guide is not already annotated, the
  report notes it — a large count of disagreements usually means the wrong
  setting for that library.

### Validation at load

- Every `guide_id` unique; duplicates raise `ManifestError`.
- `family` matches `[A-Za-z0-9_.-]{1,8}`.
- `parse_target_from` is `first`, `second` or blank.
- `target_ensg` matches `ENS[A-Z]*G\d+` when non-blank.
- `control_of` names a gene that appears as a `target_gene` in the same family.
- Guides in the object but **missing** from the whitelist: warn, parse from the
  guide ID as in v1.1.0, assign `family = "unassigned"`, and list every one in
  a report note. `unassigned` is its own family for NTC scoping — never merged
  into a declared one. Guides in the whitelist but absent from the object are
  counted and reported.

---

## 2. Hashtag whitelist

One row per valid hashtag combination.

```csv
sample,demux_id,aliquot,hashtag_set,family,cell_line,library,condition,replicate
sample_1,S01,a1,prot:hash.A+prot:hash.B,A,A375,brunello_v1,untreated,1
sample_1,S01,a2,prot:hash.G,A,A375,brunello_v1,untreated,1
sample_1,S02,a1,prot:hash.A+prot:hash.C,A,A375,brunello_v1,untreated,2
sample_1,S04,a1,prot:hash.B+prot:hash.F,B,A375,emt_focused,untreated,1
sample_1,S04,a2,prot:hash.D,B,A375,emt_focused,untreated,1
sample_1,S06,a1,prot:hash.C+prot:hash.G,C,HT29,emt_focused,untreated,1
```

Two things to read off this example:

- Families `A` and `B` are the same cell line with different libraries;
  families `B` and `C` are the same library in different cell lines. Neither
  attribute alone defines the family, which is why it is its own column.
- `S01` and `S04` each appear **twice**: one aliquot tagged combinatorially,
  another tagged with a single hashtag. Both aliquots resolve to the same
  biological sample. See below.

### One sample, several tagging schemes

A `demux_id` is a **biological sample and may appear on any number of rows**.
Uniqueness is enforced on the `hashtag_set`, not on the `demux_id`, precisely
so that one sample can be split into aliquots that were tagged differently:

| row | tagging | resolves to |
|---|---|---|
| `S01, a1, prot:hash.A+prot:hash.B` | combinatorial | `demux_id = S01` |
| `S01, a2, prot:hash.G` | single | `demux_id = S01` |

Cells matching either set get `hto_demux_id = S01` and are pooled for every
downstream comparison. `hto_aliquot` is retained on `obs` as its own column, so
the aliquots can still be compared against each other — worth doing once before
trusting the pool, since differently-tagged aliquots are a plausible source of
batch effect.

This also covers the reverse case: a sample tagged combinatorially in one lane
and singly in another, or a rescue aliquot added late with whatever tags were
left.

### Columns

| column | required | meaning |
|---|---|---|
| `sample` | **yes** | Links to the sample manifest's sample column — the sequencing library / 10x channel, not the biological sample. |
| `demux_id` | **yes** | Biological sample identity. **Repeats across rows** when one sample was tagged more than one way. |
| `hashtag_set` | **yes** | `+`-delimited hashtag names for combinatorial tagging, or a bare name for single tagging. Names must match the hashtag feature names exactly (`prot:hash.A`, not `hash.A`). Unique within a sample. |
| `aliquot` | conditional | Distinguishes rows that share a `demux_id`. Required when a `demux_id` appears more than once; may be blank otherwise. |
| `family` | no | Guide population for this sub-sample. **Must match a `family` value in the gRNA whitelist.** This is the link that scopes the NTC pool. |
| anything else | no | Free metadata (`cell_line`, `library`, `condition`, `replicate`, `timepoint`, …) joined onto `obs` and offered as a comparison axis, exactly like existing manifest metadata columns. |

### Samples that were intentionally not hashtag labelled

Write **`untagged`** in `hashtag_set`. Do not omit the row and do not leave the
cell blank — a blank cell is rejected, because "deliberately unlabelled" and
"nobody filled this in" must not look the same.

```csv
sample,demux_id,hashtag_set,family
sample_1,S01,prot:hash.A+prot:hash.B,A
sample_1,S02,prot:hash.C,A
sample_1,S03,untagged,B
```

Cells with no positive hashtag then resolve to `S03` — with its `demux_id`,
`family` and metadata — instead of falling into `Negative`. That matters twice
over: `Negative` is also where hashtag capture failure lands, and a cell with
no family drops out of the family-scoped perturbation comparisons entirely.

Accepted markers: `untagged`, `unhashed`, `no_hashtag`, `no_hashtags`,
`nohashtag`, `unlabelled`, `unlabeled`. Only one untagged sample per pool —
cells carrying no hashtag are indistinguishable from each other, so a second
one could not be resolved.

**The caveat, which the report states rather than hides.** Cells assigned to an
untagged sample necessarily include every cell from a *tagged* sample whose
hashtag capture failed. Hashtag data alone cannot separate them. If the
untagged sample has its own guide family, the guide-family cross-check does
separate them; otherwise treat its cell count as an upper bound.

### Semantics

- Order within a set is irrelevant — sets are normalised to a sorted frozenset
  before matching.
- Single and combinatorial tagging can coexist freely, both across samples and
  within one sample's aliquots.
- The same hashtag may appear in several sets, as long as no two **sets** are
  identical.
- Several `demux_id`s may share a family — the normal case, one per
  condition/replicate.

### Validation at load

- Every hashtag name appears in the object's hashtag features; unknown names
  raise `ManifestError` listing both the unknown names and the names present.
- No duplicate `hashtag_set` within a sample; `(sample, hashtag_set)` is the
  primary key.
- `(sample, demux_id, aliquot)` unique. If a `demux_id` appears on more than
  one row with a blank or duplicated `aliquot`, raise — the rows would be
  indistinguishable on `obs`.
- All rows sharing a `demux_id` must agree on `family` and on every metadata
  column. Disagreement raises, since the same biological sample cannot be two
  cell lines or two conditions. This catches copy-paste errors that would
  otherwise silently merge unrelated cells.
- Every `family` appears in the gRNA whitelist; otherwise raise, since a
  population with no guides cannot be analysed.
- Warn when a hashtag present in the object appears in no set — it was
  sequenced but is not part of the declared design.

### Reported per aliquot

When any `demux_id` spans multiple aliquots, the report adds a per-aliquot
breakdown — cell count, median UMIs, %mito, guide-assignment rate — so an
aliquot that behaved differently is visible before its cells are pooled into
the sample.

---

## How the two connect

```
   cell
     ├── hashtag positive set ──► hashtag whitelist ──► demux_id, family, condition, cell_line
     └── assigned guide ────────► gRNA whitelist ─────► family, target_gene, role
                                                          │
                          the two families must agree ────┘
```

This yields three things v1.1.0 could not produce:

1. **Family-scoped controls.** Comparison groups become
   `target_key = f"{target_gene}_{family}"`, so family-A knockdowns are
   measured against family-A NTC cells only.
   `GuideConfig.pool_ntc_across_families` defaults to `False`.
2. **Resolved / Ambiguous / Negative** hashtag classes replacing
   singlet/multiplet, with `pct_hto_resolved` as the summary metric and a table
   of the most common unexpected sets.
3. **A new cross-check panel: guide family vs hashtag family.** A cell
   hashtagged family A but carrying a family B guide is a doublet or index hop.
   Such cells keep both labels plus a `family_conflict` flag, are counted in
   the panel, and are **excluded from knockdown, E-distance and DE** so they
   cannot contaminate either baseline. The off-diagonal rate is a direct,
   interpretable contamination estimate — measurable *only* because both
   whitelists are declared.

---

## Plot labels

Family is a short opaque token, so it reads cleanly as a suffix and keeps
target-level axes compact:

| level | label |
|---|---|
| target | `ABT1_A`, `NTC_A`, `NTC_B`, `CDH1_B`, `CD55_C` |
| guide | `ABT1_A_v1.0s2`, `CDH1_B_g1`, `CD55_C_sg` |
| matched control | `ABT1ic_A_v1.1s2` (`target_gene` stays `NTC`) |

Guide-level suffixes come from whichever fields the matching pattern captured
(`_v{version}s{spacer_number}`, `_g{index}`, `_sg`). `version` keeps its
literal value — `1.0` and `1.1` are different reagents. Any labels still
colliding get a 4-character spacer-derived suffix, so uniqueness never depends
on the library's naming discipline. `short_label` in the whitelist overrides
all of this per guide; the family suffix is still appended unless the override
already ends in one.

The separate NTC pools appear as `NTC_A`, `NTC_B`, `NTC_C`, `NTC_D` — visible
and countable in the report rather than silently merged.

---

## `dragen_runs_share_cells` — required when a sample has more than one run

Only relevant when the h5ad has no hashtag matrix at all and the pipeline is
falling back to reading `<dragen_path>/<prefix>.scRNA.cellhashing.tsv`
directly (see `hto_dragen.py`). Irrelevant, and not required, for a sample
with exactly one `prefix`/`dragen_path` row.

```csv
sample,prefix,dragen_path,h5ad_path,output_path,...,dragen_runs_share_cells
sample_1,PH20260504_1_1,.../PH20260504_1_1/dragen_output,...,yes
sample_1,PH20260504_1_2,.../PH20260504_1_2/dragen_output,...,yes
sample_1,PH20260504_1_3,.../PH20260504_1_3/dragen_output,...,yes
sample_1,PH20260504_1_4,.../PH20260504_1_4/dragen_output,...,yes
```

A manifest can have several samples, and a sample's multiple runs can mean
either of two different things that look identical on disk:

- **the same cells/cDNA, split across several library preparations** (e.g.
  one pool sequenced across 4 lanes/libraries for depth). A barcode
  recovered from more than one run's `cellhashing.tsv` is the same physical
  cell, and its hashtag counts must be **summed** across runs.
- **independent runs or pools** that happen to share a row in this manifest.
  A barcode recovered from more than one run's file there is coincidental
  reuse of the same string from DRAGEN's shared combinatorial barcode
  whitelist (which is written over close to its full space, not just real
  cells) — summing would inflate a real cell's count with an unrelated run's
  background. Only the strongest-matching run's counts are kept.

Set `yes` for the first case, `no` for the second, once per sample (every run
row for that sample must agree — a sample can't be both). **Required**
whenever a sample has more than one run: with it blank, the DRAGEN fallback
is skipped for that sample entirely rather than guessing, because guessing
wrong changes every reported hashtag count for that sample by up to *N*×,
where *N* is its number of runs.

Establish which case applies with `diagnose_cellhashing_tsv.py --manifest
<this file>` before setting it: section G compares actual hashtag counts for
the same barcode across every run and shows directly whether they carry
comparable real signal in every run (→ `yes`) or real signal in one and
near-zero elsewhere (→ `no`).

Excluded from condition-column autodetection like the other run/threshold
columns: it is a declaration about library preparation, not a comparison you
would ever want plotted as an experimental condition, even though its value
is allowed to differ from sample to sample within one manifest.
