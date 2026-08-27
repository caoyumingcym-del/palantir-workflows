# perturbseq_report v1.2.0 — running on ICA

Self-contained. Nothing needs to be installed from this bundle: the launcher
adds the package directory to `sys.path` itself.

---

## 1. Dependencies

```bash
pip install -r requirements.txt
```

Required: `numpy pandas matplotlib anndata scanpy scipy scikit-learn`.
Recommended: `leidenalg igraph` (graph-based clustering — without them the
pipeline falls back to k-means and says so in the report), `harmonypy`
(batch correction), `pyyaml` (YAML `--config`).

Python >= 3.10.

---

## 2. Prepare the inputs

Three files, all CSV. Templates are in `examples/`.

### a. Sample manifest — required

```csv
sample,prefix,dragen_path,h5ad_path,output_path,grna_whitelist,hashtag_whitelist,HTO,min_genes,max_genes,min_counts,max_counts,max_mito
sample_1,PH20260504_1_1,/data/.../dragen/PH20260504_1_1,/data/.../MDL1856.h5ad,/data/.../out,whitelists/grna.csv,whitelists/hto.csv,yes,521.7,7481,822,11480,10
```

`h5ad_path`, `output_path`, `grna_whitelist` and `hashtag_whitelist` are
**global**: identical on every row. All paths resolve relative to the
manifest's own directory, so a relative `whitelists/grna.csv` works regardless
of the working directory the job starts in.

Leave the five threshold columns blank to have them derived from the data
(see the explore step below).

### b. gRNA whitelist — optional but strongly recommended

Two columns are mandatory. Everything else is derived from the guide ID.

```csv
guide_id,family
ABT1_target_version_1.0_spacer_number_2_spacer_target_ENSG00000146109_TCCATG...,A
ABT1_target_version_1.1_spacer_number_2_spacer_target_ONE_INTERGENIC_SITE_...,A
CDH1_1_TGAACCACCAGGGTATACGT,B
NTC_10_ACGTTGACCATGCTAAGGCA,B
```

`family` is what scopes each cell's NTC control pool. Without this file every
guide lands in one pool, which is wrong for any experiment running more than
one guide population. See `examples/WHITELISTS.md` for the optional columns.

### c. Hashtag whitelist — optional, needed for combinatorial tagging

```csv
sample,demux_id,aliquot,hashtag_set,family,cell_line,condition,replicate
sample_1,S01,a1,prot:hash.A+prot:hash.B,A,A375,untreated,1
sample_1,S01,a2,prot:hash.G,A,A375,untreated,1
sample_1,S02,a1,prot:hash.C,B,HT29,untreated,1
```

Mandatory: `sample`, `demux_id`, `hashtag_set`. `aliquot` becomes required once
a `demux_id` repeats. Everything else is optional metadata, though `family` is
what links a cell's hashtag to its guide population.

A sample **intentionally not hashtag labelled** gets `untagged` in
`hashtag_set` — not a blank cell, and not an omitted row:

```csv
sample_1,S04,a1,untagged,D,RPE1
```

Hashtag names must match the data exactly (`prot:hash.A`, not `hash.A`) — a
mismatch stops the run and lists the real names. `family` values must exist in
the gRNA whitelist.

---

## 3. Run

The pipeline is **explore-first**: a run with no thresholds stops after QC so
the distributions can be reviewed before anything is filtered.

```bash
# Step 1 -- QC only. Writes analysis_outputs/qc_explore.html
python run_perturbseq_report.py --manifest /path/to/sample_manifest.csv --explore

# Step 2 -- fill the threshold columns in the manifest, then run end to end.
#           Writes analysis_outputs/qc_report.html
python run_perturbseq_report.py --manifest /path/to/sample_manifest.csv
```

To skip the review step and use auto-derived thresholds in one pass:

```bash
python run_perturbseq_report.py --manifest /path/to/sample_manifest.csv --auto-thresholds
```

Explore and full runs write to **different** filenames, so finding
`qc_report.html` on disk always means a complete analysis.

### Useful flags

| flag | when |
|---|---|
| `--subsample-cells 100000` | large object, memory-constrained node |
| `--low-memory` | trades speed for peak RSS |
| `--counts-layer counts` | `X` holds normalised values and raw counts are in a layer |
| `--force-recompute` | ignore QC metrics / PCA already present in the object |
| `--inspect FILE.h5ad` | print the object's structure and exit; no analysis |
| `--config cfg.yaml` | override any config dataclass field |

`python run_perturbseq_report.py --help` lists all of them.

### Resources

The reference run (339k cells × 38.6k genes) needs roughly **48–64 GB RAM** and
1–2 hours single-threaded. `--subsample-cells` is the escape hatch; the report
states the subsample size prominently, since per-target cell counts scale down
with it.

---

## 4. Outputs

Everything lands under `output_path/analysis_outputs/`:

```
qc_report.html          self-contained, images inlined -- the deliverable
qc_explore.html         from --explore runs
figures/                PNGs, also embedded in the HTML
tables/                 every table as CSV, including:
  guide_target_mapping.csv     guide -> target, family, label, which pattern matched
  hto_thresholds.csv           per-hashtag threshold and separability verdict
  hto_unexpected_sets.csv      positive sets matching no declared combination
  crosscheck_family.csv        guide family vs hashtag family
  perturbation_knockdown.csv   per target_key, with its family's control count
checkpoints/            intermediate state for --rebuild-report
artifacts.json          everything the report is built from
```

The HTML is a single file with images base64-inlined, so it can be copied off
ICA on its own.

---

## 5. Verify before trusting a run

```bash
# 56 regression checks, no data needed, ~10 seconds
python tests/test_v120_changes.py

# Applies the new parsing/separability code to the real h5ad and prints
# each number against the v1.1.0 value. Read-only, backed mode, no pipeline.
python verify_on_real_data.py /data/.../MDL1856.h5ad --grna-whitelist whitelists/grna.csv
```

On MDL-1856 the second command should report 0 unparsed guides (v1.1.0: 168),
~77 target groups (v1.1.0: 186), ~178 guides yielding an ENSG (v1.1.0: 0), the
GEX matrix narrowing by 321 features, and `prot:hash.C` no longer flagged while
`prot:hash.F` and `prot:hash.D` are.

---

## 6. First things to read in the report

1. **Appendix → Modality detection.** Confirms guide and hashtag features were
   removed from the gene-expression matrix. If a "features were also present in
   var" warning appears, that is v1.2.0 catching the v1.1.0 contamination bug.
2. **Guides → Guide ID to target mapping.** Check a few rows. `annotation_source`
   says whether values came from the whitelist or were derived.
3. **Guides → notes.** Any guide not in the whitelist, or whose two names
   disagree, is listed here.
4. **Hashtags → thresholds table.** `separability` is graded
   `clean / shallow / unimodal / degenerate`. Anything not `clean` means calls
   involving that tag are unreliable.
5. **Cross-checks → guide family vs hashtag family.** The off-diagonal rate is
   a doublet/index-hop estimate independent of any doublet caller.

---

## 7. Known constraints

- `leidenalg` absent → k-means fallback, stated in the report.
- Guides absent from the gRNA whitelist go to family `unassigned` and are never
  merged into a declared family's control pool. The run continues and lists
  them.
- Cells whose guide family disagrees with their hashtag family are excluded
  from knockdown, E-distance and DE. They are counted, not dropped silently.
- `CHANGELOG_v1.2.0.md` documents every behaviour change from v1.1.0 with the
  measurement that motivated it.
