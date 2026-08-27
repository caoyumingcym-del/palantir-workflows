"""
Regression tests for every v1.2.0 change.

Each test names the v1.1.0 behaviour it prevents from coming back. Runs with
plain asserts so it works without pytest:

    python tests/test_v120_changes.py
"""
from __future__ import annotations

import copy
import re
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturbseq_report.config import GuideConfig, HTOConfig, ModalityConfig
from perturbseq_report.guide import GuideParser, call_guides
from perturbseq_report.hto import (
    AMBIGUOUS, NEGATIVE, RESOLVED, call_hashtags, compute_thresholds,
)
from perturbseq_report.manifest import ManifestError
from perturbseq_report.modalities import Modality, split_modalities
from perturbseq_report.perturb import target_annotations
from perturbseq_report.whitelists import (
    cross_check_families, load_guide_whitelist, load_hashtag_whitelist,
    normalise_hashtag_set,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def write(text: str, suffix: str = ".csv") -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


# Real guide IDs from MDL-1856, covering all three naming conventions in it.
REAL_IDS = [
    "ABT1_target_version_1.0_spacer_number_2_spacer_target_ENSG00000146109_TCCATGTTGACTGACACGAG",
    "ABT1_target_version_2.0_spacer_number_2_spacer_target_ENSG00000146109_CAACATGGAGGCAGAGGAAT",
    "ABT1_target_version_1.1_spacer_number_2_spacer_target_ONE_INTERGENIC_SITE_GATGTTACTCACAACCAACC",
    "AKT2_target_version_1.0_spacer_number_1_spacer_target_ENSG00000105221_CGTTGGGCCTGCCTCGGAGG",
    "AKT2_target_version_4.0_spacer_number_1_spacer_target_ENSG00000105221_GGCGCCGGCAGCGGCAGCGG",
    "BMS1_target_version_1.1_spacer_number_2_spacer_target_ONE_INTERGENIC_SITE_GAGCCTACACACCTGAGCAT",
    "INTERGENIC_CONTROL_target_version_7.0_spacer_number_1_spacer_target__INTERGENIC_CONTROL_TAAATACGGTCGTTAATCCC",
    "ONE_INTERGENIC_SITE_target_version_13.0_spacer_number_2_spacer_target_ONE_INTERGENIC_SITE_ACCCAGGATTGTATAACCAT",
    "CDH1_1_TGAACCACCAGGGTATACGT",
    "SNAI2_5_CAGATTCCTCATGTTTGTGC",
    "CD55_singleguide",
    "NTC_10_ACGTTGACCATGCTAAGGCA",
]


# ===========================================================================
def test_guide_parsing() -> None:
    print("\n[guide-ID parsing]")
    p = GuideParser(GuideConfig())
    m = p.parse_all(REAL_IDS).set_index("guide_id")

    # v1.1.0: 168 of 321 guides unparsed, because no regex matched the
    # "_target_version_..._spacer_target_..." scheme.
    check("every real guide ID parses", not p.unparsed, f"unparsed={p.unparsed}")

    # v1.1.0: target_ensg was None for every guide in the library, so
    # resolve_gene() could never fall back to the Ensembl ID.
    row = m.loc[REAL_IDS[0]]
    check("ENSG extracted from structured ID",
          row["target_ensg"] == "ENSG00000146109", str(row["target_ensg"]))
    check("gene symbol taken from the prefix", row["target_gene"] == "ABT1")

    # The doubled underscore in `spacer_target__INTERGENIC_CONTROL`.
    check("doubled-underscore control ID parses",
          m.loc[REAL_IDS[6], "is_ntc"] is np.True_
          or bool(m.loc[REAL_IDS[6], "is_ntc"]))

    # ONE_INTERGENIC_SITE has an underscore before INTERGENIC; a character
    # class of [A-Z0-9]* silently dropped all of these into a looser pattern.
    check("ONE_INTERGENIC_SITE recognised as a control token",
          bool(m.loc[REAL_IDS[7], "is_ntc"]))

    # version_X.1 guides: prefix says ABT1, spacer target says intergenic.
    mc = m.loc[REAL_IDS[2]]
    check("matched control resolves to NTC", mc["target_gene"] == "NTC")
    check("matched control records its gene", mc["control_of"] == "ABT1")
    check("matched control role", mc["role"] == "matched_control")
    check("two-name disagreement flagged", bool(mc["name_conflict"]))

    # parse_target_from="first" must flip that decision.
    p2 = GuideParser(GuideConfig(parse_target_from="first"))
    m2 = p2.parse_all(REAL_IDS).set_index("guide_id")
    check("parse_target_from='first' treats it as targeting",
          m2.loc[REAL_IDS[2], "target_gene"] == "ABT1",
          str(m2.loc[REAL_IDS[2], "target_gene"]))

    # A structured ENSG target must win over the whole-ID NTC substring search:
    # v1.1.0 ran ntc_regex on the full ID unconditionally.
    p3 = GuideParser(GuideConfig())
    r = p3.parse("CTRL1_target_version_1.0_spacer_number_1_spacer_target_ENSG00000111111_ACGTACGTACGTACGTACGT")
    check("gene named CTRL1 is not mistaken for a control",
          not r.is_ntc and r.target_gene == "CTRL1", f"{r.target_gene} ntc={r.is_ntc}")


def test_labels_and_families() -> None:
    print("\n[labels and families]")
    wl = load_guide_whitelist(write(
        "guide_id,family\n" + "".join(
            f"{g},{'A' if i < 8 else 'B'}\n" for i, g in enumerate(REAL_IDS)
        )
    ))
    p = GuideParser(GuideConfig(), wl)
    m = p.parse_all(REAL_IDS).set_index("guide_id")

    check("label carries gene, family and guide discriminator",
          m.loc[REAL_IDS[0], "short_label"] == "ABT1_A_v1.0s2",
          m.loc[REAL_IDS[0], "short_label"])
    check("matched-control label keeps its gene prefix",
          m.loc[REAL_IDS[2], "short_label"] == "ABT1ic_A_v1.1s2",
          m.loc[REAL_IDS[2], "short_label"])
    check("index-style label", m.loc[REAL_IDS[8], "short_label"] == "CDH1_B_g1",
          m.loc[REAL_IDS[8], "short_label"])
    check("singleguide label", m.loc[REAL_IDS[10], "short_label"] == "CD55_B_sg",
          m.loc[REAL_IDS[10], "short_label"])

    # Version must not be zero-stripped: 1.0 and 1.1 are different reagents,
    # and stripping collapsed six distinct matched controls onto one label.
    check("version kept literal",
          "v1.1s2" in m.loc[REAL_IDS[2], "short_label"])

    check("labels are unique", m["short_label"].is_unique,
          str(m["short_label"][m["short_label"].duplicated()].tolist()))

    # Control pools must be family-scoped.
    keys = set(m["target_key"])
    check("NTC pools are per family", {"NTC_A", "NTC_B"} <= keys, str(sorted(keys)))
    check("targets are per family", "ABT1_A" in keys and "CDH1_B" in keys)

    # ...unless explicitly pooled.
    p2 = GuideParser(GuideConfig(pool_ntc_across_families=True), wl)
    keys2 = set(p2.parse_all(REAL_IDS)["target_key"])
    check("pool_ntc_across_families collapses controls",
          "NTC" in keys2 and "NTC_A" not in keys2, str(sorted(keys2)))


def test_label_uniqueness_under_collision() -> None:
    print("\n[label uniqueness]")
    # Two libraries independently using NTC_10 -- the exact case that made
    # gene-only labels collide.
    ids = ["NTC_10_ACGTACGTACGTACGTACGT", "NTC_10_TTTTCCCCGGGGAAAACCCC"]
    wl = load_guide_whitelist(write(
        "guide_id,family\n" + "".join(f"{g},A\n" for g in ids)
    ))
    m = GuideParser(GuideConfig(), wl).parse_all(ids)
    check("colliding labels are disambiguated", m["short_label"].is_unique,
          str(m["short_label"].tolist()))


def test_unlisted_guides_get_their_own_family() -> None:
    print("\n[unlisted guides]")
    wl = load_guide_whitelist(write(
        f"guide_id,family\n{REAL_IDS[0]},A\n{REAL_IDS[6]},A\n"
    ))
    p = GuideParser(GuideConfig(), wl)
    m = p.parse_all(REAL_IDS[:9]).set_index("guide_id")
    check("unlisted guides are recorded", len(p.unlisted) == 7, str(len(p.unlisted)))
    check("unlisted guides land in their own family",
          m.loc[REAL_IDS[1], "family"] == "unassigned",
          m.loc[REAL_IDS[1], "family"])
    check("unlisted controls do not join a declared family's pool",
          m.loc[REAL_IDS[7], "target_key"] == "NTC_unassigned",
          m.loc[REAL_IDS[7], "target_key"])


def test_modality_split_removes_guides_from_gex() -> None:
    print("\n[modality split]")

    class FakeAD:
        def __init__(self, uns_key="gRNA_features"):
            genes = [f"GENE{i}" for i in range(50)]
            guides = [f"G{i}_1_ACGTACGTACGTACGTACGT" for i in range(8)]
            self.var = pd.DataFrame(index=genes + guides)
            self.obs = pd.DataFrame(index=[f"c{i}" for i in range(20)])
            self.n_obs, self.n_vars = 20, len(self.var)
            self.X = np.ones((20, self.n_vars))
            self.obsm = {"gRNA_counts": np.ones((20, 8))}
            self.uns = {uns_key: np.array(guides)}
            self.layers = {}

        def __getitem__(self, k):
            _rows, cols = k
            new = copy.copy(self)
            new.var = self.var[cols]
            new.n_vars = int(np.sum(cols))
            new.X = self.X[:, cols]
            return new

        def copy(self):
            return copy.copy(self)

    gr = GuideConfig().guide_id_regexes
    # v1.1.0: guides resolved from obsm never set guide_mask, so all 8 stayed
    # in GEX and turned up as cluster marker "genes".
    res = split_modalities(FakeAD(), ModalityConfig(), guide_id_regexes=gr)
    check("guides removed from GEX when named in uns",
          res.gex.n_vars == 50, f"n_vars={res.gex.n_vars}")

    res2 = split_modalities(FakeAD("unknown_key"), ModalityConfig(),
                            guide_id_regexes=gr)
    check("guides removed from GEX even when names need recovery",
          res2.gex.n_vars == 50, f"n_vars={res2.gex.n_vars}")
    check("recovery reported",
          any("recovered from var_names" in n for n in res2.notes))


def test_separability() -> None:
    print("\n[hashtag separability]")
    rng = np.random.default_rng(1)
    n = 20000
    df = pd.DataFrame({
        "bimodal": np.concatenate([rng.normal(-2, .6, int(n * .7)),
                                   rng.normal(2.5, .7, int(n * .3))]),
        "unimodal": rng.normal(0, 1, n),
        "spike": np.concatenate([np.zeros(int(n * .985)),
                                 rng.normal(3, .5, int(n * .015))]),
    })
    th = compute_thresholds(df, HTOConfig()).set_index("hashtag")

    check("genuinely bimodal -> clean",
          th.loc["bimodal", "separability"] == "clean",
          th.loc["bimodal", "separability"])
    # v1.1.0 scored this 2.63 SD and passed it at a 2.5 cut.
    check("unimodal noise -> not clean",
          th.loc["unimodal", "separability"] == "unimodal",
          f"{th.loc['unimodal', 'separability']} "
          f"sep_sd={th.loc['unimodal', 'separation_sd']:.2f}")
    # v1.1.0 scored the spike 7.4 SD and reported it as well separated.
    check("degenerate spike -> degenerate",
          th.loc["spike", "separability"] == "degenerate",
          th.loc["spike", "separability"])
    check("standardised gap alone would have passed the noise",
          th.loc["unimodal", "separation_sd"] >= HTOConfig().min_separation_sd,
          "the old metric no longer decides this")


def test_separability_on_low_count_hashtags() -> None:
    """A low-abundance hashtag must not be scored 'clean' by discreteness.

    After log1p a handful of small integer counts is a comb of spikes with
    empty bins between them. A fixed-width smoother reads those empty bins as
    a trough and calls a hashtag that captured nothing perfectly bimodal --
    which is the exact failure this metric exists to catch. The bandwidth is
    therefore set from the data's spread (Silverman), not from the bin count.
    """
    print("\n[separability on low-count hashtags]")
    from perturbseq_report.hto import normalise
    from perturbseq_report.synthetic import make_bundle

    all_ok = True
    for seed in (0, 1, 2, 7, 11, 42):
        b = make_bundle(seed=seed)
        n = b.hto_counts.shape[0]
        mod = Modality(kind="hto", X=b.hto_counts, names=b.hto_names,
                       source="syn", obs_names=[str(i) for i in range(n)])
        th = compute_thresholds(normalise(mod), HTOConfig()).set_index("hashtag")
        broken = set(b.truth.broken_hashtags)
        ok = (
            all(not bool(th.loc[x, "well_separated"]) for x in broken)
            and all(bool(th.loc[h, "well_separated"])
                    for h in th.index if h not in broken)
        )
        all_ok &= ok
    check("planted broken hashtag flagged across seeds", all_ok)


def test_hashtag_design() -> None:
    print("\n[hashtag design]")
    names = [f"prot:hash.{c}" for c in "ABCDEFGH"]
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.negative_binomial(2, 0.8, size=(n, 8)).astype(float)
    grp = rng.integers(0, 3, size=n)
    for i in range(n):
        if grp[i] == 0:
            X[i, 0] += rng.poisson(400); X[i, 1] += rng.poisson(400)
        elif grp[i] == 1:
            X[i, 6] += rng.poisson(400)
        else:
            X[i, 3] += rng.poisson(400)      # hash.D -- not in the design
    mod = Modality(kind="hto", X=X, names=names, source="t",
                   obs_names=[f"c{i}" for i in range(n)])

    wl = load_hashtag_whitelist(write(
        "sample,demux_id,aliquot,hashtag_set,family\n"
        "s1,S1,a1,prot:hash.A+prot:hash.B,A\n"
        "s1,S1,a2,prot:hash.G,A\n"
    ), known_hashtags=names)

    calls = call_hashtags(mod, HTOConfig(), whitelist=wl)
    pc = calls.per_cell
    check("design flagged as declared", calls.design_declared)
    check("classes are Resolved/Ambiguous/Negative",
          set(pc["hto_class"]) <= {RESOLVED, AMBIGUOUS, NEGATIVE},
          str(set(pc["hto_class"])))

    # The whole point: one sample tagged two different ways pools to one id.
    combi = pc.loc[pc["hto_positive_set"] == "prot:hash.A+prot:hash.B", "hto_demux_id"]
    single = pc.loc[pc["hto_positive_set"] == "prot:hash.G", "hto_demux_id"]
    check("combinatorial aliquot resolves to the sample",
          len(combi) > 0 and set(combi) == {"S1"}, str(set(combi)))
    check("single-tag aliquot resolves to the SAME sample",
          len(single) > 0 and set(single) == {"S1"}, str(set(single)))
    check("aliquots stay distinguishable",
          set(pc.loc[pc["hto_class"] == RESOLVED, "hto_aliquot"]) == {"a1", "a2"},
          str(set(pc.loc[pc["hto_class"] == RESOLVED, "hto_aliquot"])))

    # hash.D is a real population that is not declared -- must be Ambiguous,
    # never silently rescued into a neighbouring sample.
    d_only = pc.loc[pc["hto_positive_set"] == "prot:hash.D", "hto_class"]
    check("undeclared combination is Ambiguous, not rescued",
          len(d_only) > 0 and set(d_only) == {AMBIGUOUS}, str(set(d_only)))
    check("unexpected sets are tabulated",
          calls.unexpected_sets is not None and len(calls.unexpected_sets) > 0)
    check("pct_resolved replaces pct_singlet",
          "pct_resolved" in calls.rates)

    # Without a whitelist, v1.1.0 behaviour, and no crash.
    plain = call_hashtags(mod, HTOConfig())
    check("no whitelist falls back to singlet/multiplet",
          not plain.design_declared and "pct_singlet" in plain.rates)
    check("fallback says so in a note",
          any("No hashtag whitelist" in n for n in plain.notes))


def test_intentionally_untagged_sample() -> None:
    """A sample deliberately left unlabelled must be declarable.

    Omitting it from the whitelist puts its cells in Negative, where they are
    indistinguishable from hashtag capture failure -- and, worse, they carry no
    family, so they drop out of the family-scoped comparisons entirely.
    """
    print("\n[intentionally untagged sample]")
    names = [f"prot:hash.{c}" for c in "ABCD"]
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.negative_binomial(2, 0.8, size=(n, 4)).astype(float)
    grp = rng.integers(0, 3, size=n)
    for i in range(n):
        if grp[i] == 0:
            X[i, 0] += rng.poisson(400); X[i, 1] += rng.poisson(400)
        elif grp[i] == 1:
            X[i, 2] += rng.poisson(400)
        # grp == 2 gets no tag at all: the untagged sample
    mod = Modality(kind="hto", X=X, names=names, source="t",
                   obs_names=[f"c{i}" for i in range(n)])

    header = "sample,demux_id,hashtag_set,family\n"
    tagged = "s1,S1,prot:hash.A+prot:hash.B,A\ns1,S2,prot:hash.C,A\n"

    undeclared = load_hashtag_whitelist(write(header + tagged),
                                        known_hashtags=names)
    c1 = call_hashtags(mod, HTOConfig(), whitelist=undeclared)
    neg = c1.per_cell.loc[c1.per_cell["hto_positive_set"] == NEGATIVE]
    check("undeclared: untagged cells fall into Negative",
          set(neg["hto_class"]) == {NEGATIVE})
    check("undeclared: they carry no family",
          set(neg["hto_family"]) == {""})

    declared = load_hashtag_whitelist(
        write(header + tagged + "s1,S3,untagged,B\n"), known_hashtags=names)
    check("untagged row recognised", declared.has_untagged)
    c2 = call_hashtags(mod, HTOConfig(), whitelist=declared)
    neg2 = c2.per_cell.loc[c2.per_cell["hto_positive_set"] == NEGATIVE]
    check("declared: untagged cells resolve to the sample",
          set(neg2["hto_class"]) == {RESOLVED}, str(set(neg2["hto_class"])))
    check("declared: they get the sample's demux_id",
          set(neg2["hto_demux_id"]) == {"S3"}, str(set(neg2["hto_demux_id"])))
    check("declared: they get the sample's family",
          set(neg2["hto_family"]) == {"B"}, str(set(neg2["hto_family"])))
    check("declared: tagged samples are unaffected",
          c2.rates["pct_ambiguous"] == c1.rates["pct_ambiguous"])
    check("the capture-failure ambiguity is stated, not hidden",
          any("UNAVOIDABLE CAVEAT" in nt for nt in c2.notes))

    # A blank cell must NOT be read as "untagged" -- that is an omission.
    try:
        load_hashtag_whitelist(write(header + tagged + "s1,S3,,B\n"),
                               known_hashtags=names)
    except ManifestError as exc:
        check("blank hashtag_set still rejected, and points at the fix",
              "untagged" in str(exc))
    else:
        check("blank hashtag_set still rejected", False, "no error raised")

    # Two untagged rows in one pool cannot be told apart.
    try:
        load_hashtag_whitelist(
            write(header + tagged + "s1,S3,untagged,B\ns1,S4,untagged,C\n"),
            known_hashtags=names)
    except ManifestError:
        check("two untagged samples in one pool rejected", True)
    else:
        check("two untagged samples in one pool rejected", False)


def test_whitelist_validation() -> None:
    print("\n[whitelist validation]")

    def expect_error(label: str, fn) -> None:
        try:
            fn()
        except ManifestError:
            check(label, True)
        except Exception as exc:  # noqa: BLE001
            check(label, False, f"wrong exception: {type(exc).__name__}: {exc}")
        else:
            check(label, False, "no error raised")

    expect_error("duplicate guide_id rejected", lambda: load_guide_whitelist(write(
        "guide_id,family\nG1,A\nG1,B\n")))
    expect_error("blank family rejected", lambda: load_guide_whitelist(write(
        "guide_id,family\nG1,\n")))
    expect_error("over-long family rejected", lambda: load_guide_whitelist(write(
        "guide_id,family\nG1,this_is_far_too_long\n")))
    expect_error("bad parse_target_from rejected", lambda: load_guide_whitelist(write(
        "guide_id,family,parse_target_from\nG1,A,third\n")))
    expect_error("bad ENSG rejected", lambda: load_guide_whitelist(write(
        "guide_id,family,target_ensg\nG1,A,NOTANID\n")))

    expect_error("duplicate hashtag_set rejected", lambda: load_hashtag_whitelist(write(
        "sample,demux_id,hashtag_set\ns,S1,h.A+h.B\ns,S2,h.B+h.A\n")))
    expect_error("repeated demux_id without aliquot rejected",
                 lambda: load_hashtag_whitelist(write(
                     "sample,demux_id,hashtag_set\ns,S1,h.A\ns,S1,h.B\n")))
    expect_error("aliquots disagreeing on metadata rejected",
                 lambda: load_hashtag_whitelist(write(
                     "sample,demux_id,aliquot,hashtag_set,cell_line\n"
                     "s,S1,a1,h.A,A375\ns,S1,a2,h.B,HT29\n")))
    expect_error("unknown hashtag name rejected",
                 lambda: load_hashtag_whitelist(
                     write("sample,demux_id,hashtag_set\ns,S1,hash.A\n"),
                     known_hashtags=["prot:hash.A"]))

    # Repeated demux_id WITH distinct aliquots is the supported case.
    ok = load_hashtag_whitelist(write(
        "sample,demux_id,aliquot,hashtag_set\ns,S1,a1,h.A+h.B\ns,S1,a2,h.C\n"))
    check("one sample, two tagging schemes accepted", ok.has_aliquots)
    check("hashtag set order does not matter",
          normalise_hashtag_set("h.B+h.A") == normalise_hashtag_set("h.A+h.B"))

    # The two files must share one family vocabulary.
    g = load_guide_whitelist(write("guide_id,family\nG1,A\n"))
    h = load_hashtag_whitelist(write(
        "sample,demux_id,hashtag_set,family\ns,S1,h.A,ZZ\n"))
    expect_error("family vocabulary mismatch rejected",
                 lambda: cross_check_families(g, h))


def test_family_scoped_controls() -> None:
    print("\n[family-scoped controls]")
    mapping = pd.DataFrame({
        "guide_id": ["g1", "g2", "g3", "g4"],
        "target_gene": ["ABT1", "NTC", "CDH1", "NTC"],
        "target_ensg": ["ENSG1", None, "ENSG2", None],
        "family": ["A", "A", "B", "B"],
        "target_key": ["ABT1_A", "NTC_A", "CDH1_B", "NTC_B"],
        "is_ntc": [False, True, False, True],
    })
    ann = target_annotations(mapping, "NTC")
    check("each family maps to its own control group",
          ann.ntc_key_by_family == {"A": "NTC_A", "B": "NTC_B"},
          str(ann.ntc_key_by_family))
    check("both control groups recognised",
          ann.ntc_keys == {"NTC_A", "NTC_B"})
    check("ENSG survives to the target lookup",
          ann.ensg_by_key.get("ABT1_A") == "ENSG1")
    check("family count", ann.n_families == 2)


def test_guide_calling_end_to_end() -> None:
    print("\n[guide calling end to end]")
    wl = load_guide_whitelist(write(
        "guide_id,family\n" + "".join(
            f"{g},{'A' if i < 8 else 'B'}\n" for i, g in enumerate(REAL_IDS)
        )
    ))
    rng = np.random.default_rng(0)
    n = 240
    X = rng.poisson(0.3, size=(n, len(REAL_IDS))).astype(float)
    for i in range(n):
        X[i, i % len(REAL_IDS)] += rng.poisson(80)
    mod = Modality(kind="guide", X=X, names=REAL_IDS, source="t",
                   obs_names=[f"c{i}" for i in range(n)])
    ga = call_guides(mod, GuideConfig(), whitelist=wl)

    check("per-cell frame carries target_key", "target_key" in ga.per_cell.columns)
    check("per-cell frame carries family", "family" in ga.per_cell.columns)
    check("per-cell frame carries short_label", "short_label" in ga.per_cell.columns)
    check("families detected", ga.families == ["A", "B"], str(ga.families))
    check("targets counted per family, controls excluded",
          ga.n_targets == 5, str(ga.n_targets))
    check("family note emitted",
          any("scoped per family" in n for n in ga.notes))


def main() -> int:
    for fn in (
        test_guide_parsing,
        test_labels_and_families,
        test_label_uniqueness_under_collision,
        test_unlisted_guides_get_their_own_family,
        test_modality_split_removes_guides_from_gex,
        test_separability,
        test_separability_on_low_count_hashtags,
        test_hashtag_design,
        test_intentionally_untagged_sample,
        test_whitelist_validation,
        test_family_scoped_controls,
        test_guide_calling_end_to_end,
    ):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
