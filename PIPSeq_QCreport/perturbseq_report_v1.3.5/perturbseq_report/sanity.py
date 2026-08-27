"""
Input plausibility checks on the expression matrix.

Added in v1.2.1 after an h5ad whose ``var_names`` were permuted relative to the
columns of ``X`` produced two complete, confident, entirely wrong reports. The
defect was invisible in the report itself: guide calling, hashtag calling and
QC gating all read ``obs``/``obsm`` and were fine, while every gene-level
result -- knockdown, DE, cluster markers, HVG selection -- was reading a
different gene than the one it named.

What made it survivable was that nothing ever asked an obvious question. ACTB
was detected in 0.02% of cells and EEF1A1 in none of them. Two lines of
arithmetic would have caught it before the first figure was drawn.

Two checks, both cheap:

1. **Housekeeping detection.** A handful of genes that are expressed in
   essentially every cell of every human tissue. If they are not, the matrix is
   not what its labels claim.
2. **var statistics cross-check.** When ``var['n_cells_by_counts']`` is present
   -- scanpy writes it -- it was computed when the labels were aligned. If it
   disagrees with the matrix, the two have since diverged.

Neither check can prove a matrix is correct. Both can prove one is wrong, which
is the useful direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Expressed in ~all cells of ~all human samples. Deliberately a mix of
# ribosomal, cytoskeletal, MHC and housekeeping enzymes, so that a matrix
# missing one class still trips the others.
HUMAN_HOUSEKEEPING = (
    "ACTB", "GAPDH", "B2M", "EEF1A1", "RPL13A", "RPLP0", "PPIA", "TPT1",
    "RPS18", "RPL10", "UBC", "HSP90AB1", "PGK1", "TUBB", "MALAT1",
)
MOUSE_HOUSEKEEPING = (
    "Actb", "Gapdh", "B2m", "Eef1a1", "Rpl13a", "Rplp0", "Ppia", "Tpt1",
    "Rps18", "Rpl10", "Ubc", "Hsp90ab1", "Pgk1", "Tubb5", "Malat1",
)


@dataclass
class MatrixCheck:
    """Outcome of the input plausibility checks."""

    ok: bool
    checked: int = 0
    median_detection: float = float("nan")
    probes: pd.DataFrame | None = None
    var_stat_agreement: float | None = None
    notes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "plausible" if self.ok else "IMPLAUSIBLE"


def _densify(x) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=float)


def check_expression_matrix(
    X,
    var_names: Sequence[str],
    var: pd.DataFrame | None = None,
    min_detection: float = 0.30,
    min_probes_found: int = 3,
    max_cells: int = 20000,
    random_state: int = 0,
) -> MatrixCheck:
    """Is this matrix plausibly the gene expression it claims to be?

    ``min_detection`` is the median detection rate required across whichever
    housekeeping probes are present. It is set well below the ~0.9 a real
    matrix gives, so that a shallow or heavily-filtered experiment does not
    trip it; the failure this catches produced 0.0002.
    """
    names = [str(v) for v in var_names]
    pos = {n: i for i, n in enumerate(names)}
    pos_upper = {n.upper(): i for i, n in enumerate(names)}

    probes = [g for g in HUMAN_HOUSEKEEPING if g in pos]
    if len(probes) < min_probes_found:
        probes = [g for g in MOUSE_HOUSEKEEPING if g in pos]
    if len(probes) < min_probes_found:
        probes = [
            g for g in HUMAN_HOUSEKEEPING if g.upper() in pos_upper
        ]
        pos = pos_upper
        probes = [g.upper() for g in probes]

    res = MatrixCheck(ok=True)
    if len(probes) < min_probes_found:
        res.notes.append(
            f"Only {len(probes)} housekeeping gene(s) were found in var_names, "
            f"so matrix plausibility could not be checked. This is expected for "
            f"a non-human/mouse reference or a pre-subset feature list."
        )
        return res

    n_cells = int(X.shape[0])
    if n_cells > max_cells:
        rng = np.random.default_rng(random_state)
        rows = np.sort(rng.choice(n_cells, size=max_cells, replace=False))
        Xs = X[rows]
    else:
        Xs = X
    n_used = int(Xs.shape[0])

    rows_out = []
    for g in probes:
        j = pos[g]
        col = _densify(Xs[:, [j]]).ravel()
        rows_out.append({"gene": g, "pct_detected": 100.0 * float((col > 0).mean())})
    df = pd.DataFrame(rows_out).sort_values("pct_detected")
    res.probes = df
    res.checked = len(df)
    med = float(np.median(df["pct_detected"])) / 100.0
    res.median_detection = med

    if med < min_detection:
        res.ok = False
        worst = df.head(5)
        res.failures.append(
            f"Housekeeping genes are not detected at plausible rates: median "
            f"{100 * med:.2f}% across {len(df)} probes in {n_used:,} cells "
            f"(expected >{100 * min_detection:.0f}%). Least detected: "
            + ", ".join(f"{r.gene} {r.pct_detected:.2f}%"
                        for r in worst.itertuples())
            + ". The matrix is not the gene expression its var_names describe "
              "-- most often the columns are permuted relative to the labels, "
              "or X holds something other than counts."
        )

    # --- cross-check against scanpy's own per-gene statistics --------------
    if var is not None and "n_cells_by_counts" in var.columns:
        claimed = pd.to_numeric(
            var["n_cells_by_counts"], errors="coerce"
        ).to_numpy(dtype=float)
        idx = [pos[g] for g in probes]
        # Scale the claimed full-object counts to the sampled cell count.
        scale = n_used / max(n_cells, 1)
        expect = claimed[idx] * scale
        got = df.set_index("gene").loc[probes, "pct_detected"].to_numpy() / 100.0 * n_used
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = got / np.where(expect > 0, expect, np.nan)
        agree = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float("nan")
        res.var_stat_agreement = agree
        if np.isfinite(agree) and not (0.7 < agree < 1.4):
            res.ok = False
            res.failures.append(
                f"var['n_cells_by_counts'] disagrees with the matrix: measured "
                f"detection is {agree:.2f}x what var claims for the same genes. "
                f"Those statistics were written when the labels matched the "
                f"columns, so a disagreement means they no longer do."
            )
        elif np.isfinite(agree):
            res.notes.append(
                f"var['n_cells_by_counts'] agrees with the matrix "
                f"(ratio {agree:.2f}), so labels and columns are consistent."
            )

    if res.ok:
        res.notes.append(
            f"Matrix plausibility OK: {len(df)} housekeeping genes detected in "
            f"a median {100 * med:.1f}% of cells."
        )
    return res


def format_failure(res: MatrixCheck) -> str:
    """The message shown when the pipeline refuses to run."""
    lines = [
        "Input expression matrix failed plausibility checks.",
        "",
    ]
    lines += [f"  * {f}" for f in res.failures]
    if res.probes is not None:
        lines += ["", "  housekeeping detection:"]
        lines += [
            f"    {r.gene:<10} {r.pct_detected:6.2f}%"
            for r in res.probes.itertuples()
        ]
    lines += [
        "",
        "  Running anyway would produce a complete report in which every",
        "  gene-level result names one gene and reports another. Diagnose with",
        "  diagnose_what_is_X.py, or pass --skip-input-check to override.",
    ]
    return "\n".join(lines)
