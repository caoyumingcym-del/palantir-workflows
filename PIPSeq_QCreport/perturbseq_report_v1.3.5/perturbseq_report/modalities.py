"""
Locating the GEX / guide / hashtag submatrices inside one .h5ad.

Perturb-seq h5ads store the non-GEX modalities in at least four different
places depending on which tool wrote them, so this is genuinely fiddly.  Two
design decisions make it manageable:

1. **Guide and hashtag matrices come out as plain numpy arrays plus a name
   list**, not AnnData objects.  Only the GEX branch needs scanpy.  Everything
   downstream of guide/HTO extraction is therefore testable without a
   single-cell stack installed, and the extraction logic itself can be
   exercised against a lightweight stand-in.

2. **Every extraction records where it found the data** (``Modality.source``)
   and that string goes into the report.  The original silently substituted a
   zero-feature placeholder AnnData when a modality was not found, so an
   experiment where the guide matrix was stored under an unexpected key
   produced a report full of empty guide panels with no indication that the
   data existed but had not been located.

Fixed from the original
-----------------------
``_resolve_feature_names_from_uns`` contained an unreachable branch::

    names = list(raw.keys()) if len(raw) == n_features else list(raw.values())

For any dict, ``len(raw.keys()) == len(raw.values()) == len(raw)``, so the
condition can never distinguish a name-keyed dict from an index-keyed one --
it always took ``.keys()``.  An h5ad storing ``uns[key] = {0: "GuideA",
1: "GuideB"}`` therefore got the integers ``0, 1`` as its guide names.  Here
the dict's keys are inspected to decide.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from .config import ModalityConfig
from .stats import to_dense


class AnnDataLike(Protocol):
    """The subset of the AnnData API this module needs."""

    X: Any
    obs: pd.DataFrame
    var: pd.DataFrame
    obsm: Any
    uns: Any
    n_obs: int
    n_vars: int


@dataclass
class Modality:
    """A non-GEX count matrix with named features."""

    kind: str                        # "guide" | "hto"
    X: np.ndarray                    # cells x features, dense float
    names: list[str]
    source: str                      # where it was found, for the report
    obs_names: list[str] = field(default_factory=list)

    @property
    def present(self) -> bool:
        return self.X.size > 0 and len(self.names) > 0

    @property
    def n_features(self) -> int:
        return len(self.names)

    @property
    def n_cells(self) -> int:
        return int(self.X.shape[0]) if self.X.ndim == 2 else 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.X, columns=self.names,
            index=self.obs_names or range(self.n_cells),
        )

    def subset_cells(self, mask: np.ndarray) -> "Modality":
        mask = np.asarray(mask)
        if mask.dtype == bool:
            idx = np.flatnonzero(mask)
        else:
            idx = mask.astype(int)
        return Modality(
            kind=self.kind, X=self.X[idx], names=list(self.names),
            source=self.source,
            obs_names=[self.obs_names[i] for i in idx] if self.obs_names else [],
        )

    def reindex_cells(self, target_obs_names: Sequence[str]) -> "Modality":
        """Align to another object's cell order by barcode.

        Cells absent from this modality get zeros.  This is how a guide matrix
        stored in ``obsm`` (already aligned) and one reconstructed from ``obs``
        columns (possibly not) are made interchangeable downstream.
        """
        if not self.obs_names:
            if self.n_cells == len(target_obs_names):
                return self
            raise ValueError(
                f"{self.kind} modality has {self.n_cells} cells but no barcodes, "
                f"and the target has {len(target_obs_names)}; cannot align."
            )
        pos = {b: i for i, b in enumerate(self.obs_names)}
        out = np.zeros((len(target_obs_names), self.n_features), dtype=np.float64)
        for j, b in enumerate(target_obs_names):
            i = pos.get(b)
            if i is not None:
                out[j] = self.X[i]
        return Modality(
            kind=self.kind, X=out, names=list(self.names), source=self.source,
            obs_names=list(target_obs_names),
        )


def empty_modality(kind: str, reason: str, n_cells: int = 0) -> Modality:
    return Modality(
        kind=kind, X=np.zeros((n_cells, 0), dtype=np.float64), names=[],
        source=reason,
    )


# ===========================================================================
# Feature-name resolution
# ===========================================================================
def resolve_feature_names(
    uns: Any, candidate_keys: Sequence[str], n_features: int, kind: str
) -> tuple[list[str] | None, str]:
    """Find the vector of feature names for an obsm matrix.

    Returns ``(names, description)``; ``names`` is None if nothing usable was
    found, in which case the caller should synthesise placeholders *and say so*
    in the report -- unnamed guides make the guide section nearly useless, so
    the reader needs to know.
    """
    if uns is None:
        return None, "no uns"
    for key in candidate_keys:
        try:
            raw = uns[key]
        except (KeyError, TypeError, IndexError):
            continue
        names = _coerce_name_vector(raw, n_features)
        if names is not None:
            return names, f"uns[{key!r}]"
    return None, f"none of uns{list(candidate_keys)} usable"


def _coerce_name_vector(raw: Any, n_features: int) -> list[str] | None:
    """Turn whatever is in uns into a list of n_features names."""
    if raw is None:
        return None

    # pandas objects
    if isinstance(raw, pd.DataFrame):
        for col in ("name", "names", "id", "gene_ids", "feature_name"):
            if col in raw.columns and len(raw) == n_features:
                return [str(x) for x in raw[col].tolist()]
        if len(raw) == n_features:
            return [str(x) for x in raw.iloc[:, 0].tolist()]
        return None
    if isinstance(raw, pd.Series):
        return [str(x) for x in raw.tolist()] if len(raw) == n_features else None

    # dict: decide by inspecting the KEYS, not by comparing len(keys) to
    # len(values), which is always equal and was the original's bug.
    if isinstance(raw, dict):
        keys = list(raw.keys())
        if len(keys) != n_features:
            return None
        keys_are_indices = all(
            isinstance(k, (int, np.integer))
            or (isinstance(k, str) and k.isdigit())
            for k in keys
        )
        if keys_are_indices:
            # index -> name: sort by index and take the values
            try:
                ordered = sorted(raw.items(), key=lambda kv: int(kv[0]))
            except (TypeError, ValueError):
                return None
            return [str(v) for _, v in ordered]
        return [str(k) for k in keys]

    # array-like
    try:
        arr = np.asarray(raw).ravel()
    except Exception:
        return None
    if arr.size != n_features:
        return None
    return [
        (x.decode() if isinstance(x, bytes) else str(x)) for x in arr.tolist()
    ]


# ===========================================================================
# obsm / obs extraction
# ===========================================================================
def _from_obsm(
    adata: AnnDataLike,
    obsm_keys: Sequence[str],
    uns_keys: Sequence[str],
    kind: str,
) -> Modality | None:
    obsm = getattr(adata, "obsm", None)
    if obsm is None:
        return None
    for key in obsm_keys:
        try:
            M = obsm[key]
        except (KeyError, TypeError, IndexError):
            continue
        X = to_dense(M)
        if X.ndim != 2 or X.shape[1] == 0:
            continue
        names, how = resolve_feature_names(
            getattr(adata, "uns", None), uns_keys, X.shape[1], kind
        )
        source = f"obsm[{key!r}]"
        if names is None:
            # Some writers put names on the obsm DataFrame's own columns.
            if isinstance(M, pd.DataFrame):
                names = [str(c) for c in M.columns]
                source += " (names from DataFrame columns)"
            else:
                names = [f"{kind}_{i}" for i in range(X.shape[1])]
                source += f" (NAMES NOT FOUND -- {how}; using placeholders)"
        else:
            source += f", names from {how}"
        return Modality(
            kind=kind, X=X, names=names, source=source,
            obs_names=[str(x) for x in adata.obs.index],
        )
    return None


def _from_obs_columns(
    adata: AnnDataLike, cfg: ModalityConfig, kind: str
) -> Modality | None:
    """Reconstruct a hashtag matrix that was flattened into obs columns."""
    if kind != "hto":
        return None
    obs = adata.obs
    cols: list[str] = []
    for c in obs.columns:
        name = str(c)
        if not any(name.startswith(p) for p in cfg.hto_obs_prefixes):
            continue
        if any(name.endswith(s) for s in cfg.hto_obs_exclude_suffixes):
            continue
        if not pd.api.types.is_numeric_dtype(obs[c]):
            continue
        cols.append(c)
    if not cols:
        return None

    X = obs[cols].to_numpy(dtype=np.float64)
    source = f"obs columns matching {list(cfg.hto_obs_prefixes)}"
    if cfg.hto_obs_cols_are_log1p:
        X = np.rint(np.expm1(np.clip(X, 0, None)))
        source += " (inverted log1p)"

    def strip(name: str) -> str:
        for p in cfg.hto_obs_prefixes:
            if name.startswith(p):
                return name[len(p):] or name
        return name

    return Modality(
        kind=kind, X=X, names=[strip(str(c)) for c in cols], source=source,
        obs_names=[str(x) for x in obs.index],
    )


def _from_var_feature_types(
    adata: AnnDataLike, cfg: ModalityConfig
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Boolean masks over var for guide and hashtag features."""
    var = adata.var
    for col in cfg.feature_type_cols:
        if col not in var.columns:
            continue
        ft = var[col].astype(str).str.lower()
        guide = ft.apply(
            lambda s: any(t in s for t in cfg.guide_feature_type_tokens)
        ).to_numpy()
        hto = ft.apply(
            lambda s: any(t in s for t in cfg.hto_feature_type_tokens)
        ).to_numpy()
        if guide.any() or hto.any():
            return guide, hto, f"var[{col!r}]"
    return None


def _guide_like_var_names(
    adata: AnnDataLike, cfg: ModalityConfig, guide_regexes: Sequence[str]
) -> np.ndarray:
    """Last-resort guide detection from var_names looking like guide IDs."""
    pats = [re.compile(p) for p in guide_regexes]
    names = [str(v) for v in adata.var.index]
    return np.array([any(p.match(n) for n in [nm] for p in pats) for nm in names])


# ===========================================================================
# Public entry point
# ===========================================================================
@dataclass
class SplitResult:
    """Outcome of splitting one h5ad into modalities."""

    gex: Any                    # AnnData restricted to gene-expression features
    guide: Modality
    hto: Modality
    notes: list[str] = field(default_factory=list)
    gex_source: str = ""

    @property
    def has_guides(self) -> bool:
        return self.guide.present

    @property
    def has_hto(self) -> bool:
        return self.hto.present


def split_modalities(
    adata: AnnDataLike,
    cfg: ModalityConfig,
    guide_id_regexes: Sequence[str] = (),
    counts: Any = None,
) -> SplitResult:
    """Split one AnnData into GEX (AnnData) plus guide and HTO (plain matrices).

    Resolution order, each recorded in ``notes``:

    ``counts`` overrides ``adata.X`` when extracting the guide and hashtag
    matrices. This matters on an already-analysed object: its ``X`` holds
    normalised, log-transformed values, and guide "UMI counts" read from it
    would be log values of order 0-5. Every downstream guide threshold (">10
    reads", purity) would then reject almost every cell -- silently, since a
    low assignment rate looks like a failed experiment rather than a bug.

      1. ``var['feature_types']`` -- the CellRanger/DRAGEN convention.
      2. ``obsm`` matrices with names from ``uns``.
      3. ``obs`` columns, for hashtags flattened into obs.
      4. Guide-ID-shaped ``var_names``, as a last resort.

    The GEX matrix is whatever is left after removing guide and hashtag
    features.  A subset is always taken as an explicit copy, so no downstream
    stage can accidentally write through a view into the caller's object -- a
    class of bug the original was exposed to, since it mutated ``adata`` in
    place across notebook cells.
    """
    notes: list[str] = []
    n_obs = int(adata.n_obs)
    obs_names = [str(x) for x in adata.obs.index]

    guide_mask = np.zeros(int(adata.n_vars), dtype=bool)
    hto_mask = np.zeros(int(adata.n_vars), dtype=bool)
    gex_source = "all var features"

    def _counts_source():
        return adata if counts is None else _CountsView(adata, counts)

    ft = _from_var_feature_types(adata, cfg)
    if ft is not None:
        guide_mask, hto_mask, where = ft
        gex_source = f"var features not flagged guide/hashtag in {where}"
        notes.append(
            f"Feature types read from {where}: "
            f"{int(guide_mask.sum())} guide, {int(hto_mask.sum())} hashtag, "
            f"{int((~guide_mask & ~hto_mask).sum())} gene-expression features."
        )

    # --- guide -------------------------------------------------------------
    guide: Modality
    if guide_mask.any():
        guide = Modality(
            kind="guide",
            X=to_dense(_subset_X(_counts_source(), guide_mask)),
            names=[str(v) for v in adata.var.index[guide_mask]],
            source=f"{gex_source.split(' in ')[-1]} (in-matrix guide features)",
            obs_names=obs_names,
        )
    else:
        found = _from_obsm(
            adata, cfg.guide_obsm_keys, cfg.guide_feature_uns_keys, "guide"
        )
        if found is None and guide_id_regexes:
            m = _guide_like_var_names(adata, cfg, guide_id_regexes)
            if m.any() and not m.all():
                guide_mask = m
                found = Modality(
                    kind="guide", X=to_dense(_subset_X(_counts_source(), m)),
                    names=[str(v) for v in adata.var.index[m]],
                    source="var_names matching guide-ID patterns",
                    obs_names=obs_names,
                )
                notes.append(
                    f"No feature-type column or guide obsm key found; "
                    f"{int(m.sum())} var_names matched guide-ID patterns and were "
                    f"treated as guides."
                )
        guide = found or empty_modality(
            "guide",
            f"not found (checked var feature types, obsm"
            f"{list(cfg.guide_obsm_keys)}, and guide-ID-shaped var_names)",
            n_obs,
        )

    # --- hashtags ----------------------------------------------------------
    if hto_mask.any():
        hto = Modality(
            kind="hto",
            X=to_dense(_subset_X(_counts_source(), hto_mask)),
            names=[str(v) for v in adata.var.index[hto_mask]],
            source="in-matrix hashtag features",
            obs_names=obs_names,
        )
    else:
        hto = (
            _from_obsm(adata, cfg.hto_obsm_keys, cfg.hto_feature_uns_keys, "hto")
            or _from_obs_columns(adata, cfg, "hto")
            or empty_modality(
                "hto",
                f"not found (checked var feature types, obsm"
                f"{list(cfg.hto_obsm_keys)}, and obs columns prefixed "
                f"{list(cfg.hto_obs_prefixes)})",
                n_obs,
            )
        )

    # --- recover placeholder feature names from var, when possible ---------
    #
    # Runs BEFORE the notes below, so a successful recovery does not also emit
    # an "names could not be resolved" warning that is no longer true.
    #
    # An obsm matrix whose names could not be resolved gets placeholders
    # ("guide_0", "guide_1", ...). That leaves per-feature panels unlabelled,
    # and -- worse -- makes the GEX back-fill impossible, because there is no
    # name to match against var. When var contains exactly as many
    # guide-ID-shaped names as the matrix has columns, those are almost
    # certainly the same features in the same order, so adopt them.
    if guide.present and guide_id_regexes:
        placeholder = all(
            re.fullmatch(r"guide_\d+", str(n)) for n in guide.names
        )
        if placeholder:
            m = _guide_like_var_names(adata, cfg, guide_id_regexes)
            if int(m.sum()) == guide.n_features and int(m.sum()) > 0:
                recovered = [str(v) for v in adata.var.index[m]]
                guide = Modality(
                    kind=guide.kind, X=guide.X, names=recovered,
                    source=guide.source + " (names recovered from var)",
                    obs_names=guide.obs_names,
                )
                notes.append(
                    f"{len(recovered)} guide feature names were unresolvable "
                    f"from uns and have been recovered from var_names matching "
                    f"guide-ID patterns. Check the first few against the "
                    f"guide-ID table: the match assumes column order agrees."
                )

    for mod in (guide, hto):
        if mod.present:
            notes.append(f"{mod.kind}: {mod.n_features} features from {mod.source}")
            if any(re.fullmatch(rf"{mod.kind}_\d+", n) for n in mod.names):
                notes.append(
                    f"WARNING: {mod.kind} feature names could not be resolved and "
                    f"are placeholders. Per-feature panels will be unlabelled. "
                    f"Check the h5ad's uns keys."
                )
        else:
            notes.append(f"{mod.kind}: {mod.source}")

    # --- remove guide/hashtag features from GEX, however they were found ---
    #
    # THIS IS THE FIX for the v1.1.0 bug that put guide IDs in the cluster
    # marker lists. The masks above are only populated by the feature-types
    # branch and the guide-ID-shaped-var_names branch. When the matrices are
    # resolved from `obsm` instead -- as they are for any object with
    # obsm['gRNA_counts'] and no feature_types column, which is exactly
    # MDL-1856 -- both masks stay all-False, `gex_mask` comes out all-True, and
    # all 321 guide features remain in the GEX matrix. They then feed HVG
    # selection, PCA, Leiden and the marker test, so clusters end up defined by
    # which guide a cell carries rather than by its transcriptome.
    #
    # Back-filling by feature NAME closes the gap for every resolution path at
    # once, including any added later.
    vn = pd.Index([str(v) for v in adata.var.index])
    for mod, mask_name in ((guide, "guide"), (hto, "hto")):
        if not mod.present:
            continue
        also_in_var = np.asarray(vn.isin({str(n) for n in mod.names}))
        if not also_in_var.any():
            continue
        already = guide_mask if mask_name == "guide" else hto_mask
        newly = int((also_in_var & ~already).sum())
        if mask_name == "guide":
            guide_mask = guide_mask | also_in_var
        else:
            hto_mask = hto_mask | also_in_var
        if newly:
            notes.append(
                f"WARNING: {newly} {mask_name} feature(s) were also present in "
                f"var and have been removed from the gene-expression matrix. "
                f"They were resolved from {mod.source}, so no feature-type "
                f"column marked them as non-GEX. Left in place they contaminate "
                f"highly-variable-gene selection, the embedding and the cluster "
                f"marker lists."
            )

    # --- gene expression ---------------------------------------------------
    gex_mask = ~(guide_mask | hto_mask)
    if not gex_mask.any():
        raise ValueError(
            "Every feature in the h5ad was classified as a guide or hashtag, "
            "leaving no gene-expression features. This usually means the "
            "feature-type tokens are matching too broadly; check "
            "ModalityConfig.guide_feature_type_tokens / hto_feature_type_tokens."
        )
    gex = adata[:, gex_mask].copy() if not gex_mask.all() else adata.copy()

    if guide.present:
        guide = guide.reindex_cells(obs_names)
    if hto.present:
        hto = hto.reindex_cells(obs_names)

    return SplitResult(
        gex=gex, guide=guide, hto=hto, notes=notes, gex_source=gex_source
    )


class _CountsView:
    """Minimal shim presenting an alternative matrix as ``.X``."""

    __slots__ = ("_adata", "X")

    def __init__(self, adata: Any, counts: Any):
        self._adata = adata
        self.X = counts

    def __getattr__(self, name):
        return getattr(self._adata, name)


def _subset_X(adata: AnnDataLike, mask: np.ndarray) -> Any:
    """Column-subset the count matrix without materialising the whole thing."""
    X = adata.X
    idx = np.flatnonzero(mask)
    try:
        return X[:, idx]
    except Exception:
        return to_dense(X)[:, idx]


def resolve_column(
    obs: pd.DataFrame, candidates: Sequence[str], required: bool = False,
    what: str = "column",
) -> str | None:
    """First candidate column actually present in obs."""
    for c in candidates:
        if c in obs.columns:
            return c
    if required:
        raise ValueError(
            f"None of the candidate {what} names {list(candidates)} is present in "
            f"obs. Available: {list(obs.columns)[:40]}"
        )
    return None
