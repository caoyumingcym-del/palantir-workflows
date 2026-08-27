"""
A minimal duck-typed AnnData stand-in, for testing modality splitting without
requiring the single-cell stack.

This exists because ``modalities.split_modalities`` is where a lot of the
original pipeline's bugs lived (unresolvable feature names, silent
zero-feature placeholders, all-features-classified-as-guides), and those bugs
are only reachable through the AnnData API.  Testing them against real AnnData
would make the test suite depend on scanpy; testing them against this
stand-in exercises exactly the same code paths in ``split_modalities``.

It implements only what ``modalities`` touches: ``X``, ``obs``, ``var``,
``obsm``, ``uns``, ``n_obs``, ``n_vars``, column subsetting and ``.copy()``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FakeAnnData:
    def __init__(
        self,
        X: np.ndarray,
        obs: pd.DataFrame | None = None,
        var: pd.DataFrame | None = None,
        obsm: dict | None = None,
        uns: dict | None = None,
    ):
        self.X = np.asarray(X, dtype=float)
        n, p = self.X.shape
        self.obs = (
            obs
            if obs is not None
            else pd.DataFrame(index=[f"cell_{i}" for i in range(n)])
        )
        self.var = (
            var
            if var is not None
            else pd.DataFrame(index=[f"gene_{j}" for j in range(p)])
        )
        self.obsm = dict(obsm or {})
        self.uns = dict(uns or {})
        self.layers: dict = {}
        # scanpy calls view_to_actual(), which reads .is_view, on almost every
        # preprocessing function. Without this the stand-in blows up as soon as
        # scanpy is installed -- which is exactly what happened the first time
        # these tests ran in an environment that had it.
        self.is_view = False
        self.raw = None

    @property
    def n_obs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_vars(self) -> int:
        return int(self.X.shape[1])

    @property
    def obs_names(self):
        return self.obs.index

    @property
    def var_names(self):
        return self.var.index

    @property
    def shape(self):
        return self.X.shape

    def __getitem__(self, key):
        rows, cols = (key if isinstance(key, tuple) else (key, slice(None)))
        r = self._idx(rows, self.n_obs)
        c = self._idx(cols, self.n_vars)
        return FakeAnnData(
            X=self.X[np.ix_(r, c)],
            obs=self.obs.iloc[r].copy(),
            var=self.var.iloc[c].copy(),
            obsm={k: np.asarray(v)[r] for k, v in self.obsm.items()},
            uns=dict(self.uns),
        )

    @staticmethod
    def _idx(key, n):
        if isinstance(key, slice):
            return np.arange(n)[key]
        arr = np.asarray(key)
        if arr.dtype == bool:
            return np.flatnonzero(arr)
        return arr.astype(int)

    def _inplace_subset_var(self, mask):
        """scanpy mutates var in place in a few code paths."""
        idx = self._idx(mask, self.n_vars)
        self.X = self.X[:, idx]
        self.var = self.var.iloc[idx].copy()

    def copy(self):
        return FakeAnnData(
            X=self.X.copy(),
            obs=self.obs.copy(),
            var=self.var.copy(),
            obsm={k: np.asarray(v).copy() for k, v in self.obsm.items()},
            uns=dict(self.uns),
        )

    def __repr__(self):
        return f"FakeAnnData({self.n_obs} x {self.n_vars})"


def make_combined(
    n_cells: int = 200,
    n_genes: int = 50,
    n_guides: int = 8,
    n_htos: int = 4,
    seed: int = 0,
    layout: str = "feature_types",
) -> FakeAnnData:
    """Build a fake object with guides/HTOs in one of the four supported layouts.

    ``layout`` is one of:
      ``feature_types``  -- all modalities in X, distinguished by var column
      ``obsm``           -- guides/HTOs in obsm with names in uns
      ``obsm_dict_uns``  -- names stored as an index->name dict (the case the
                            original silently got wrong)
      ``obs_cols``       -- HTOs flattened into obs columns
      ``gex_only``       -- no guides, no HTOs
    """
    rng = np.random.default_rng(seed)
    gex = rng.poisson(1.0, size=(n_cells, n_genes)).astype(float)
    guides = np.zeros((n_cells, n_guides))
    # one dominant guide per cell plus ambient noise
    winners = rng.integers(0, n_guides, size=n_cells)
    guides[np.arange(n_cells), winners] = rng.poisson(60, size=n_cells)
    guides += rng.poisson(0.4, size=(n_cells, n_guides))
    htos = rng.poisson(3, size=(n_cells, n_htos)).astype(float)
    hwin = rng.integers(0, n_htos, size=n_cells)
    htos[np.arange(n_cells), hwin] += rng.poisson(300, size=n_cells)

    gene_names = [f"GENE{j}" for j in range(n_genes - 3)] + ["MT-CO1", "RPS6", "RPL13"]
    guide_names = [f"TGT{i//2}_ENSG{i//2:011d}_sg{i%2+1}" for i in range(n_guides)]
    hto_names = [f"HTO{i+1}" for i in range(n_htos)]
    obs = pd.DataFrame(
        {"sample": rng.choice(["s1", "s2"], n_cells)},
        index=[f"BC{i:05d}" for i in range(n_cells)],
    )

    if layout == "feature_types":
        X = np.hstack([gex, guides, htos])
        var = pd.DataFrame(
            {
                "feature_types": (
                    ["Gene Expression"] * n_genes
                    + ["CRISPR Guide Capture"] * n_guides
                    + ["Antibody Capture"] * n_htos
                )
            },
            index=gene_names + guide_names + hto_names,
        )
        return FakeAnnData(X, obs=obs, var=var)

    if layout == "obsm":
        var = pd.DataFrame(index=gene_names)
        return FakeAnnData(
            gex, obs=obs, var=var,
            obsm={"gRNA_counts": guides, "HTO_counts": htos},
            uns={"gRNA_features": np.array(guide_names),
                 "HTO_features": np.array(hto_names)},
        )

    if layout == "obsm_dict_uns":
        var = pd.DataFrame(index=gene_names)
        return FakeAnnData(
            gex, obs=obs, var=var,
            obsm={"gRNA_counts": guides, "HTO_counts": htos},
            uns={
                "gRNA_features": {i: n for i, n in enumerate(guide_names)},
                "HTO_features": {i: n for i, n in enumerate(hto_names)},
            },
        )

    if layout == "obs_cols":
        var = pd.DataFrame(index=gene_names)
        o = obs.copy()
        for i, name in enumerate(hto_names):
            o[f"prot:hash.{name}"] = htos[:, i]
            o[f"prot:hash.{name}_CLR"] = np.log1p(htos[:, i])   # must be excluded
        return FakeAnnData(
            gex, obs=o, var=var,
            obsm={"gRNA_counts": guides},
            uns={"gRNA_features": np.array(guide_names)},
        )

    if layout == "gex_only":
        return FakeAnnData(gex, obs=obs, var=pd.DataFrame(index=gene_names))

    raise ValueError(f"unknown layout {layout!r}")
