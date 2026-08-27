"""
Inspect an .h5ad without loading it.

Reads only HDF5 metadata via h5py, so it costs megabytes and milliseconds no
matter how large the file is. This exists because the expensive failure mode --
the process being killed during ``read_h5ad`` with no output at all -- gives you
nothing to diagnose from. Running this first tells you whether the file can fit
in memory before you try.

    python run_perturbseq_report.py --inspect path/to/file.h5ad

Reports the matrix shape, whether X is stored sparse or dense, the dtype, the
non-zero count, memory estimates for both layouts, and where the guide and
hashtag matrices actually live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MatrixInfo:
    encoding: str            # "csr_matrix" | "csc_matrix" | "dense" | "unknown"
    shape: tuple[int, int] | None
    dtype: str
    nnz: int | None

    @property
    def is_sparse(self) -> bool:
        return self.encoding in ("csr_matrix", "csc_matrix")

    @property
    def dense_bytes(self) -> int:
        if not self.shape:
            return 0
        return self.shape[0] * self.shape[1] * 8

    @property
    def sparse_bytes(self) -> int:
        if self.nnz is None:
            return 0
        # float64 data + int32/int64 indices, plus the indptr vector.
        return self.nnz * 12

    @property
    def load_bytes(self) -> int:
        """What reading this file into memory will actually cost."""
        return self.sparse_bytes if self.is_sparse else self.dense_bytes

    @property
    def density(self) -> float | None:
        if self.nnz is None or not self.shape:
            return None
        total = self.shape[0] * self.shape[1]
        return self.nnz / total if total else None


@dataclass
class H5adInfo:
    path: Path
    file_bytes: int
    X: MatrixInfo
    n_obs: int | None = None
    n_vars: int | None = None
    obs_columns: list[str] = field(default_factory=list)
    var_columns: list[str] = field(default_factory=list)
    obsm: dict[str, Any] = field(default_factory=dict)
    layers: list[str] = field(default_factory=list)
    uns_keys: list[str] = field(default_factory=list)
    raw_present: bool = False
    errors: list[str] = field(default_factory=list)


def _read_matrix_info(node) -> MatrixInfo:
    """Describe /X (or a layer) from HDF5 metadata alone."""
    import h5py

    attrs = dict(getattr(node, "attrs", {}))

    def _decode(v):
        if isinstance(v, bytes):
            return v.decode()
        return v

    encoding = _decode(attrs.get("encoding-type", "")) or ""

    if isinstance(node, h5py.Dataset):
        shape = tuple(int(x) for x in node.shape) if node.shape else None
        return MatrixInfo(
            encoding="dense",
            shape=shape if shape and len(shape) == 2 else None,
            dtype=str(node.dtype),
            nnz=None,
        )

    # Sparse: a group holding data / indices / indptr, with shape in attrs.
    shape = attrs.get("shape")
    if shape is not None:
        shape = tuple(int(x) for x in shape)
    elif "data" in node and "indptr" in node:
        shape = None

    nnz = int(node["data"].shape[0]) if "data" in node else None
    dtype = str(node["data"].dtype) if "data" in node else "?"
    return MatrixInfo(
        encoding=encoding or ("csr_matrix" if "indptr" in node else "unknown"),
        shape=shape if shape and len(shape) == 2 else None,
        dtype=dtype,
        nnz=nnz,
    )


def _group_columns(node) -> list[str]:
    """Column names of an AnnData obs/var group."""
    import h5py

    if node is None:
        return []
    if isinstance(node, h5py.Dataset):
        # Old-style structured array
        return list(node.dtype.names or [])
    out = []
    for key in node.keys():
        if key.startswith("_"):
            continue
        out.append(key)
    return out


def inspect(path: str | Path) -> H5adInfo:
    """Describe an .h5ad using only its metadata."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"h5ad not found: {p}")

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "h5py is required to inspect an .h5ad. It ships with anndata; "
            "install it with `pip install h5py`."
        ) from exc

    info = H5adInfo(
        path=p,
        file_bytes=p.stat().st_size,
        X=MatrixInfo("unknown", None, "?", None),
    )

    with h5py.File(p, "r") as f:
        if "X" in f:
            try:
                info.X = _read_matrix_info(f["X"])
            except Exception as exc:
                info.errors.append(f"could not read /X: {exc}")
        else:
            info.errors.append("no /X in this file")

        for key, target in (("obs", "obs_columns"), ("var", "var_columns")):
            if key in f:
                try:
                    setattr(info, target, _group_columns(f[key]))
                except Exception as exc:
                    info.errors.append(f"could not read /{key}: {exc}")

        # Cell and gene counts: prefer the index length, fall back to X's shape.
        for key, attr in (("obs", "n_obs"), ("var", "n_vars")):
            if key not in f:
                continue
            node = f[key]
            try:
                idx_name = node.attrs.get("_index", b"_index")
                if isinstance(idx_name, bytes):
                    idx_name = idx_name.decode()
                if idx_name in node:
                    setattr(info, attr, int(node[idx_name].shape[0]))
            except Exception:
                pass
        if info.n_obs is None and info.X.shape:
            info.n_obs = info.X.shape[0]
        if info.n_vars is None and info.X.shape:
            info.n_vars = info.X.shape[1]

        if "obsm" in f:
            for key in f["obsm"].keys():
                try:
                    node = f["obsm"][key]
                    shape = getattr(node, "shape", None)
                    if shape is None and hasattr(node, "attrs"):
                        shape = node.attrs.get("shape")
                    info.obsm[key] = (
                        tuple(int(x) for x in shape) if shape is not None else "?"
                    )
                except Exception:
                    info.obsm[key] = "?"

        if "layers" in f:
            info.layers = list(f["layers"].keys())
        if "uns" in f:
            info.uns_keys = list(f["uns"].keys())
        info.raw_present = "raw" in f

    return info


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def format_report(info: H5adInfo, available_gb: float | None = None) -> str:
    """Human-readable summary with an explicit feasibility verdict."""
    L: list[str] = []
    L.append(f"h5ad: {info.path}")
    L.append(f"  file on disk      : {_gb(info.file_bytes)}")
    L.append(f"  cells x features  : "
             f"{(info.n_obs or 0):,} x {(info.n_vars or 0):,}")
    L.append(f"  X storage         : "
             f"{'SPARSE (' + info.X.encoding + ')' if info.X.is_sparse else 'DENSE'}")
    L.append(f"  X dtype           : {info.X.dtype}")
    if info.X.nnz is not None:
        density = info.X.density
        L.append(f"  non-zero entries  : {info.X.nnz:,}"
                 + (f"  ({density * 100:.2f}% of the matrix)"
                    if density is not None else ""))
    L.append("")
    L.append("  memory to load X:")
    if info.X.is_sparse:
        L.append(f"    as stored (sparse) : {_gb(info.X.sparse_bytes)}   <-- what "
                 f"you will actually use")
        L.append(f"    if densified       : {_gb(info.X.dense_bytes)}")
    else:
        L.append(f"    as stored (dense)  : {_gb(info.X.dense_bytes)}   <-- what "
                 f"you will actually use")
        L.append("    if converted       : unknown until converted, typically "
                 "5-15x smaller")

    if info.layers:
        L.append("")
        L.append(f"  layers            : {', '.join(info.layers)}  "
                 f"(each is another full matrix)")
    if info.raw_present:
        L.append(f"  .raw present      : yes (another full matrix)")

    L.append("")
    L.append(f"  obsm keys         : "
             f"{', '.join(f'{k} {v}' for k, v in info.obsm.items()) or '(none)'}")
    L.append(f"  uns keys          : {', '.join(info.uns_keys) or '(none)'}")
    L.append(f"  obs columns       : {', '.join(info.obs_columns[:15])}"
             + (" ..." if len(info.obs_columns) > 15 else ""))
    L.append(f"  var columns       : {', '.join(info.var_columns[:15])}"
             + (" ..." if len(info.var_columns) > 15 else ""))

    # ---- verdict -----------------------------------------------------------
    load = info.X.load_bytes
    peak_est = load * 2.5           # copies during processing
    L.append("")
    L.append(f"  estimated peak for a full run: ~{_gb(int(peak_est))} "
             f"(load + working copies)")
    if available_gb is not None:
        L.append(f"  memory available on this machine: {available_gb:.1f} GB")
        if peak_est / 1e9 > available_gb * 0.8:
            L.append("")
            L.append("  VERDICT: this will probably NOT fit. Options, best first:")
            if not info.X.is_sparse:
                L.append("    1. Convert X to sparse (see below). Usually enough "
                         "on its own.")
            L.append("    2. Set embedding.copy_input=false to avoid one copy.")
            L.append("    3. Run on a subset of cells or split by sample.")
            L.append("    4. Request a machine with more memory.")
        else:
            L.append("")
            L.append("  VERDICT: this should fit comfortably.")

    if not info.X.is_sparse:
        L.append("")
        L.append("  X is stored DENSE. Single-cell counts are typically 90-95%")
        L.append("  zeros, so converting is usually a large win. One-off, and it")
        L.append("  needs enough memory to hold the dense matrix once:")
        L.append("")
        L.append("      import anndata as ad, scipy.sparse as sp")
        L.append("      a = ad.read_h5ad('in.h5ad')")
        L.append("      a.X = sp.csr_matrix(a.X)")
        L.append("      a.write_h5ad('out.h5ad')")
        L.append("")
        L.append("  If it will not fit even once, convert in chunks with")
        L.append("  anndata's backed mode, or ask whoever produced the file to")
        L.append("  write it sparse.")

    if info.errors:
        L.append("")
        L.append("  problems reading the file:")
        for e in info.errors:
            L.append(f"    - {e}")
    return "\n".join(L)


def available_memory_gb() -> float | None:
    """Available RAM in GB, or None if it cannot be determined."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6      # kB -> GB
    except OSError:
        pass
    try:
        import os
        return (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / 1e9
    except (ValueError, OSError, AttributeError):
        return None
