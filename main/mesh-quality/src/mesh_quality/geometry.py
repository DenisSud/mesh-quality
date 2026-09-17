"""Geometry-only features for the 3D mesh quality task.

Each object is a triangle mesh: ``vertices`` (n, 3) float32, ``faces`` (m, 3)
int32, per the task spec. The renders hide part of the information that decides a
label (polygon budget, hole count, primitive-ness), so these features are meant
to be used *alongside* image features, not instead of them.

Everything is vectorised; the structural features (boundary/non-manifold edges,
connected components) are computed from the full edge set when the mesh fits
inside ``cap_faces`` and are left as ``nan`` otherwise.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Feature order is part of the cache contract: keep it stable.
FEATURE_NAMES: tuple[str, ...] = (
    "n_verts",
    "n_faces",
    "log_faces",
    "bbox_dx",
    "bbox_dy",
    "bbox_dz",
    "bbox_diag",
    "bbox_flatness",       # smallest extent / largest extent: plane-like objects
    "center_dist",         # bbox center distance from origin
    "unused_vert_frac",
    "edge_len_med",        # normalised by bbox diagonal
    "edge_len_cv",
    "area_med",            # median triangle area / diag^2
    "degenerate_frac",     # zero-area triangles
    "normal_entropy",      # bits, area-weighted face-normal histogram (32x16)
    "normal_max_bin",      # area fraction of the single most populated bin
    "axis_aligned_frac",   # area fraction of faces normal to a bbox axis
    "radial_frac",         # area fraction of faces normal to the radial direction
    "boundary_edge_frac",  # edges used by exactly one triangle (holes, open surfaces)
    "nonmanifold_edge_frac",
    "n_components",
    "largest_component_frac",
    "faces_per_component",  # n_faces / n_components, log10
)

_NAN_FEATURES = ("boundary_edge_frac", "nonmanifold_edge_frac", "n_components", "largest_component_frac", "faces_per_component")


def _empty_features() -> dict[str, float]:
    return {name: float("nan") for name in FEATURE_NAMES}


def mesh_features(vertices: np.ndarray, faces: np.ndarray, cap_faces: int = 2_000_000) -> dict[str, float]:
    """Compute geometry features for a single mesh.

    ``cap_faces`` bounds the cost of the structural features; above it the
    ``_NAN_FEATURES`` entries stay nan and the rest of the mesh is still scored
    from a strided face sample.
    """
    out = _empty_features()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    nv, nf = len(v), len(f)
    out["n_verts"] = float(nv)
    out["n_faces"] = float(nf)
    out["log_faces"] = float(np.log10(nf + 1.0))
    if nv == 0 or nf == 0:
        return out

    lo, hi = v.min(axis=0), v.max(axis=0)
    extent = hi - lo
    diag = float(np.linalg.norm(extent))
    if diag <= 0:
        diag = 1.0
    out["bbox_dx"], out["bbox_dy"], out["bbox_dz"] = (float(x) for x in extent)
    out["bbox_diag"] = diag
    out["bbox_flatness"] = float(extent.min() / max(extent.max(), 1e-12))
    out["center_dist"] = float(np.linalg.norm((lo + hi) / 2.0))

    used = np.zeros(nv, dtype=bool)
    used[f.ravel()] = True
    out["unused_vert_frac"] = float(1.0 - used.mean())

    stride = max(1, nf // 500_000)
    tri = v[f[::stride]]
    e01 = tri[:, 1] - tri[:, 0]
    e12 = tri[:, 2] - tri[:, 1]
    e20 = tri[:, 0] - tri[:, 2]
    lens = np.linalg.norm(np.stack((e01, e12, e20), axis=1), axis=2)
    cross = np.cross(e01, e12)
    double_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * double_area

    edge_len = lens.ravel() / diag
    out["edge_len_med"] = float(np.median(edge_len))
    mean_len = float(edge_len.mean())
    out["edge_len_cv"] = float(edge_len.std() / max(mean_len, 1e-12))

    out["area_med"] = float(np.median(areas) / diag**2)
    out["degenerate_frac"] = float((double_area <= 1e-12 * diag**2).mean())

    # Area-weighted face-normal statistics.
    good = double_area > 0
    if good.any():
        normals = cross[good] / double_area[good, None]
        weights = areas[good] / areas[good].sum()
        theta = np.arccos(np.clip(normals[:, 2], -1.0, 1.0))
        phi = np.arctan2(normals[:, 1], normals[:, 0])
        ti = np.clip((theta / np.pi * 16).astype(np.int64), 0, 15)
        pi_ = np.clip(((phi + np.pi) / (2 * np.pi) * 32).astype(np.int64), 0, 31)
        hist = np.bincount(ti * 32 + pi_, weights=weights, minlength=512)
        out["normal_entropy"] = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
        out["normal_max_bin"] = float(hist.max())
        align = np.abs(normals).max(axis=1)
        out["axis_aligned_frac"] = float(weights[align > 0.9999].sum())
        centroids = ((tri[good][:, 0] + tri[good][:, 1] + tri[good][:, 2]) / 3.0 - (lo + hi) / 2.0)
        radial = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        out["radial_frac"] = float(weights[np.abs((normals * radial).sum(axis=1)) > 0.9999].sum())
    else:
        out["normal_entropy"] = 0.0
        out["normal_max_bin"] = 1.0
        out["axis_aligned_frac"] = 0.0
        out["radial_frac"] = 0.0

    if nf <= cap_faces:
        _structural_features(out, f, nv, nf)
    return out


def _structural_features(out: dict[str, float], f: np.ndarray, nv: int, nf: int) -> None:
    """Boundary/non-manifold edge fractions and connected components."""
    edges = np.concatenate((f[:, (0, 1)], f[:, (1, 2)], f[:, (2, 0)]))
    key = edges.min(axis=1) * np.int64(nv + 1) + edges.max(axis=1)
    key.sort()
    is_new = np.empty(len(key), dtype=bool)
    is_new[0] = True
    np.not_equal(key[1:], key[:-1], out=is_new[1:])
    start = np.flatnonzero(is_new)
    counts = np.diff(np.append(start, len(key)))
    out["boundary_edge_frac"] = float((counts == 1).mean())
    out["nonmanifold_edge_frac"] = float((counts > 2).mean())

    # Connected components over the vertex graph induced by unique edges.
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:  # pragma: no cover - scipy is a hard dependency in practice
        return
    unique_edges = edges[start]
    graph = coo_matrix(
        (np.ones(len(unique_edges), dtype=np.int8), (unique_edges[:, 0], unique_edges[:, 1])),
        shape=(nv, nv),
    )
    n_comp, labels = connected_components(graph, directed=False)
    out["n_components"] = float(n_comp)
    out["largest_component_frac"] = float(np.bincount(labels).max() / nv)
    out["faces_per_component"] = float(np.log10(nf / max(n_comp, 1)))


def _one(args: tuple[str, str, int]) -> tuple[str, dict[str, float]]:
    path, item_id, cap_faces = args
    with np.load(path) as data:
        feats = mesh_features(data["vertices"], data["faces"], cap_faces=cap_faces)
    return item_id, feats


def extract_features(
    npz_paths: Sequence[tuple[str, str]],
    workers: int = 1,
    cap_faces: int = 2_000_000,
) -> pd.DataFrame:
    """Extract features for ``(item_id, npz_path)`` pairs into a DataFrame."""
    jobs = [(path, item_id, cap_faces) for item_id, path in npz_paths]
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            results: Iterable[tuple[str, dict[str, float]]] = pool.imap_unordered(_one, jobs, chunksize=8)
            rows = [{"item_id": item_id, **feats} for item_id, feats in results]
    else:
        rows = [{"item_id": item_id, **feats} for item_id, feats in map(_one, jobs)]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["item_id", *FEATURE_NAMES])
    return frame.set_index("item_id").loc[[item_id for item_id, _ in npz_paths]].reset_index()
