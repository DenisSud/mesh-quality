"""Mesh surface tokenization for the JEPA / mesh-patch encoder.

Pipeline per item (pure numpy -- no open3d/trimesh/torch_cluster on Python 3.14):

    load npz -> normalise (centre + bbox diagonal) -> area-weighted surface
    sample (n_sample points) -> FPS (n_patch centres) -> kNN ball (k_points) ->
    per-patch statistics over the full Voronoi face region -> global voxel
    descriptor.

Two token streams come out of this (see ``knowledge/mesh-jepa-plan.md``):

* ``pts``   (n_patch, k_points, n_feat) -- tokenizer input, point features are
  patch-local and scale free;
* ``stats`` (n_patch, n_stats) -- hand-computed measurements of the patch, fed
  to the probe as raw (uncompressable) tokens, never through a learned encoder.

All items produce fixed-shape arrays, so the cache is a set of flat memmaps
indexed by row (fast random access for training, no per-item files).

Design notes
------------
* Faces are assigned to patches by nearest centre (Voronoi) so statistics
  aggregate over every face of the mesh, not just the sampled points. For meshes
  above ``_STATS_CAP_FACES`` the assignment runs over a strided face subset
  (counts are rescaled by the stride): ratio statistics stay unbiased and the
  cost of a 26M-face mesh stays flat.
* ``valid`` marks patches that actually own surface area. Small meshes (8 verts /
  4 faces exist in the corpus) produce many empty patches; train with the mask
  instead of trusting zeros.
* Coordinates are divided by the bbox diagonal, so patch features are scale free;
  absolute size lives in ``globals[log_diag]`` plus the probe's ``[GEOM]`` token.
* Degenerate faces get zero weight in sampling and are excluded from density
  moments, but are counted in ``degenerate_frac``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

CFG_VERSION = 2
_FACE_CHUNK = 2_000_000      # sampling / voxelisation face chunk
_STATS_CHUNK = 100_000       # (chunk, n_patch) distance blocks stay ~100 MB
_STATS_CAP_FACES = 1_000_000  # strided face subset for per-patch statistics
_VOXEL_CAP_FACES = 2_000_000
_QUANTILE_FACES = 131_072


@dataclass(frozen=True)
class TokenConfig:
    """Tokenizer hyper-parameters (cache contract: change -> new cache version)."""

    n_sample: int = 16384        # area-weighted surface points
    n_patch: int = 256           # FPS patch centres
    k_points: int = 32           # points per patch (kNN then resampled)
    voxel_res: int = 16          # global occupancy grid (voxel_res**3)
    census_cap_faces: int = 8_000_000   # above this, structural flags are skipped
    weld_quant: float = 1e-5     # vertex weld grid, relative to the bbox diagonal

    @property
    def n_feat(self) -> int:
        return len(POINT_NAMES)

    @property
    def n_stats(self) -> int:
        return len(STAT_NAMES)

    @property
    def n_global(self) -> int:
        return len(GLOBAL_NAMES)


GLOBAL_NAMES: tuple[str, ...] = (
    "log_diag",              # absolute size in mesh units -> `scale`
    "log_n_faces",
    "log_n_verts",
    "log_area_total",        # total area / diag^2
    "bbox_flatness",
    "center_dist_rel",       # bbox centre distance from origin / diag
    "degenerate_frac",
    "boundary_edge_frac",    # 0 when has_structure == 0
    "nonmanifold_edge_frac",
    "has_structure",         # 1 when the edge census ran, else 0
    "has_area",              # 0 for zero-area meshes (all area stats meaningless)
)

STAT_NAMES: tuple[str, ...] = (
    "log1p_faces",
    "log_area_frac",         # patch area / total area
    "log_density",           # log10 faces per unit normalised area
    "log_area_uniformity",   # median face area / mean face area in the patch
    "log_edge_med",          # median edge length / diag
    "edge_len_cv",
    "normal_var",            # 1 - |area-weighted mean unit normal|
    "boundary_edge_frac",
    "nonmanifold_edge_frac",
    "degenerate_frac",
    "frac_normal_dev_gt60",  # fraction of faces facing >60 deg from the mean normal
    "log_radius",            # log(patch radius / diag)
    "radial_pos",            # |patch centre - bbox centre| / diag
    "surface_ratio",         # patch area / (pi r^2): folded vs flat region
    "area_frac_outside_r",   # patch area farther than 1.5r from the centre
    "outward_frac",          # area-weighted fraction facing away from bbox centre
)

POINT_NAMES: tuple[str, ...] = (
    "dx_r", "dy_r", "dz_r",   # (p - centre) / r
    "nx", "ny", "nz",         # face normal (sign ambiguous)
    "normal_align",           # |dot(n, patch mean normal)|
    "log_area",               # log(face area / diag^2)
    "log_edge",               # log(face mean edge / diag)
    "boundary_edge",          # 1 when the source face has a boundary edge
    "patch_radius",           # r / diag (broadcast, gives the tokenizer scale)
)


# --------------------------------------------------------------------------- #
# mesh loading / normalisation
# --------------------------------------------------------------------------- #


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
    return vertices, faces


def normalize_mesh(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Centre on the bbox centre, divide by the bbox diagonal, drop bad faces.

    Absolute size is *not* lost: ``diag`` is returned and lands in the globals.
    """
    if len(faces):
        bad = (
            (faces < 0).any(axis=1)
            | (faces >= len(vertices)).any(axis=1)
            | (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 0] == faces[:, 2])
        )
        if bad.any():
            faces = faces[~bad]
    if len(vertices) == 0:
        return vertices, faces, {"diag": 1.0, "center": np.zeros(3), "n_raw": 0.0}
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo))
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0
    return (vertices - center) / diag, faces, {"diag": diag, "center": center, "n_raw": float(len(vertices))}


def _chunks(n: int, size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n, size):
        yield start, min(start + size, n)


def _tri_geometry(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(areas, unit normals, mean edge lengths) for a (c, 3, 3) triangle array."""
    e = np.stack((tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]), axis=1)
    lens = np.linalg.norm(e, axis=2)
    cross = np.cross(e[:, 0], e[:, 1])
    two_a = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(two_a, 1e-30)[:, None]
    return 0.5 * two_a, normals, lens.mean(axis=1)


# --------------------------------------------------------------------------- #
# area-weighted surface sampling
# --------------------------------------------------------------------------- #


def sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
    chunk: int = _FACE_CHUNK,
) -> dict[str, Any]:
    """Uniformly sample ``n_points`` on the mesh surface (area weighted).

    Degenerate faces get zero weight automatically. One full area pass is enough
    (``np.cumsum`` + ``searchsorted``); geometry is recomputed only for the
    touched faces.
    """
    nf = len(faces)
    out: dict[str, Any] = {
        "points": np.zeros((n_points, 3), np.float64),
        "face": np.zeros(n_points, np.int64),
        "normal": np.zeros((n_points, 3), np.float32),
        "log_area": np.zeros(n_points, np.float32),
        "log_edge": np.zeros(n_points, np.float32),
        "area_total": 0.0,
        "n_degenerate": 0.0,
    }
    if nf == 0 or n_points == 0:
        return out

    areas = np.empty(nf, dtype=np.float32)
    for s, e in _chunks(nf, chunk):
        a, _, _ = _tri_geometry(vertices[faces[s:e]])
        areas[s:e] = a
    out["area_total"] = float(areas.sum())
    out["n_degenerate"] = float((areas <= 0).sum())
    if out["area_total"] <= 0:
        return out

    cum = np.cumsum(areas, dtype=np.float64)
    target = rng.random(n_points) * cum[-1]
    loc = np.clip(np.searchsorted(cum, target, side="right"), 0, nf - 1)
    del cum, areas

    uniq, inv = np.unique(loc, return_inverse=True)
    _, normals, edges = _tri_geometry(vertices[faces[uniq]])
    r1 = np.sqrt(rng.random(n_points))
    r2 = rng.random(n_points)
    tri = vertices[faces[loc]]
    out["points"] = (
        (1.0 - r1)[:, None] * tri[:, 0] + (r1 * (1.0 - r2))[:, None] * tri[:, 1] + (r1 * r2)[:, None] * tri[:, 2]
    )
    out["face"] = loc
    out["normal"] = normals[inv]
    out["log_area"] = np.log(np.maximum(_tri_area_of(vertices, faces, loc), 1e-30))
    out["log_edge"] = np.log(np.maximum(edges[inv], 1e-30))
    return out


def _tri_area_of(vertices: np.ndarray, faces: np.ndarray, loc: np.ndarray) -> np.ndarray:
    tri = vertices[faces[loc]]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


# --------------------------------------------------------------------------- #
# farthest point sampling / kNN
# --------------------------------------------------------------------------- #


def fps(points: np.ndarray, k: int, start: int | None = None) -> np.ndarray:
    """Greedy farthest point sampling. Deterministic start: min coordinate sum."""
    n = len(points)
    k = min(k, n)
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    if start is None:
        start = int(points.sum(axis=1).argmin())
    chosen = np.empty(k, dtype=np.int64)
    dist = np.full(n, np.inf, dtype=np.float64)
    cur = points[start]
    for i in range(k):
        d = ((points - cur) ** 2).sum(axis=1)
        np.minimum(dist, d, out=dist)
        idx = int(dist.argmax())
        chosen[i] = idx
        cur = points[idx]
    return chosen


def knn_patches(centers: np.ndarray, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices (k) nearest to every centre plus the patch radius (max distance)."""
    if len(points) == 0:
        return np.zeros((len(centers), 0), dtype=np.int64), np.zeros(len(centers))
    c2 = (centers**2).sum(1)[:, None]
    p2 = (points**2).sum(1)[None, :]
    d2 = np.maximum(c2 + p2 - 2.0 * centers @ points.T, 0.0)
    k = min(k, len(points))
    part = np.argpartition(d2, k - 1, axis=1)[:, :k]
    rows = np.arange(len(centers))[:, None]
    dsel = d2[rows, part]
    order = np.argsort(dsel, axis=1)
    return part[rows, order], np.sqrt(np.maximum(dsel[rows, order][:, -1], 0.0))


# --------------------------------------------------------------------------- #
# edge census (weld -> boundary / non-manifold flags)
# --------------------------------------------------------------------------- #


def _weld(vertices: np.ndarray, quant: float) -> np.ndarray:
    grid = np.rint(vertices / max(quant, 1e-12)).astype(np.int64)
    grid -= grid.min(axis=0, keepdims=True)
    key = (grid[:, 0] << 42) ^ (grid[:, 1] << 21) ^ grid[:, 2]
    return np.unique(key, return_inverse=True)[1]


def census(
    vertices: np.ndarray, faces: np.ndarray, cfg: TokenConfig
) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
    """Boundary-edge count per face (0..3), non-manifold flag, global fractions.

    Returns ``ok=False`` when the mesh exceeds ``census_cap_faces`` (the two
    fractions are then 0 and the caller must honour ``has_structure``).
    """
    nf = len(faces)
    if nf == 0:
        return np.zeros(0, np.int8), np.zeros(0, bool), 0.0, 0.0, True
    if nf > cfg.census_cap_faces:
        return np.full(nf, -1, np.int8), np.zeros(nf, bool), 0.0, 0.0, False
    inv = _weld(vertices, cfg.weld_quant)
    f = inv[faces]
    edges = np.concatenate((f[:, (0, 1)], f[:, (1, 2)], f[:, (2, 0)]))
    nv = int(inv.max()) + 1
    key = edges.min(axis=1) * np.int64(nv + 1) + edges.max(axis=1)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    per_edge = counts[inverse].reshape(nf, 3)
    return (
        (per_edge == 1).sum(axis=1).astype(np.int8),
        (per_edge > 2).any(axis=1),
        float((counts == 1).mean()),
        float((counts > 2).mean()),
        True,
    )


# --------------------------------------------------------------------------- #
# patch statistics over the full Voronoi face region
# --------------------------------------------------------------------------- #


def assign_faces(centroids: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-centre assignment for a face-centroid block."""
    d2 = (centroids**2).sum(1)[:, None] + (centers**2).sum(1)[None, :] - 2.0 * centroids @ centers.T
    return np.argmin(np.maximum(d2, 0.0), axis=1)


def patch_stats(
    vertices: np.ndarray,
    faces: np.ndarray,
    centers: np.ndarray,
    radius: np.ndarray,
    cfg: TokenConfig,
    bnd: np.ndarray,
    nonman: np.ndarray,
    has_structure: bool,
    area_total: float,
) -> np.ndarray:
    """Per-patch statistics aggregated over the (possibly strided) face set."""
    k = len(centers)
    stats = np.zeros((k, cfg.n_stats), dtype=np.float32)
    nf = len(faces)
    if nf == 0 or area_total <= 0:
        return stats

    stride = max(1, nf // _STATS_CAP_FACES)
    sub = np.arange(0, nf, stride)
    counts = np.zeros(k)
    area_sum = np.zeros(k)
    area_out = np.zeros(k)
    degen = np.zeros(k)
    bnd_area = np.zeros(k)
    nonman_area = np.zeros(k)
    nrm_sum = np.zeros((k, 3))
    outward = np.zeros(k)

    for s, e in _chunks(len(sub), _STATS_CHUNK):
        idx = sub[s:e]
        tri = vertices[faces[idx]]
        areas, normals, _ = _tri_geometry(tri)
        block_cen = tri.mean(axis=1)
        a_idx = assign_faces(block_cen, centers)
        good = areas > 0
        a = np.where(good, areas, 0.0)
        nn = np.where(good[:, None], normals, 0.0)
        counts += np.bincount(a_idx, minlength=k)
        area_sum += np.bincount(a_idx, weights=a, minlength=k)
        degen += np.bincount(a_idx, weights=(~good).astype(np.float64), minlength=k)
        for j in range(3):
            nrm_sum[:, j] += np.bincount(a_idx, weights=a * nn[:, j], minlength=k)
        outward += np.bincount(a_idx, weights=a * ((nn * block_cen).sum(axis=1) > 0), minlength=k)
        far = np.linalg.norm(block_cen - centers[a_idx], axis=1) > 1.5 * radius[a_idx]
        area_out += np.bincount(a_idx, weights=a * far, minlength=k)
        if has_structure:
            bnd_area += np.bincount(a_idx, weights=a * (bnd[idx] > 0), minlength=k)
            nonman_area += np.bincount(a_idx, weights=a * nonman[idx], minlength=k)

    nrm_norm = np.linalg.norm(nrm_sum, axis=1)
    w = np.maximum(area_sum, 1e-30)
    mean_n = nrm_sum / np.maximum(nrm_norm[:, None], 1e-30)

    # Unbiased (not area-weighted, not count-rescaled) face-quantile subsample.
    qstride = max(1, len(sub) // _QUANTILE_FACES)
    qsub = sub[::qstride]
    med_area = np.zeros(k)
    med_edge = np.zeros(k)
    cv_edge = np.zeros(k)
    dev60 = np.zeros(k)
    for s, e in _chunks(len(qsub), _STATS_CHUNK):
        tri = vertices[faces[qsub[s:e]]]
        areas, normals, edges = _tri_geometry(tri)
        a_idx = assign_faces(tri.mean(axis=1), centers)
        align = np.abs((normals * mean_n[a_idx]).sum(axis=1))
        for p in np.unique(a_idx):
            sel = a_idx == p
            aa, ee = areas[sel], edges[sel]
            med_area[p] = float(np.median(aa))
            med_edge[p] = float(np.median(ee))
            cv_edge[p] = float(ee.std() / max(ee.mean(), 1e-12))
            dev60[p] = float((align[sel] < 0.5).mean())
    empty = med_area <= 0
    med_area[empty] = area_sum[empty] / np.maximum(counts[empty], 1.0)
    med_edge[empty] = radius[empty]

    r = np.maximum(radius, 1e-12)
    eps = 1e-12
    stats[:, 0] = np.log1p(counts * stride)
    stats[:, 1] = np.log(np.maximum(area_sum / area_total, eps))
    stats[:, 2] = np.log10(np.maximum(counts / w, eps))
    stats[:, 3] = np.log(np.maximum(med_area / np.maximum(area_sum / np.maximum(counts, 1.0), eps), eps))
    stats[:, 4] = np.log(np.maximum(med_edge, eps))
    stats[:, 5] = cv_edge
    stats[:, 6] = 1.0 - nrm_norm / w
    stats[:, 7] = bnd_area / w if has_structure else 0.0
    stats[:, 8] = nonman_area / w if has_structure else 0.0
    stats[:, 9] = degen / np.maximum(counts, 1.0)
    stats[:, 10] = dev60
    stats[:, 11] = np.log(r)
    stats[:, 12] = np.linalg.norm(centers, axis=1)
    stats[:, 13] = area_sum / (np.pi * r**2)
    stats[:, 14] = area_out / w
    stats[:, 15] = outward / w
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.clip(stats, -30.0, 30.0)


# --------------------------------------------------------------------------- #
# global voxel descriptor
# --------------------------------------------------------------------------- #


def voxel_descriptor(vertices: np.ndarray, faces: np.ndarray, res: int, chunk: int = _FACE_CHUNK) -> np.ndarray:
    """log1p face-centroid count per voxel of the normalised bbox (strided above cap)."""
    out = np.zeros(res**3, dtype=np.float32)
    nf = len(faces)
    if nf == 0:
        return out
    stride = max(1, nf // _VOXEL_CAP_FACES)
    grid = np.empty(len(range(0, nf, stride)), dtype=np.int64)
    for s, e in _chunks(len(grid), chunk):
        idx = np.arange(0, nf, stride)[s:e]
        block_cen = vertices[faces[idx]].mean(axis=1)
        q = np.clip(((block_cen + 0.5) * res).astype(np.int64), 0, res - 1)
        grid[s:e] = (q[:, 0] * res + q[:, 1]) * res + q[:, 2]
    out[:] = np.log1p(np.bincount(grid, minlength=res**3).astype(np.float32) * stride)
    return out


# --------------------------------------------------------------------------- #
# per-item pipeline
# --------------------------------------------------------------------------- #


def tokenize_item(path: str | Path, cfg: TokenConfig = TokenConfig(), seed: int = 0) -> dict[str, np.ndarray]:
    """Full tokenizer for one mesh. Returns fixed-shape arrays (see module doc)."""
    vertices, faces = load_mesh(path)
    vertices, faces, meta = normalize_mesh(vertices, faces)
    diag = float(meta["diag"])
    rng = np.random.default_rng(seed)

    sample = sample_surface(vertices, faces, cfg.n_sample, rng)
    pts = sample["points"]
    centers = pts[fps(pts, cfg.n_patch)] if len(pts) else np.zeros((cfg.n_patch, 3))
    members, radius = knn_patches(centers, pts, cfg.k_points)
    bnd, nonman, bnd_frac, nonman_frac, has_structure = census(vertices, faces, cfg)

    k = cfg.k_points
    feats = np.zeros((cfg.n_patch, k, cfg.n_feat), dtype=np.float32)
    r = np.maximum(radius, 1e-12)
    for p in range(cfg.n_patch):
        sel = members[p]
        if len(sel) == 0:
            continue
        take = rng.permutation(len(sel))[:k] if len(sel) >= k else rng.integers(0, len(sel), k)
        src = sample["face"][sel[take]]
        feats[p, :, 0:3] = (pts[sel[take]] - centers[p]) / r[p]
        feats[p, :, 3:6] = sample["normal"][sel[take]]
        feats[p, :, 7] = sample["log_area"][sel[take]]
        feats[p, :, 8] = sample["log_edge"][sel[take]]
        if has_structure:
            feats[p, :, 9] = (bnd[src] > 0).astype(np.float32)
        feats[p, :, 10] = radius[p]
    mean_n = feats[:, :, 3:6].mean(axis=1)
    mean_n /= np.maximum(np.linalg.norm(mean_n, axis=1, keepdims=True), 1e-12)
    feats[:, :, 6] = np.abs((feats[:, :, 3:6] * mean_n[:, None, :]).sum(axis=2))

    stats = patch_stats(vertices, faces, centers, radius, cfg, bnd, nonman, has_structure, float(sample["area_total"]))
    valid = stats[:, 1] > np.log(1e-9)

    extent = (vertices.max(0) - vertices.min(0)) if len(vertices) else np.zeros(3)
    glob = np.array(
        [
            np.log(diag),
            np.log1p(len(faces)),
            np.log1p(len(vertices)),
            np.log(max(float(sample["area_total"]), 1e-12)),
            float(extent.min() / max(extent.max(), 1e-12)) if len(vertices) else 0.0,
            float(np.linalg.norm(meta["center"])) / diag,
            float(sample["n_degenerate"]) / max(len(faces), 1),
            bnd_frac,
            nonman_frac,
            float(has_structure),
            float(sample["area_total"] > 0),
        ],
        dtype=np.float32,
    )
    return {
        "pts": feats.astype(np.float16),
        "stats": stats,
        "centers": centers.astype(np.float16),
        "radius": radius.astype(np.float16),
        "voxel": voxel_descriptor(vertices, faces, cfg.voxel_res),
        "globals": glob,
        "valid": valid.astype(np.uint8),
    }


# --------------------------------------------------------------------------- #
# cache build / load
# --------------------------------------------------------------------------- #

_ARRAYS: dict[str, tuple[str, Any]] = {
    "pts": ("float16", lambda c: (c.n_patch, c.k_points, c.n_feat)),
    "stats": ("float16", lambda c: (c.n_patch, c.n_stats)),
    "centers": ("float16", lambda c: (c.n_patch, 3)),
    "radius": ("float16", lambda c: (c.n_patch,)),
    "voxel": ("float16", lambda c: (c.voxel_res**3,)),
    "globals": ("float32", lambda c: (c.n_global,)),
    "valid": ("uint8", lambda c: (c.n_patch,)),
}


def item_seed(item_id: str) -> int:
    """Stable per-item RNG seed (re-tokenizing one item reproduces its row)."""
    return int(hashlib.blake2b(item_id.encode(), digest_size=4).hexdigest(), 16)


def _worker(args: tuple[int, str, TokenConfig]) -> tuple[int, dict[str, np.ndarray]]:
    _, path, cfg = args
    return args[0], tokenize_item(path, cfg, seed=item_seed(Path(path).stem))


def build_cache(
    npz_paths: Sequence[str | Path],
    out_dir: str | Path,
    cfg: TokenConfig = TokenConfig(),
    workers: int = 8,
    rows: Sequence[int] | None = None,
    resume: bool = False,
    verbose: bool = True,
) -> Path:
    """Tokenize meshes into a flat memmapped cache directory.

    ``npz_paths`` is the full split (defines the row order / item_ids); ``rows``
    selects a subset to process now, which lets the caller keep huge meshes out
    of the worker pool (``resume=True`` appends to an existing cache).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(npz_paths)
    mode = "r+" if resume else "w+"
    arrays = {
        name: np.lib.format.open_memmap(out / f"{name}.npy", mode=mode, dtype=dtype, shape=(n, *shape_fn(cfg)))
        for name, (dtype, shape_fn) in _ARRAYS.items()
    }
    if not resume:
        for arr in arrays.values():
            arr[:] = 0

    sel = list(range(n)) if rows is None else list(rows)
    jobs = [(i, str(npz_paths[i]), cfg) for i in sel]
    done = 0
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers, maxtasksperchild=64) as pool:
            for index, item in pool.imap_unordered(_worker, jobs, chunksize=1):
                for name, arr in arrays.items():
                    arr[index] = item[name]
                done += 1
                if verbose and done % 250 == 0:
                    print(f"  {done}/{len(jobs)}", flush=True)
    else:
        for job in jobs:
            index, item = _worker(job)
            for name, arr in arrays.items():
                arr[index] = item[name]
            done += 1
            if verbose and done % 25 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    for arr in arrays.values():
        arr.flush()
    del arrays
    (out / "item_ids.txt").write_text("\n".join(Path(p).stem for p in npz_paths) + "\n")
    (out / "meta.json").write_text(json.dumps({"config": asdict(cfg), "version": CFG_VERSION, "n": n}, indent=2))
    return out


def load_cache(cache_dir: str | Path, mmap: bool = True) -> dict[str, np.ndarray]:
    """Open a tokenizer cache (memmapped by default) for training."""
    cache_dir = Path(cache_dir)
    out = {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r" if mmap else None) for name in _ARRAYS}
    out["item_ids"] = (cache_dir / "item_ids.txt").read_text().split()
    return out
