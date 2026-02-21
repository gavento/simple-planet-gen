"""Tectonic plate generation using Voronoi tessellation on a sphere."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldgen.noise import spherical_fbm, spherical_fbm_warped
from worldgen.world import WorldData, WorldParams


def _random_points_on_sphere(n: int, rng: np.random.RandomState):
    """Generate n uniformly distributed points on a unit sphere."""
    z = rng.uniform(-1, 1, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(1 - z * z)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


def _well_spaced_points_on_sphere(n: int, rng: np.random.RandomState, iterations: int = 50):
    """Generate n roughly evenly-spaced points on a sphere via repulsion.

    Uses Lloyd relaxation on the sphere: iteratively move points away
    from their nearest neighbors.
    """
    x, y, z = _random_points_on_sphere(n, rng)
    pts = np.column_stack([x, y, z])

    for _ in range(iterations):
        tree = cKDTree(pts)
        # For each point, find its nearest neighbor and push away
        dists, indices = tree.query(pts, k=2)
        nearest = pts[indices[:, 1]]

        # Repulsion vector (away from nearest neighbor)
        repulsion = pts - nearest
        # Normalize and scale by inverse distance
        rep_norm = np.linalg.norm(repulsion, axis=1, keepdims=True)
        rep_norm = np.maximum(rep_norm, 1e-10)
        repulsion = repulsion / rep_norm * 0.05

        pts += repulsion
        # Re-project onto unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts /= norms

    return pts[:, 0], pts[:, 1], pts[:, 2]


def _points_near_boundaries(
    n: int,
    major_centers: np.ndarray,
    rng: np.random.RandomState,
):
    """Generate n points preferentially near boundaries between major plates.

    Places minor plate centers near the edges of existing plates,
    creating realistic microplates and island arcs.
    """
    n_major = len(major_centers)
    tree = cKDTree(major_centers)

    # Generate candidate points and keep those near boundaries
    candidates_needed = n
    minor_pts = []

    while len(minor_pts) < candidates_needed:
        batch = 500
        cx, cy, cz = _random_points_on_sphere(batch, rng)
        pts = np.column_stack([cx, cy, cz])

        # Find distances to two nearest major plates
        dists, _ = tree.query(pts, k=2)
        # "Nearness to boundary" = how similar the two distances are
        boundary_score = 1.0 - (dists[:, 1] - dists[:, 0]) / (dists[:, 1] + 1e-10)

        # Accept points with probability proportional to boundary_score^2
        accept_prob = boundary_score ** 2
        accepted = rng.random(batch) < accept_prob

        for pt in pts[accepted]:
            minor_pts.append(pt)
            if len(minor_pts) >= candidates_needed:
                break

    minor_pts = np.array(minor_pts[:n])
    return minor_pts[:, 0], minor_pts[:, 1], minor_pts[:, 2]


def _random_tangent_vectors(x, y, z, rng: np.random.RandomState):
    """Generate random tangent vectors (angular velocity) for plate motion."""
    n = len(x)
    ex, ey, ez = _random_points_on_sphere(n, rng)
    speed = rng.uniform(0.3, 1.0, n)
    vx = speed * (ey * z - ez * y)
    vy = speed * (ez * x - ex * z)
    vz = speed * (ex * y - ey * x)
    return vx, vy, vz


def generate_tectonics(world: WorldData, params: WorldParams):
    """Generate tectonic plates with mixed sizes and tilt vectors.

    Major plates are well-spaced (large). Minor plates cluster near
    boundaries (small microplates, island arcs).

    Produces:
        plate_ids: (H, W) int32 - plate assignment for each cell
        plate_types: (num_plates,) int32 - 0=oceanic, 1=continental
        plate_tilt_elevation: (H, W) float32 - elevation from plate tilt
        plate_base_offsets: (num_plates,) float32 - per-plate base elevation offset
        boundary_distance: (H, W) float32 - distance to nearest boundary (degrees)
        boundary_convergence: (H, W) float32 - convergence rate at boundaries
    """
    rng = np.random.RandomState(params.seed)
    n_major = params.num_major_plates
    n_minor = params.num_minor_plates
    num_plates = n_major + n_minor

    # --- Generate plate centers ---
    # Major plates: well-spaced via repulsion
    mx, my, mz = _well_spaced_points_on_sphere(n_major, rng)
    major_centers = np.column_stack([mx, my, mz])

    # Minor plates: half near boundaries (microplates), half random (medium plates)
    if n_minor > 0:
        n_micro = n_minor * 2 // 3
        n_medium = n_minor - n_micro

        parts_x, parts_y, parts_z = [mx], [my], [mz]

        # Medium plates: randomly placed with moderate spacing
        if n_medium > 0:
            med_x, med_y, med_z = _random_points_on_sphere(n_medium, rng)
            parts_x.append(med_x)
            parts_y.append(med_y)
            parts_z.append(med_z)

        # Microplates: clustered near boundaries of all existing plates
        if n_micro > 0:
            all_so_far = np.column_stack(
                [np.concatenate(parts_x), np.concatenate(parts_y), np.concatenate(parts_z)]
            )
            mic_x, mic_y, mic_z = _points_near_boundaries(n_micro, all_so_far, rng)
            parts_x.append(mic_x)
            parts_y.append(mic_y)
            parts_z.append(mic_z)

        cx = np.concatenate(parts_x)
        cy = np.concatenate(parts_y)
        cz = np.concatenate(parts_z)
    else:
        cx, cy, cz = mx, my, mz

    # Plate motion vectors (with neighbor correlation)
    vx, vy, vz = _random_tangent_vectors(cx, cy, cz, rng)

    # Relax velocities so adjacent plates move in similar directions
    vel_tree = cKDTree(np.column_stack([cx, cy, cz]))
    for _ in range(2):
        _, nb_idx = vel_tree.query(np.column_stack([cx, cy, cz]), k=min(4, num_plates))
        nb_vx = vx[nb_idx[:, 1:]].mean(axis=1)
        nb_vy = vy[nb_idx[:, 1:]].mean(axis=1)
        nb_vz = vz[nb_idx[:, 1:]].mean(axis=1)
        vx = 0.7 * vx + 0.3 * nb_vx
        vy = 0.7 * vy + 0.3 * nb_vy
        vz = 0.7 * vz + 0.3 * nb_vz

    # --- Classify plates ---
    # Continental plates are more likely to be major plates
    plate_types = np.zeros(num_plates, dtype=np.int32)
    n_continental = max(1, int(num_plates * params.continental_ratio))

    # Bias toward major plates being continental
    major_continental = min(n_continental, n_major)
    major_cont_idx = rng.choice(n_major, major_continental, replace=False)
    plate_types[major_cont_idx] = 1

    # Fill remaining continental slots from minor plates
    remaining = n_continental - major_continental
    if remaining > 0 and n_minor > 0:
        minor_cont_idx = rng.choice(
            np.arange(n_major, num_plates), remaining, replace=False
        )
        plate_types[minor_cont_idx] = 1

    # --- Per-plate tilt vectors ---
    # Random direction in 3D (will be projected onto sphere surface)
    tilt_dirs = rng.randn(num_plates, 3)
    tilt_dirs /= np.linalg.norm(tilt_dirs, axis=1, keepdims=True)
    # Random tilt magnitude (some plates tilt more than others)
    tilt_mags = rng.uniform(0.3, 1.0, num_plates) * params.plate_tilt_strength

    # --- Build KD-tree and assign cells ---
    centers = np.column_stack([cx, cy, cz])
    tree = cKDTree(centers)

    # --- Per-plate base elevation offsets (spatially correlated) ---
    plate_base_offsets = rng.normal(0, 200, num_plates).astype(np.float32)
    # Relax offsets so nearby plates have similar heights
    for _ in range(3):
        _, neighbor_idx = tree.query(centers, k=min(5, num_plates))
        neighbor_mean = plate_base_offsets[neighbor_idx[:, 1:]].mean(axis=1)
        plate_base_offsets = 0.5 * plate_base_offsets + 0.5 * neighbor_mean
    # Continental plates: offset range +-500m, oceanic: +-200m
    plate_base_offsets *= np.where(plate_types == 1, 1.2, 0.5)

    grid_points = np.column_stack(
        [world.sphere_x.ravel(), world.sphere_y.ravel(), world.sphere_z.ravel()]
    )

    # --- Domain-warped Voronoi boundaries ---
    # Warp sample coordinates for organic, curving plate boundaries
    warp_x = spherical_fbm_warped(
        world.sphere_x, world.sphere_y, world.sphere_z,
        frequency=params.plate_warp_frequency, octaves=3, persistence=0.5,
        warp_strength=0.2, seed=params.seed + 200,
    )
    warp_y = spherical_fbm_warped(
        world.sphere_x, world.sphere_y, world.sphere_z,
        frequency=params.plate_warp_frequency, octaves=3, persistence=0.5,
        warp_strength=0.2, seed=params.seed + 201,
    )
    warp_z = spherical_fbm_warped(
        world.sphere_x, world.sphere_y, world.sphere_z,
        frequency=params.plate_warp_frequency, octaves=3, persistence=0.5,
        warp_strength=0.2, seed=params.seed + 202,
    )

    ws = params.plate_warp_strength
    warped_x = world.sphere_x + ws * warp_x
    warped_y = world.sphere_y + ws * warp_y
    warped_z = world.sphere_z + ws * warp_z
    # Re-normalize to unit sphere
    warped_norm = np.sqrt(warped_x**2 + warped_y**2 + warped_z**2)
    warped_x /= warped_norm
    warped_y /= warped_norm
    warped_z /= warped_norm

    warped_points = np.column_stack(
        [warped_x.ravel(), warped_y.ravel(), warped_z.ravel()]
    )

    # Query 2 nearest plates using warped coordinates
    dists, indices = tree.query(warped_points, k=2)
    dist1 = dists[:, 0].reshape(world.height, world.width)
    dist2 = dists[:, 1].reshape(world.height, world.width)
    idx1 = indices[:, 0].reshape(world.height, world.width)
    idx2 = indices[:, 1].reshape(world.height, world.width)

    # Fine-detail additive noise (secondary perturbation)
    fine_noise = spherical_fbm(
        world.sphere_x, world.sphere_y, world.sphere_z,
        frequency=params.plate_noise_frequency * 2, octaves=3, persistence=0.5,
        seed=params.seed + 102,
    )
    noise_weight = params.plate_noise_weight * 0.3  # reduced since warp does the heavy lifting
    mean_dist = dist1.mean()
    perturbation = noise_weight * fine_noise * mean_dist

    plate_ids = np.where(
        dist1 - dist2 + perturbation <= 0, idx1, idx2
    ).astype(np.int32)

    # Boundary distance
    boundary_distance = np.degrees(np.arcsin(np.clip((dist2 - dist1) / 2, 0, 1)))

    # --- Compute plate tilt elevation ---
    # For each cell, project its position onto its plate's tilt axis
    plate_tilt_elevation = np.zeros((world.height, world.width), dtype=np.float32)
    for p in range(num_plates):
        mask = plate_ids == p
        if not mask.any():
            continue
        # Dot product of cell position with tilt direction
        dot = (
            world.sphere_x[mask] * tilt_dirs[p, 0]
            + world.sphere_y[mask] * tilt_dirs[p, 1]
            + world.sphere_z[mask] * tilt_dirs[p, 2]
        )
        # Subtract the center's dot product so tilt is relative to plate center
        center_dot = cx[p] * tilt_dirs[p, 0] + cy[p] * tilt_dirs[p, 1] + cz[p] * tilt_dirs[p, 2]
        plate_tilt_elevation[mask] = (dot - center_dot) * tilt_mags[p]

    # --- Convergence at boundaries (vectorized) ---
    boundary_convergence = np.zeros((world.height, world.width), dtype=np.float32)
    p1 = plate_ids
    p2 = np.where(dist1 - dist2 + perturbation <= 0, idx2, idx1).astype(np.int32)
    near_boundary = (boundary_distance < 5.0) & (p1 != p2)

    pa = p1[near_boundary]
    pb = p2[near_boundary]
    dvx = vx[pa] - vx[pb]
    dvy = vy[pa] - vy[pb]
    dvz = vz[pa] - vz[pb]
    nx = cx[pb] - cx[pa]
    ny = cy[pb] - cy[pa]
    nz = cz[pb] - cz[pa]
    norm = np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-10)
    boundary_convergence[near_boundary] = (dvx * nx + dvy * ny + dvz * nz) / norm

    # --- Store results ---
    world["plate_ids"] = plate_ids
    world["plate_types"] = plate_types
    world["plate_tilt_elevation"] = plate_tilt_elevation
    world["plate_base_offsets"] = plate_base_offsets
    world["boundary_distance"] = boundary_distance.astype(np.float32)
    world["boundary_convergence"] = boundary_convergence
    world["plate_centers"] = centers.astype(np.float32)
    world["plate_velocities"] = np.column_stack([vx, vy, vz]).astype(np.float32)
    world.metadata["num_plates"] = num_plates
    world.metadata["num_major_plates"] = n_major
    world.metadata["num_minor_plates"] = n_minor
