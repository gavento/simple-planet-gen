"""Tectonic plate generation using Voronoi tessellation on a sphere."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldgen.noise import spherical_noise_single
from worldgen.world import WorldData, WorldParams


def _random_points_on_sphere(n: int, rng: np.random.RandomState):
    """Generate n uniformly distributed points on a unit sphere."""
    z = rng.uniform(-1, 1, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(1 - z * z)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


def _random_tangent_vectors(x, y, z, rng: np.random.RandomState):
    """Generate random tangent vectors (angular velocity) for plate motion.

    Returns (vx, vy, vz) velocity vectors tangent to the sphere at each point.
    """
    n = len(x)
    # Random Euler poles (axis of rotation for each plate)
    ex, ey, ez = _random_points_on_sphere(n, rng)
    # Angular speed
    speed = rng.uniform(0.3, 1.0, n)
    # Velocity at plate center = omega × r (cross product)
    vx = speed * (ey * z - ez * y)
    vy = speed * (ez * x - ex * z)
    vz = speed * (ex * y - ey * x)
    return vx, vy, vz


def generate_tectonics(world: WorldData, params: WorldParams):
    """Generate tectonic plates via noise-perturbed Voronoi on a sphere.

    Produces:
        plate_ids: (H, W) int32 - plate assignment for each cell
        plate_types: (num_plates,) int32 - 0=oceanic, 1=continental
        boundary_distance: (H, W) float32 - distance to nearest boundary (degrees)
        boundary_convergence: (H, W) float32 - convergence rate at boundaries
    """
    rng = np.random.RandomState(params.seed)
    num_plates = params.num_plates

    # Generate plate centers on unit sphere
    cx, cy, cz = _random_points_on_sphere(num_plates, rng)

    # Plate motion vectors
    vx, vy, vz = _random_tangent_vectors(cx, cy, cz, rng)

    # Classify plates: continental or oceanic
    n_continental = max(1, int(num_plates * params.continental_ratio))
    plate_types = np.zeros(num_plates, dtype=np.int32)
    continental_idx = rng.choice(num_plates, n_continental, replace=False)
    plate_types[continental_idx] = 1

    # Build KD-tree from plate centers
    centers = np.column_stack([cx, cy, cz])
    tree = cKDTree(centers)

    # Grid points on unit sphere
    grid_points = np.column_stack(
        [world.sphere_x.ravel(), world.sphere_y.ravel(), world.sphere_z.ravel()]
    )

    # Add noise to perturb the Voronoi boundaries
    noise_vals = spherical_noise_single(
        world.sphere_x,
        world.sphere_y,
        world.sphere_z,
        frequency=params.plate_noise_frequency,
        seed=params.seed + 100,
    )

    # Query KD-tree for 2 nearest plates
    dists, indices = tree.query(grid_points, k=2)
    dist1 = dists[:, 0].reshape(world.height, world.width)
    dist2 = dists[:, 1].reshape(world.height, world.width)
    idx1 = indices[:, 0].reshape(world.height, world.width)
    idx2 = indices[:, 1].reshape(world.height, world.width)

    # Perturb distances with noise to make boundaries irregular
    noise_weight = params.plate_noise_weight
    perturbed1 = dist1 + noise_weight * noise_vals * dist1.mean()
    perturbed2 = dist2 + noise_weight * noise_vals * dist1.mean()

    # Assign to nearest plate (with perturbation)
    plate_ids = np.where(perturbed1 <= perturbed2, idx1, idx2).astype(np.int32)

    # Boundary distance: angular distance between the two nearest plate distances
    # Small values = near boundary
    boundary_distance = np.degrees(np.arcsin(np.clip((dist2 - dist1) / 2, 0, 1)))

    # Compute convergence at boundaries (fully vectorized)
    boundary_convergence = np.zeros((world.height, world.width), dtype=np.float32)

    # Get the two plates meeting at each cell
    p1 = plate_ids
    p2 = np.where(perturbed1 <= perturbed2, idx2, idx1).astype(np.int32)

    # Only compute for cells near boundaries where plates differ
    near_boundary = (boundary_distance < 3.0) & (p1 != p2)

    pa = p1[near_boundary]
    pb = p2[near_boundary]

    # Relative velocity
    dvx = vx[pa] - vx[pb]
    dvy = vy[pa] - vy[pb]
    dvz = vz[pa] - vz[pb]

    # Normal: direction between plate centers
    nx = cx[pb] - cx[pa]
    ny = cy[pb] - cy[pa]
    nz = cz[pb] - cz[pa]
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    norm = np.maximum(norm, 1e-10)
    nx /= norm
    ny /= norm
    nz /= norm

    # Convergence = dot(dv, normal)
    boundary_convergence[near_boundary] = dvx * nx + dvy * ny + dvz * nz

    world["plate_ids"] = plate_ids
    world["plate_types"] = plate_types
    world["boundary_distance"] = boundary_distance.astype(np.float32)
    world["boundary_convergence"] = boundary_convergence
    world.metadata["num_plates"] = num_plates
