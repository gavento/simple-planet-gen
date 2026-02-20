"""Elevation generation from tectonic plates + fractal noise."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from worldgen.noise import spherical_fbm
from worldgen.world import WorldData, WorldParams


def generate_elevation(world: WorldData, params: WorldParams):
    """Generate terrain elevation from plate tectonics + noise.

    Produces:
        elevation: (H, W) float32 - elevation in meters
    """
    plate_ids = world["plate_ids"]
    plate_types = world["plate_types"]
    boundary_distance = world["boundary_distance"]
    boundary_convergence = world["boundary_convergence"]

    H, W = world.height, world.width

    # --- Base elevation from plate type ---
    is_continental = plate_types[plate_ids].astype(np.float64)
    base = np.where(
        is_continental > 0.5,
        params.continental_base,
        params.oceanic_base,
    )

    # --- Boundary features ---
    # Mountains at convergent boundaries
    bw = params.boundary_mountain_width
    boundary_factor = np.exp(-0.5 * (boundary_distance / bw) ** 2)

    # Convergent: mountains/trenches
    # convergence > 0 → converging → mountains
    # convergence < 0 → diverging → ridges (smaller)
    convergent_height = boundary_convergence * boundary_factor

    # Continental-continental convergence → big mountains
    # Ocean-continental → subduction mountains on continental side
    # Ocean-ocean divergence → mid-ocean ridges
    mountain_addition = np.zeros((H, W), dtype=np.float64)

    conv_pos = np.clip(convergent_height, 0, None)
    conv_neg = np.clip(-convergent_height, 0, None)

    # Convergent boundaries: tall mountains
    mountain_addition += conv_pos * params.boundary_mountain_height

    # Divergent boundaries: mid-ocean ridges or rift valleys (smaller)
    mountain_addition += conv_neg * params.boundary_mountain_height * 0.3

    # --- Fractal noise for terrain detail ---
    noise_terrain = spherical_fbm(
        world.lat_grid,
        world.lon_grid,
        world.sphere_x,
        world.sphere_y,
        world.sphere_z,
        frequency=params.elevation_noise_frequency,
        octaves=params.elevation_noise_octaves,
        persistence=params.elevation_noise_persistence,
        seed=params.seed + 200,
    )

    # Scale noise differently for continental vs oceanic
    noise_scale_land = 1500.0  # meters of variation on continents
    noise_scale_ocean = 800.0  # meters of variation in oceans
    noise_scale = np.where(is_continental > 0.5, noise_scale_land, noise_scale_ocean)
    noise_contribution = noise_terrain * noise_scale

    # --- Continental shelf: smooth transition at plate edges ---
    # Add some noise-based variation to the continental shelf
    shelf_noise = spherical_fbm(
        world.lat_grid,
        world.lon_grid,
        world.sphere_x,
        world.sphere_y,
        world.sphere_z,
        frequency=3.0,
        octaves=4,
        persistence=0.5,
        seed=params.seed + 300,
    )

    # Smooth the continental/oceanic transition
    continental_smooth = gaussian_filter(is_continental.astype(np.float64), sigma=5)
    base_smooth = (
        continental_smooth * params.continental_base
        + (1 - continental_smooth) * params.oceanic_base
    )
    # Add shelf variation
    base_smooth += shelf_noise * 500 * continental_smooth * (1 - continental_smooth) * 4

    # --- Combine ---
    elevation = base_smooth + mountain_addition + noise_contribution

    # Clamp to reasonable range
    elevation = np.clip(elevation, params.ocean_depth, params.mountain_height * 1.2)

    # Light smoothing for realism
    elevation = gaussian_filter(elevation, sigma=1.5)

    world["elevation"] = elevation.astype(np.float32)
