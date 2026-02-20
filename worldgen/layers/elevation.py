"""Elevation generation from tectonic plates + multi-scale noise."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from worldgen.noise import spherical_fbm, spherical_fbm_warped
from worldgen.world import WorldData, WorldParams


def generate_elevation(world: WorldData, params: WorldParams):
    """Generate terrain elevation from plate tectonics + layered noise.

    Builds elevation from several components:
    1. Continental vs oceanic base (smoothed transition = shelf)
    2. Large-scale continental warping (highlands, basins, plateaus)
    3. Mountain ranges at convergent boundaries (wide, with foothills)
    4. Mid-ocean ridges at divergent boundaries
    5. Domain-warped multi-octave noise for organic terrain detail
    6. Fine detail noise scaled by local roughness

    Produces:
        elevation: (H, W) float32 - elevation in meters
    """
    plate_ids = world["plate_ids"]
    plate_types = world["plate_types"]
    boundary_distance = world["boundary_distance"]
    boundary_convergence = world["boundary_convergence"]

    H, W = world.height, world.width
    sx, sy, sz = world.sphere_x, world.sphere_y, world.sphere_z

    is_continental = plate_types[plate_ids].astype(np.float64)

    # =========================================================
    # 1. CONTINENTAL SHELF — smooth coast-to-abyss transition
    # =========================================================
    # Wide Gaussian smoothing of the continental mask creates a
    # gradual shelf/slope rather than a sharp step at the coast.
    shelf_sigma = max(5, W // 80)  # ~4-5° of smoothing
    continental_smooth = gaussian_filter(is_continental, sigma=shelf_sigma)

    # Add noise to the shelf edge so it's not perfectly smooth
    shelf_noise = spherical_fbm(
        sx, sy, sz,
        frequency=4.0, octaves=4, persistence=0.5,
        seed=params.seed + 300,
    )
    # Warp the smooth field with noise (shifts the shelf edge irregularly)
    continental_warped = np.clip(
        continental_smooth + 0.15 * shelf_noise * (1 - continental_smooth**2),
        0, 1,
    )

    base = (
        continental_warped * params.continental_base
        + (1 - continental_warped) * params.oceanic_base
    )

    # =========================================================
    # 2. LARGE-SCALE CONTINENTAL STRUCTURE
    # =========================================================
    # Low-frequency, high-amplitude noise that creates broad highlands,
    # lowland basins, and plateaus within continental interiors.
    # This is the #1 thing that was missing before.
    continental_structure = spherical_fbm_warped(
        sx, sy, sz,
        frequency=2.5, octaves=4, persistence=0.45, lacunarity=2.0,
        warp_strength=0.3, warp_octaves=2,
        seed=params.seed + 400,
    )

    # Apply only on continental areas, scaled by how "interior" we are
    interior_factor = np.clip(continental_warped * 2 - 0.5, 0, 1)
    # Range: some areas 1500m above base, some 1000m below
    base += continental_structure * 1800.0 * interior_factor

    # Second large-scale layer at slightly different frequency for variety
    continental_structure2 = spherical_fbm(
        sx, sy, sz,
        frequency=4.0, octaves=3, persistence=0.5,
        seed=params.seed + 410,
    )
    base += continental_structure2 * 600.0 * interior_factor

    # =========================================================
    # 3. MOUNTAIN RANGES (convergent boundaries)
    # =========================================================
    # Much wider influence than before, with asymmetric profile:
    # steep on one side, gradual foothills on the other.
    bw_core = params.boundary_mountain_width  # core width
    bw_foothills = bw_core * 3  # foothill extent

    # Core mountain profile (narrow, tall)
    core_factor = np.exp(-0.5 * (boundary_distance / bw_core) ** 2)
    # Foothill profile (wide, lower)
    foothill_factor = np.exp(-0.5 * (boundary_distance / bw_foothills) ** 2)

    # Add noise to mountain height for variation along the range
    mountain_noise = spherical_fbm(
        sx, sy, sz,
        frequency=8.0, octaves=4, persistence=0.5,
        seed=params.seed + 500,
    )
    # Ensure positive with variation (0.3 to 1.0 range)
    height_variation = np.clip(0.65 + 0.35 * mountain_noise, 0.3, 1.0)

    # Convergent = positive convergence
    conv_strength = np.clip(boundary_convergence, 0, None)
    div_strength = np.clip(-boundary_convergence, 0, None)

    # Scale convergence to [0, 1] range
    conv_max = np.percentile(conv_strength[conv_strength > 0], 95) if np.any(conv_strength > 0) else 1.0
    conv_norm = np.clip(conv_strength / max(conv_max, 1e-6), 0, 1.5)

    div_max = np.percentile(div_strength[div_strength > 0], 95) if np.any(div_strength > 0) else 1.0
    div_norm = np.clip(div_strength / max(div_max, 1e-6), 0, 1.5)

    # Mountain elevation: core peaks + broad foothills
    mountain_height = params.boundary_mountain_height * height_variation * conv_norm
    mountains = (
        0.6 * mountain_height * core_factor
        + 0.4 * mountain_height * foothill_factor
    )

    # Continental-continental convergence: extra height
    both_continental = is_continental * np.roll(is_continental, 1, axis=1)
    mountains *= 1.0 + 0.3 * both_continental * core_factor

    # Divergent boundaries: mid-ocean ridges (lower, broader)
    ridge_height = 1500.0 * div_norm * height_variation
    ridges = ridge_height * foothill_factor * (1 - continental_warped)

    base += mountains + ridges

    # =========================================================
    # 4. DOMAIN-WARPED TERRAIN NOISE (organic detail)
    # =========================================================
    # This replaces the old uniform fBm with warped noise that
    # produces more natural, varied terrain patterns.
    terrain_noise = spherical_fbm_warped(
        sx, sy, sz,
        frequency=params.elevation_noise_frequency,
        octaves=params.elevation_noise_octaves,
        persistence=params.elevation_noise_persistence,
        lacunarity=2.0,
        warp_strength=0.35,
        warp_octaves=3,
        seed=params.seed + 200,
    )

    # Scale differently for land vs ocean, and more in rugged areas
    noise_scale_land = 1200.0
    noise_scale_ocean = 600.0
    noise_scale = (
        continental_warped * noise_scale_land
        + (1 - continental_warped) * noise_scale_ocean
    )
    # Mountains get extra roughness (rugged terrain near ranges)
    ruggedness = 1.0 + 1.5 * core_factor * conv_norm
    base += terrain_noise * noise_scale * ruggedness

    # =========================================================
    # 5. FINE DETAIL (higher frequency, lower amplitude)
    # =========================================================
    fine_noise = spherical_fbm(
        sx, sy, sz,
        frequency=15.0, octaves=4, persistence=0.45,
        seed=params.seed + 250,
    )
    # Fine detail is subtle: 200m on land, 100m in ocean
    fine_scale = continental_warped * 200.0 + (1 - continental_warped) * 100.0
    base += fine_noise * fine_scale

    # =========================================================
    # 6. OCEAN FLOOR FEATURES
    # =========================================================
    # Abyssal hills and seamounts in deep ocean
    ocean_noise = spherical_fbm(
        sx, sy, sz,
        frequency=12.0, octaves=3, persistence=0.5,
        seed=params.seed + 350,
    )
    ocean_features = ocean_noise * 400.0 * (1 - continental_warped)
    base += ocean_features

    # =========================================================
    # FINAL: clamp and gentle smoothing
    # =========================================================
    elevation = np.clip(base, params.ocean_depth, params.mountain_height * 1.2)

    # Very light smoothing to remove grid artifacts, not terrain features
    elevation = gaussian_filter(elevation, sigma=0.8)

    world["elevation"] = elevation.astype(np.float32)
