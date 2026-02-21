"""Elevation generation from tectonic plates + multi-scale terrain."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from worldgen.noise import spherical_fbm, spherical_fbm_warped
from worldgen.world import WorldData, WorldParams


def generate_elevation(world: WorldData, params: WorldParams):
    """Generate terrain elevation from plate tectonics + layered terrain.

    Components:
    1. Continental/oceanic base with complex coastline breakup
    2. Per-plate tilt and base elevation offsets
    3. Terrain provinces (roughness map controls noise character)
    4. Mountain ranges at convergent boundaries (wide, variable)
    5. Mid-ocean ridges at divergent boundaries
    6. Domain-warped terrain noise (scaled by roughness provinces)

    Produces:
        elevation: (H, W) float32 - elevation in meters
    """
    plate_ids = world["plate_ids"]
    plate_types = world["plate_types"]
    plate_tilt = world["plate_tilt_elevation"]
    plate_offsets = world["plate_base_offsets"]
    boundary_distance = world["boundary_distance"]
    boundary_convergence = world["boundary_convergence"]

    H, W = world.height, world.width
    sx, sy, sz = world.sphere_x, world.sphere_y, world.sphere_z
    is_continental = plate_types[plate_ids].astype(np.float64)

    # =========================================================
    # 1. CONTINENTAL BASE + COMPLEX COASTLINE
    # =========================================================
    # Start with smooth continental/oceanic transition
    shelf_sigma = max(5, W // 60)
    continental_smooth = gaussian_filter(is_continental, sigma=shelf_sigma)

    # --- Large-scale continent shaping ---
    # Low-frequency noise that reshapes entire continent outlines,
    # creating organic coastlines independent of plate polygon shapes.
    continent_shape = spherical_fbm_warped(
        sx, sy, sz,
        frequency=2.0, octaves=3, persistence=0.5,
        warp_strength=0.4, warp_octaves=2,
        seed=params.seed + 280,
    )
    # Asymmetric damping: suppress noise that opposes the dominant character.
    # Deep continental interiors resist negative noise (no inland seas).
    # Deep ocean resists positive noise (no random ocean plateaus).
    # Margins get full noise → organic coastline reshaping.
    interior_strength = np.clip(continental_smooth * 2 - 0.5, 0, 1)
    ocean_strength = np.clip(1.5 - continental_smooth * 2, 0, 1)
    damping = np.ones_like(continent_shape)
    damping -= interior_strength * 0.85 * (continent_shape < 0).astype(float)
    damping -= ocean_strength * 0.85 * (continent_shape > 0).astype(float)

    continental_smooth += 0.35 * continent_shape * damping
    continental_smooth = np.clip(continental_smooth, 0, 1)

    # --- Coastline breakup ---
    # Apply strong noise at continental margins to create peninsulas,
    # bays, archipelagos, irregular shorelines.
    # Medium frequency: large features (peninsulas, gulfs)
    margin_noise_large = spherical_fbm_warped(
        sx, sy, sz,
        frequency=5.0, octaves=4, persistence=0.55, lacunarity=2.0,
        warp_strength=0.3, warp_octaves=2,
        seed=params.seed + 300,
    )
    # Higher frequency: smaller features (islands, inlets)
    margin_noise_small = spherical_fbm(
        sx, sy, sz,
        frequency=12.0, octaves=3, persistence=0.5,
        seed=params.seed + 310,
    )
    margin_noise = 0.8 * margin_noise_large + 0.2 * margin_noise_small

    # Apply most strongly at margins (where continental_smooth is transitional)
    # Bell curve centered at 0.5 (the coastline)
    margin_strength = np.exp(-2.0 * (continental_smooth - 0.5) ** 2 / 0.15)
    # Also apply moderately into the continental shelf area
    shelf_strength = np.clip(continental_smooth * 2, 0, 1) * np.clip(2 - continental_smooth * 2, 0, 1)

    continental_warped = continental_smooth + 0.35 * margin_noise * (margin_strength + 0.3 * shelf_strength)
    continental_warped = np.clip(continental_warped, 0, 1)

    # Push toward bimodal: smoothstep concentrates values at 0 and 1,
    # reducing cells at the threshold → fewer stray islands
    continental_warped = 3 * continental_warped**2 - 2 * continental_warped**3
    # Sharpen to make coastlines more defined
    continental_warped = np.clip((continental_warped - 0.35) / 0.30, 0, 1)
    # Smooth just slightly to remove pixelation
    continental_warped = gaussian_filter(continental_warped, sigma=1.5)
    continental_warped = np.clip(continental_warped, 0, 1)

    # Base elevation from continental/oceanic
    base = (
        continental_warped * params.continental_base
        + (1 - continental_warped) * params.oceanic_base
    )

    # =========================================================
    # 2. PER-PLATE TILT AND BASE OFFSETS
    # =========================================================
    # Plate tilt: creates tilted continental platforms
    # (e.g., Australia: high east, low west)
    base += plate_tilt * continental_warped  # only tilt on land

    # Per-plate base offsets: some plates higher/lower (cratons, basins)
    offsets = plate_offsets[plate_ids].astype(np.float64)
    # Smooth to avoid sharp jumps at plate boundaries
    offsets_smooth = gaussian_filter(offsets, sigma=max(3, W // 100))
    base += offsets_smooth * continental_warped

    # =========================================================
    # 3. LARGE-SCALE CONTINENTAL STRUCTURE
    # =========================================================
    # Broad highlands and lowlands within continents
    continental_structure = spherical_fbm_warped(
        sx, sy, sz,
        frequency=2.5, octaves=3, persistence=0.45,
        warp_strength=0.35, warp_octaves=2,
        seed=params.seed + 400,
    )

    interior = np.clip(continental_warped * 2 - 0.3, 0, 1)
    base += continental_structure * 1200.0 * interior

    # =========================================================
    # 4. TERRAIN PROVINCES (roughness map)
    # =========================================================
    # Low-frequency noise determines terrain character:
    # Low values → smooth plains/basins
    # High values → rugged highlands
    roughness_map = spherical_fbm_warped(
        sx, sy, sz,
        frequency=3.0, octaves=3, persistence=0.5,
        warp_strength=0.25, warp_octaves=2,
        seed=params.seed + 450,
    )
    # Map to [0.1, 1.0] range — never fully zero (some texture everywhere)
    roughness = np.clip(roughness_map * 0.5 + 0.55, 0.1, 1.0)

    # =========================================================
    # 5. MOUNTAIN RANGES (convergent boundaries)
    # =========================================================
    bw_core = params.boundary_mountain_width
    bw_foothills = bw_core * 3.5

    # --- Wobble: shift the ridge laterally so it doesn't follow
    # the plate boundary arc perfectly ---
    wobble_noise = spherical_fbm_warped(
        sx, sy, sz,
        frequency=6.0, octaves=3, persistence=0.5,
        warp_strength=0.2, warp_octaves=2,
        seed=params.seed + 520,
    )
    boundary_distance_wobbled = np.maximum(
        boundary_distance + wobble_noise * bw_core * 0.6, 0
    )

    core_factor = np.exp(-0.5 * (boundary_distance_wobbled / bw_core) ** 2)
    foothill_factor = np.exp(-0.5 * (boundary_distance_wobbled / bw_foothills) ** 2)

    # Variable height along mountain ranges
    mountain_noise = spherical_fbm(
        sx, sy, sz,
        frequency=8.0, octaves=3, persistence=0.5,
        seed=params.seed + 500,
    )
    height_variation = np.clip(0.6 + 0.4 * mountain_noise, 0.25, 1.0)

    # --- Continuity breaks: noise that occasionally drops mountain
    # height to near zero, creating gaps/passes in ranges ---
    continuity_noise = spherical_fbm(
        sx, sy, sz,
        frequency=10.0, octaves=2, persistence=0.5,
        seed=params.seed + 530,
    )
    # Values mostly ~1, but dip toward 0 in some spots
    continuity = np.clip(continuity_noise * 1.5 + 0.7, 0.05, 1.0)

    # Variable width (mountains wider in some sections)
    width_noise = spherical_fbm(
        sx, sy, sz,
        frequency=5.0, octaves=2, persistence=0.5,
        seed=params.seed + 510,
    )
    width_variation = np.clip(0.7 + 0.4 * width_noise, 0.4, 1.3)
    core_factor_var = np.exp(-0.5 * (boundary_distance_wobbled / (bw_core * width_variation)) ** 2)

    conv_strength = np.clip(boundary_convergence, 0, None)
    div_strength = np.clip(-boundary_convergence, 0, None)

    conv_max = np.percentile(conv_strength[conv_strength > 0], 95) if np.any(conv_strength > 0) else 1.0
    conv_norm = np.clip(conv_strength / max(conv_max, 1e-6), 0, 1.5)
    div_max = np.percentile(div_strength[div_strength > 0], 95) if np.any(div_strength > 0) else 1.0
    div_norm = np.clip(div_strength / max(div_max, 1e-6), 0, 1.5)

    mountain_height = params.boundary_mountain_height * height_variation * conv_norm * continuity
    mountains = (
        0.55 * mountain_height * core_factor_var
        + 0.45 * mountain_height * foothill_factor
    )

    # Continental-continental convergence: modest extra height
    both_cont = gaussian_filter(is_continental, sigma=3)
    mountains *= 1.0 + 0.15 * np.clip(both_cont - 0.3, 0, 1) * core_factor

    # Smooth the mountain contribution to soften sharp ridges
    mountains = gaussian_filter(mountains, sigma=max(2, W // 300))

    # Divergent: mid-ocean ridges
    ridge_height = 1200.0 * div_norm * height_variation
    ridges = ridge_height * foothill_factor * (1 - continental_warped)

    base += mountains + ridges

    # =========================================================
    # 6. TERRAIN NOISE (scaled by roughness provinces)
    # =========================================================
    # Domain-warped noise for organic, non-repetitive terrain
    terrain_noise = spherical_fbm_warped(
        sx, sy, sz,
        frequency=params.elevation_noise_frequency,
        octaves=params.elevation_noise_octaves,
        persistence=params.elevation_noise_persistence,
        warp_strength=0.35, warp_octaves=3,
        seed=params.seed + 200,
    )

    # Scale by roughness provinces: plains get less noise, highlands more
    noise_amplitude = (
        continental_warped * 1000.0
        + (1 - continental_warped) * 400.0
    )
    # Roughness provinces modulate amplitude
    noise_amplitude *= roughness

    # Mountains add extra ruggedness
    mountain_ruggedness = 1.0 + 2.0 * core_factor * conv_norm
    noise_amplitude *= mountain_ruggedness

    base += terrain_noise * noise_amplitude

    # =========================================================
    # 7. OCEAN FLOOR FEATURES
    # =========================================================
    ocean_noise = spherical_fbm(
        sx, sy, sz,
        frequency=10.0, octaves=3, persistence=0.5,
        seed=params.seed + 350,
    )
    base += ocean_noise * 350.0 * (1 - continental_warped)

    # =========================================================
    # FINAL: clamp and minimal smoothing
    # =========================================================
    elevation = np.clip(base, params.ocean_depth, params.mountain_height * 1.2)
    elevation = gaussian_filter(elevation, sigma=0.6)

    world["elevation"] = elevation.astype(np.float32)
