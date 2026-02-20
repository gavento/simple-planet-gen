"""Temperature model with continentality and ocean current influence."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from worldgen.world import WorldData, WorldParams


def generate_temperature(world: WorldData, params: WorldParams):
    """Generate annual average temperature.

    Model:
    - Base temperature from latitude (cosine profile)
    - Altitude lapse rate correction
    - Continental amplification (interiors more extreme)
    - Ocean SST from current anomalies
    - Coastal moderation toward nearby actual ocean temperature

    Produces:
        temperature: (H, W) float32 - temperature in °C
    """
    elevation = world["elevation"]
    land_mask = world["land_mask"]
    sst_anomaly = world["sst_anomaly"]
    H, W = world.height, world.width

    # --- Base temperature from latitude ---
    lat_rad = np.radians(world.lat_grid)
    lat_factor = np.cos(lat_rad) ** 1.2
    base_temp = (
        params.pole_temperature
        + (params.equator_temperature - params.pole_temperature) * lat_factor
    )

    # --- Altitude correction ---
    sea_level = world.metadata.get("sea_level", 0.0)
    altitude_above_sea = np.clip(elevation - sea_level, 0, None)
    altitude_correction = -params.lapse_rate * altitude_above_sea / 1000.0

    # --- Multi-scale continentality index ---
    ocean_float = (~land_mask).astype(np.float64)
    ocean_short = gaussian_filter(ocean_float, sigma=max(2, W // 150))
    ocean_medium = gaussian_filter(ocean_float, sigma=max(5, W // 30))
    ocean_long = gaussian_filter(ocean_float, sigma=max(10, W // 10))
    ocean_influence = 0.3 * ocean_short + 0.4 * ocean_medium + 0.3 * ocean_long
    continentality = np.clip(1.0 - ocean_influence, 0, 1)

    # --- Continental temperature amplification ---
    # Inland areas: departures from global mean are amplified
    global_mean = 14.0  # approximate Earth-like global mean
    temp_land_raw = base_temp + altitude_correction
    departure = temp_land_raw - global_mean
    temp_land = global_mean + departure * (
        1.0 + params.continentality_strength * continentality
    )

    # --- Ocean temperature with SST anomaly ---
    ocean_temp = base_temp + sst_anomaly

    # --- Coastal moderation toward nearby ocean temperature ---
    # Propagate ocean temps over land via Gaussian blur
    # (use NaN-aware approach: blur numerator and denominator separately)
    ocean_temp_field = np.where(land_mask, 0.0, ocean_temp)
    ocean_weight_field = np.where(land_mask, 0.0, 1.0)
    blur_sigma = max(5, W // 25)
    ocean_temp_blurred = gaussian_filter(ocean_temp_field, sigma=blur_sigma)
    ocean_weight_blurred = gaussian_filter(ocean_weight_field, sigma=blur_sigma)
    nearby_ocean_temp = np.where(
        ocean_weight_blurred > 0.01,
        ocean_temp_blurred / ocean_weight_blurred,
        base_temp,
    )

    # Blend land temp toward nearby ocean temp based on coast proximity
    coast_factor = ocean_influence  # high near coast, low inland
    moderation = params.coast_moderation_strength * coast_factor
    temp_land_moderated = temp_land * (1 - moderation) + nearby_ocean_temp * moderation

    # --- Combine land and ocean ---
    temp = np.where(land_mask, temp_land_moderated, ocean_temp)

    world["temperature"] = temp.astype(np.float32)
