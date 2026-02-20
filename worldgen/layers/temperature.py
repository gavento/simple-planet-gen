"""Temperature model based on latitude and altitude."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from worldgen.world import WorldData, WorldParams


def generate_temperature(world: WorldData, params: WorldParams):
    """Generate annual average temperature.

    Model:
    - Base temperature from latitude (cosine profile)
    - Altitude lapse rate correction
    - Ocean thermal moderation (coastal areas less extreme)

    Produces:
        temperature: (H, W) float32 - temperature in °C
    """
    elevation = world["elevation"]
    land_mask = world["land_mask"]

    # --- Base temperature from latitude ---
    # Simple cosine model: max at equator, min at poles
    lat_rad = np.radians(world.lat_grid)
    # cos^1.2 gives slightly flatter tropics, steeper drop at high latitudes
    lat_factor = np.cos(lat_rad) ** 1.2
    base_temp = (
        params.pole_temperature
        + (params.equator_temperature - params.pole_temperature) * lat_factor
    )

    # --- Altitude correction ---
    # Lapse rate: temperature decreases with altitude above sea level
    sea_level = world.metadata.get("sea_level", 0.0)
    altitude_above_sea = np.clip(elevation - sea_level, 0, None)
    altitude_correction = -params.lapse_rate * altitude_above_sea / 1000.0

    # --- Ocean thermal moderation ---
    # Ocean cells: moderate temperature toward latitude-based mean
    # Coastal land: partially moderated
    # Deep inland: unmodified
    # Compute distance from coast (approximate via smoothed land mask)
    ocean_influence = gaussian_filter(
        (~land_mask).astype(np.float64), sigma=8
    )
    moderation = params.ocean_moderation * ocean_influence

    # Ocean moderates toward the latitude base temp
    temp_unmoderated = base_temp + altitude_correction
    temp = temp_unmoderated * (1 - moderation) + base_temp * moderation

    # Ocean cells: no altitude correction, just latitude-based
    temp = np.where(land_mask, temp, base_temp - 2.0)  # ocean slightly cooler

    world["temperature"] = temp.astype(np.float32)
