"""Precipitation model based on moisture transport and orographic effects."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from worldgen.world import WorldData, WorldParams


def generate_precipitation(world: WorldData, params: WorldParams):
    """Generate annual precipitation from moisture advection.

    Model:
    - Moisture source: ocean evaporation (temperature-dependent)
    - Advection by prevailing winds
    - Orographic precipitation (windward slopes get more rain)
    - Rain shadow on leeward slopes
    - ITCZ high rainfall band near equator
    - Subtropical dry zones (~30°)
    - Moisture decays over land (distance from coast)
    - Windward coasts get more moisture

    Produces:
        precipitation: (H, W) float32 - annual precipitation in mm/year
    """
    land_mask = world["land_mask"]
    elevation = world["elevation"]
    temperature = world["temperature"]
    wind_u = world["wind_u"]
    wind_v = world["wind_v"]

    H, W = world.height, world.width

    # --- Base moisture from latitude ---
    lat = world.lat_grid
    abs_lat = np.abs(lat)

    # ITCZ: high moisture near equator
    itcz = np.exp(-0.5 * (abs_lat / 12) ** 2)

    # Subtropical dry zone: low moisture around 25-30°
    subtropical_dry = np.exp(-0.5 * ((abs_lat - 27) / 7) ** 2)

    # Mid-latitude moisture from Ferrel cell
    midlat_moisture = np.exp(-0.5 * ((abs_lat - 50) / 15) ** 2)

    # Subpolar low (around 60°) - convergence zone
    subpolar_moisture = np.exp(-0.5 * ((abs_lat - 60) / 8) ** 2)

    base_moisture = (
        0.85 * itcz
        - 0.55 * subtropical_dry
        + 0.5 * midlat_moisture
        + 0.3 * subpolar_moisture
        + 0.25
    )
    base_moisture = np.clip(base_moisture, 0.05, 1.0)

    # --- Moisture from ocean (windward coast effect) ---
    # Use multiple scales of ocean proximity for gradual decay
    ocean_float = (~land_mask).astype(np.float64)

    # Tight proximity (coastal effect)
    ocean_prox_tight = gaussian_filter(ocean_float, sigma=3)
    # Broad proximity (continental interior dryness)
    ocean_prox_broad = gaussian_filter(ocean_float, sigma=max(3, W // 40))

    # Windward coast detection: where wind blows from ocean onto land
    # Compute gradient of land mask (points from ocean toward land)
    land_float = land_mask.astype(np.float64)
    coast_dy = gaussian_filter(np.gradient(land_float, axis=0), sigma=3)
    coast_dx = gaussian_filter(np.gradient(land_float, axis=1), sigma=3)

    # Wind pushing toward land (dot product of wind with coast-inward normal)
    windward_coast = np.clip(wind_u * coast_dx + wind_v * coast_dy, 0, None)
    windward_coast_norm = windward_coast / (np.percentile(windward_coast, 99) + 1e-10)
    windward_coast_norm = np.clip(windward_coast_norm, 0, 1)

    # Moisture availability: high near ocean, decays over land
    # Windward coasts get extra moisture
    moisture_available = (
        0.4 * ocean_prox_tight**0.6
        + 0.4 * ocean_prox_broad**0.4
        + 0.2
        + 0.4 * windward_coast_norm * ocean_prox_tight
    )
    moisture_available = np.clip(moisture_available, 0.1, 1.5)

    # --- Orographic effect ---
    dy = np.gradient(elevation, axis=0)
    dx = np.gradient(elevation, axis=1)

    # Smooth gradients for more coherent orographic patterns
    dx = gaussian_filter(dx, sigma=2)
    dy = gaussian_filter(dy, sigma=2)

    # Windward slope: wind blowing uphill
    windward = wind_u * dx + wind_v * dy
    windward_positive = np.clip(windward, 0, None)
    windward_negative = np.clip(-windward, 0, None)

    # Normalize
    if np.any(windward_positive > 0):
        wmax = np.percentile(windward_positive[windward_positive > 0], 95)
    else:
        wmax = 1.0
    windward_positive /= max(wmax, 1e-6)
    windward_negative /= max(wmax, 1e-6)

    orographic_boost = 1.0 + params.orographic_factor * windward_positive
    rain_shadow = 1.0 - (1.0 - params.rain_shadow_factor) * np.clip(windward_negative, 0, 1)
    rain_shadow = np.clip(rain_shadow, params.rain_shadow_factor, 1.0)

    # --- Temperature effect on evaporation ---
    temp_factor = np.clip((temperature + 15) / 45, 0.15, 1.5)

    # --- Altitude boost: mountains that catch moisture get extra rain ---
    sea_level = world.metadata.get("sea_level", 0.0)
    altitude = np.clip(elevation - sea_level, 0, None)
    # Moderate altitudes catch moisture; very high mountains are too cold
    altitude_factor = 1.0 + 0.3 * np.exp(-0.5 * ((altitude - 1500) / 1500) ** 2)

    # --- Combine ---
    precipitation = (
        params.base_precipitation
        * base_moisture
        * moisture_available
        * orographic_boost
        * rain_shadow
        * temp_factor
        * altitude_factor
    )

    # Ocean cells: rainfall from base moisture + temp, no orographic
    ocean_precip = params.base_precipitation * base_moisture * temp_factor * 0.8
    precipitation = np.where(land_mask, precipitation, ocean_precip)

    # Smooth for realism
    precipitation = gaussian_filter(precipitation, sigma=2)
    precipitation = np.clip(precipitation, 10, None)  # minimum 10mm/year even in deserts

    world["precipitation"] = precipitation.astype(np.float32)
