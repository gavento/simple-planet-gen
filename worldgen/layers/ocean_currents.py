"""Heuristic ocean current model producing SST anomalies.

Detects coast orientations and applies known patterns:
- Western boundary warm currents (Gulf Stream, Kuroshio)
- Eastern boundary cold currents (California, Humboldt, Benguela)
- Equatorial upwelling
- High-latitude circumpolar cooling
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from worldgen.world import WorldData, WorldParams


def generate_ocean_currents(world: WorldData, params: WorldParams):
    """Generate sea-surface temperature anomaly from heuristic ocean currents.

    Produces:
        sst_anomaly: (H, W) float32 - SST anomaly in degrees C
    """
    land_mask = world["land_mask"]
    H, W = world.height, world.width
    abs_lat = np.abs(world.lat_grid)
    ocean_mask = ~land_mask

    # Smooth land mask for coast detection (avoid pixel-scale noise)
    land_smooth = gaussian_filter(land_mask.astype(np.float64), sigma=max(3, W // 200))

    # --- Detect coast orientations ---
    # East-west gradient of smoothed land mask
    # Positive = ocean-to-land going east = eastern coast of continent
    # Negative = land-to-ocean going east = western coast of continent
    land_grad_x = np.gradient(land_smooth, axis=1)

    # Proximity to coast (for falloff)
    coast_proximity_tight = gaussian_filter(land_mask.astype(np.float64), sigma=max(2, W // 150))
    coast_proximity_broad = gaussian_filter(land_mask.astype(np.float64), sigma=max(5, W // 40))

    # Ocean proximity to coast (how close this ocean cell is to land)
    ocean_near_coast_tight = coast_proximity_tight * ocean_mask
    ocean_near_coast_broad = coast_proximity_broad * ocean_mask

    # --- Western boundary warm currents ---
    # Eastern coasts of continents, lat 15-55: warm water carried poleward
    eastern_coast = np.clip(land_grad_x, 0, None)
    eastern_coast_spread = gaussian_filter(
        eastern_coast, sigma=(max(3, W // 80), max(5, W // 50))
    )

    # Latitude envelope: strongest 20-50, taper at edges
    warm_lat_env = np.exp(-0.5 * ((abs_lat - 35) / 18) ** 2)
    # Suppress in deep tropics (equatorial, <10 deg)
    warm_lat_env *= np.clip((abs_lat - 8) / 10, 0, 1)

    warm_current = (
        eastern_coast_spread * warm_lat_env * ocean_mask
    )
    # Normalize and scale
    warm_max = np.percentile(warm_current[warm_current > 0], 98) if np.any(warm_current > 0) else 1.0
    warm_current = np.clip(warm_current / max(warm_max, 1e-10), 0, 1)
    warm_anomaly = warm_current * params.western_boundary_warmth

    # --- Eastern boundary cold currents ---
    # Western coasts of continents, lat 10-40: cold upwelling water
    western_coast = np.clip(-land_grad_x, 0, None)
    western_coast_spread = gaussian_filter(
        western_coast, sigma=(max(3, W // 80), max(5, W // 50))
    )

    # Latitude envelope: strongest 15-35
    cold_lat_env = np.exp(-0.5 * ((abs_lat - 25) / 12) ** 2)
    # Also active in subtropics
    cold_lat_env *= np.clip((abs_lat - 5) / 10, 0, 1)

    cold_current = (
        western_coast_spread * cold_lat_env * ocean_mask
    )
    cold_max = np.percentile(cold_current[cold_current > 0], 98) if np.any(cold_current > 0) else 1.0
    cold_current = np.clip(cold_current / max(cold_max, 1e-10), 0, 1)
    cold_anomaly = -cold_current * params.eastern_boundary_cooling

    # --- Equatorial upwelling ---
    # Mild cold anomaly near equator in open ocean
    equatorial_env = np.exp(-0.5 * (abs_lat / 5) ** 2)
    # Stronger where ocean is far from land
    open_ocean = np.clip(1 - coast_proximity_broad * 3, 0, 1) * ocean_mask
    equatorial_anomaly = -2.0 * equatorial_env * open_ocean

    # --- High-latitude circumpolar cooling ---
    # Cold anomaly in 50-65 lat where there's open ocean
    circumpolar_env = np.exp(-0.5 * ((abs_lat - 57) / 8) ** 2)
    circumpolar_anomaly = -3.0 * circumpolar_env * ocean_mask

    # --- Combine ---
    sst_anomaly = warm_anomaly + cold_anomaly + equatorial_anomaly + circumpolar_anomaly

    # Smooth for natural appearance
    sst_anomaly = gaussian_filter(sst_anomaly, sigma=max(2, W // 200))

    # Zero on land
    sst_anomaly *= ocean_mask

    world["sst_anomaly"] = sst_anomaly.astype(np.float32)
