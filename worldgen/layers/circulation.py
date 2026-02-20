"""Atmospheric circulation model (prevailing winds)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from worldgen.world import WorldData, WorldParams


def generate_circulation(world: WorldData, params: WorldParams):
    """Generate prevailing wind patterns from atmospheric cells.

    Model (simplified Earth-like circulation):
    - Hadley cell (0-30°): surface winds toward equator, deflected west (trade winds)
    - Ferrel cell (30-60°): surface winds toward pole, deflected east (westerlies)
    - Polar cell (60-90°): surface winds toward equator, deflected west (easterlies)
    - Coriolis deflection reverses in southern hemisphere

    Produces:
        wind_u: (H, W) float32 - east-west wind component (positive = eastward)
        wind_v: (H, W) float32 - north-south wind component (positive = northward)
    """
    lat = world.lat_grid
    abs_lat = np.abs(lat)
    sign_lat = np.sign(lat)  # +1 NH, -1 SH

    max_wind = params.max_wind_speed

    # --- Meridional (north-south) component ---
    # Hadley: equatorward (toward lat=0)
    # Ferrel: poleward (away from lat=0)
    # Polar: equatorward
    wind_v = np.zeros_like(lat, dtype=np.float64)

    # Hadley cell (0-30°): air moves toward equator at surface
    hadley = abs_lat < 30
    # Strength peaks around 15°
    hadley_strength = np.sin(np.radians(abs_lat * 3)) * hadley
    wind_v -= sign_lat * hadley_strength * max_wind  # toward equator

    # Ferrel cell (30-60°): air moves poleward at surface
    ferrel = (abs_lat >= 30) & (abs_lat < 60)
    ferrel_strength = np.sin(np.radians((abs_lat - 30) * 3)) * ferrel
    wind_v += sign_lat * ferrel_strength * max_wind * 0.6  # toward pole

    # Polar cell (60-90°): air moves equatorward at surface
    polar = abs_lat >= 60
    polar_strength = np.sin(np.radians((abs_lat - 60) * 3)) * polar
    wind_v -= sign_lat * polar_strength * max_wind * 0.4  # toward equator

    # --- Zonal (east-west) component ---
    # Coriolis effect deflects right in NH, left in SH
    # Trade winds: blow from east (negative u)
    # Westerlies: blow from west (positive u)
    # Polar easterlies: blow from east (negative u)
    wind_u = np.zeros_like(lat, dtype=np.float64)

    # Trade winds (0-30°)
    wind_u -= hadley_strength * max_wind * 0.8  # easterly

    # Westerlies (30-60°)
    wind_u += ferrel_strength * max_wind * 1.0  # westerly

    # Polar easterlies (60-90°)
    wind_u -= polar_strength * max_wind * 0.5  # easterly

    # --- ITCZ convergence zone near equator ---
    # Winds weaken near equator (doldrums)
    itcz_damping = np.exp(-0.5 * (lat / 5) ** 2)
    wind_u *= 1 - 0.5 * itcz_damping
    wind_v *= 1 - 0.7 * itcz_damping

    # --- Terrain-aware effects (require elevation + land_mask) ---
    elevation = world["elevation"]
    land_mask = world["land_mask"].astype(np.float64)

    # (a) Topographic drag — mountains and land reduce wind speed
    elev_above_sea = np.clip(elevation, 0, None)
    terrain_drag = 1.0 - params.wind_terrain_drag * np.clip(elev_above_sea / 3000.0, 0, 1) * land_mask
    wind_u *= terrain_drag
    wind_v *= terrain_drag

    # (b) Thermal wind — land-sea pressure gradient creates onshore flow
    # Smooth land mask and compute gradient pointing from ocean toward land
    smooth_land = gaussian_filter(land_mask, sigma=max(3, world.width // 80))
    grad_y = sobel(smooth_land, axis=0)  # d(land)/d(lat)
    grad_x = sobel(smooth_land, axis=1)  # d(land)/d(lon)
    # Scale by latitude: strongest in tropics/subtropics, weak at poles
    thermal_scale = np.cos(np.radians(lat)) ** 2 * params.wind_thermal_contrast
    wind_u += grad_x * thermal_scale * max_wind
    wind_v += grad_y * thermal_scale * max_wind

    # (c) Topographic deflection — wind hitting mountains is rotated parallel to ridges
    smooth_elev = gaussian_filter(elevation.astype(np.float64), sigma=max(3, world.width // 150))
    elev_grad_y = sobel(smooth_elev, axis=0)  # slope in y (lat) direction
    elev_grad_x = sobel(smooth_elev, axis=1)  # slope in x (lon) direction
    # Steepness
    slope_mag = np.sqrt(elev_grad_x**2 + elev_grad_y**2)
    slope_mag_norm = np.clip(slope_mag / (slope_mag.max() + 1e-10), 0, 1)
    # Dot product: wind · slope_gradient (positive = wind blows uphill)
    dot = wind_u * elev_grad_x + wind_v * elev_grad_y
    uphill = np.clip(dot, 0, None)  # only deflect when blowing uphill
    # Rotate wind to be more parallel to contours (perpendicular to gradient)
    deflect = params.wind_deflection_strength * slope_mag_norm
    # Contour-parallel direction: rotate gradient 90° = (-grad_y, grad_x)
    grad_norm = slope_mag + 1e-10
    contour_x = -elev_grad_y / grad_norm
    contour_y = elev_grad_x / grad_norm
    # Add deflection component (scaled by how much wind was blowing uphill)
    uphill_frac = uphill / (np.abs(dot) + 1e-10)
    wind_u += deflect * uphill_frac * contour_x * np.sqrt(wind_u**2 + wind_v**2)
    wind_v += deflect * uphill_frac * contour_y * np.sqrt(wind_u**2 + wind_v**2)

    world["wind_u"] = wind_u.astype(np.float32)
    world["wind_v"] = wind_v.astype(np.float32)
