"""Atmospheric circulation model (prevailing winds)."""

from __future__ import annotations

import numpy as np

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

    world["wind_u"] = wind_u.astype(np.float32)
    world["wind_v"] = wind_v.astype(np.float32)
