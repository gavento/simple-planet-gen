"""Land/ocean mask from elevation with adjustable sea level."""

from __future__ import annotations

import numpy as np

from worldgen.world import WorldData, WorldParams


def generate_land_mask(world: WorldData, params: WorldParams):
    """Generate land/ocean mask by thresholding elevation.

    If sea_level is None, auto-adjusts to achieve target_land_fraction.

    Produces:
        land_mask: (H, W) bool - True for land, False for ocean
    """
    elevation = world["elevation"]

    if params.sea_level is not None:
        sea_level = params.sea_level
    else:
        # Auto-adjust: find sea level that gives target land fraction
        # Weight by cell area (cos latitude) for accurate fraction
        target = params.target_land_fraction
        sea_level = _find_sea_level(elevation, world.cell_area, target)

    land_mask = elevation > sea_level

    # Report actual land fraction
    land_frac = np.average(land_mask, weights=world.cell_area)
    print(f"    Sea level: {sea_level:.1f}m, land fraction: {land_frac:.1%}")

    world["land_mask"] = land_mask
    world.metadata["sea_level"] = float(sea_level)
    world.metadata["land_fraction"] = float(land_frac)


def _find_sea_level(
    elevation: np.ndarray, cell_area: np.ndarray, target: float
) -> float:
    """Binary search for sea level giving target land fraction."""
    lo, hi = float(elevation.min()), float(elevation.max())
    for _ in range(50):  # plenty of iterations for convergence
        mid = (lo + hi) / 2
        frac = np.average(elevation > mid, weights=cell_area)
        if frac > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
