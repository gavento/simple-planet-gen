"""Lake detection: inland sea conversion + glacial lake formation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label, sobel

from worldgen.noise import spherical_fbm
from worldgen.world import WorldData, WorldParams


def _merge_labels_longitude(labels, H, W):
    """Merge connected-component labels across the longitude seam."""
    # Union-find style: map higher label → lower label
    merge = {}

    def root(x):
        while x in merge:
            x = merge[x]
        return x

    for r in range(H):
        if labels[r, 0] > 0 and labels[r, W - 1] > 0:
            l0 = root(int(labels[r, 0]))
            lw = root(int(labels[r, W - 1]))
            if l0 != lw:
                merge[max(l0, lw)] = min(l0, lw)

    if merge:
        for old in sorted(merge.keys(), reverse=True):
            labels[labels == old] = root(old)


def _find_pour_point(water_mask, elevation, land_mask, H, W):
    """Find the lowest land cell on the rim of a water body.

    Returns (pour_row, pour_col, pour_elevation) or None if no rim found.
    """
    # Pad for longitude wrap before dilation
    padded = np.pad(water_mask, ((0, 0), (1, 1)), mode="wrap")
    dilated_padded = binary_dilation(padded, structure=np.ones((3, 3)))
    dilated = dilated_padded[:, 1:-1]

    rim = dilated & ~water_mask & land_mask
    rim_r, rim_c = np.where(rim)
    if len(rim_r) == 0:
        return None

    pour_idx = np.argmin(elevation[rim_r, rim_c])
    return rim_r[pour_idx], rim_c[pour_idx], elevation[rim_r[pour_idx], rim_c[pour_idx]]


def generate_lakes(world: WorldData, params: WorldParams):
    """Detect inland seas and convert to lakes; create glacial lakes.

    1. Connected-component analysis on ocean mask to find disconnected
       water bodies (inland seas). Convert to lakes with flat surface
       at pour-point elevation.
    2. In glacial zones (high latitude / high altitude), carve small
       depressions that become glacial lakes.

    Produces:
        lake_mask: (H, W) bool - True for lake cells
    Also modifies:
        elevation: lake cells set to pour-point level; glacial carving applied
    """
    elevation = world["elevation"].astype(np.float64)
    land_mask = world["land_mask"]
    H, W = world.height, world.width
    cell_area = world.cell_area
    sea_level = world.metadata["sea_level"]

    lake_mask = np.zeros((H, W), dtype=bool)

    # =================================================================
    # 1. INLAND SEA DETECTION AND CONVERSION
    # =================================================================
    ocean = ~land_mask
    labels, n_labels = label(ocean)
    _merge_labels_longitude(labels, H, W)

    # Find the true ocean: largest connected component by area
    unique_labels = np.unique(labels[labels > 0])
    if len(unique_labels) == 0:
        # No ocean at all (unlikely but handle gracefully)
        world["lake_mask"] = lake_mask
        world["elevation"] = elevation.astype(np.float32)
        return

    areas = np.array([np.sum(cell_area[labels == u]) for u in unique_labels])
    ocean_label = unique_labels[np.argmax(areas)]

    # All other water bodies are inland
    inland_labels = unique_labels[unique_labels != ocean_label]

    n_converted = 0
    n_removed = 0
    for lbl in inland_labels:
        water = labels == lbl
        n_cells = np.sum(water)

        # Skip tiny fragments (< min_lake_cells pixels)
        if n_cells < params.min_lake_cells:
            # Too small: fill in as land (raise elevation above sea level)
            elevation[water] = sea_level + 5.0
            land_mask[water] = True
            n_removed += 1
            continue

        # Find lake level from rim elevation profile.
        # Using the minimum rim cell would cluster all inland seas near sea
        # level. Instead, use a low percentile of rim elevations — this models
        # natural damming (sediment, moraines) at the narrowest outlet and
        # gives each lake a surface tied to its surrounding terrain.
        padded = np.pad(water, ((0, 0), (1, 1)), mode="wrap")
        dilated = binary_dilation(padded, structure=np.ones((3, 3)))[:, 1:-1]
        rim = dilated & ~water & land_mask
        rim_r, rim_c = np.where(rim)
        if len(rim_r) == 0:
            continue

        rim_elevs = elevation[rim_r, rim_c]
        lake_level = float(np.percentile(rim_elevs, 20))
        elevation[water] = lake_level - 1.0
        lake_mask[water] = True
        n_converted += 1

    if n_converted > 0 or n_removed > 0:
        print(f"    Lakes: {n_converted} inland seas → lakes, {n_removed} tiny fragments filled")

    # =================================================================
    # 2. GLACIAL LAKE FORMATION
    # =================================================================
    sx, sy, sz = world.sphere_x, world.sphere_y, world.sphere_z
    abs_lat = np.abs(world.lat_grid)

    # Glacial zones: high latitude or high altitude (on land)
    glacial_zone = ((abs_lat > 55) | (elevation > 3000)) & land_mask

    if np.any(glacial_zone) and params.glacial_carve_strength > 0:
        # Depression noise: only negative values create depressions
        carve_noise = spherical_fbm(
            sx, sy, sz,
            frequency=15.0, octaves=3, persistence=0.5,
            seed=params.seed + 700,
        )
        # Only carve (negative noise), scale by local slope
        dx = sobel(elevation, axis=1)
        dy = sobel(elevation, axis=0)
        slope_mag = np.sqrt(dx**2 + dy**2)
        slope_norm = slope_mag / (np.percentile(slope_mag[land_mask], 90) + 1e-10)
        slope_norm = np.clip(slope_norm, 0, 2.0)

        # Carve depressions: negative noise × slope × glacial zone
        carve = params.glacial_carve_strength * np.clip(carve_noise, -1, 0) * slope_norm
        carve *= glacial_zone.astype(np.float64)
        carve = gaussian_filter(carve, sigma=1.5)
        elevation += carve

        # Also add some direct glacial depressions (cirque-like)
        depression_noise = spherical_fbm(
            sx, sy, sz,
            frequency=20.0, octaves=2, persistence=0.5,
            seed=params.seed + 710,
        )
        depressions = params.glacial_lake_depth * np.clip(depression_noise - 0.5, -1, 0)
        depressions *= glacial_zone.astype(np.float64)
        elevation += depressions

        # Find new below-sea-level cells in glacial zones
        new_below = (elevation <= sea_level) & land_mask & ~lake_mask
        if np.any(new_below):
            gl_labels, gl_n = label(new_below)
            for gl_lbl in range(1, gl_n + 1):
                gl_water = gl_labels == gl_lbl
                n_cells = np.sum(gl_water)
                if n_cells < params.min_lake_cells:
                    # Too small: raise back above sea level
                    elevation[gl_water] = sea_level + 2.0
                    continue
                # Set to a flat surface slightly below surrounding terrain
                rim_elev = _find_pour_point(gl_water, elevation, land_mask & ~gl_water, H, W)
                if rim_elev is not None:
                    elevation[gl_water] = rim_elev[2]
                else:
                    elevation[gl_water] = sea_level
                lake_mask[gl_water] = True
                land_mask[gl_water] = False

            n_glacial = np.sum(lake_mask & glacial_zone)
            if n_glacial > 0:
                print(f"    Glacial lakes: {n_glacial} cells")

    # =================================================================
    # STORE RESULTS
    # =================================================================
    world["lake_mask"] = lake_mask
    world["elevation"] = elevation.astype(np.float32)
    world["land_mask"] = land_mask  # updated: tiny fragments filled in
    world.metadata["n_lakes"] = int(n_converted)

