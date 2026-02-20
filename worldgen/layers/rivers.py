"""River network generation using D8 flow accumulation."""

from __future__ import annotations

import numpy as np

from worldgen.world import WorldData, WorldParams

# 8 neighbor offsets: N, NE, E, SE, S, SW, W, NW
_DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
_DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
_DIST = np.array(
    [1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)]
)


def generate_rivers(world: WorldData, params: WorldParams):
    """Generate river network using D8 flow direction + accumulation.

    Produces:
        flow_accumulation: (H, W) float32 - accumulated flow (rainfall-weighted)
        river_mask: (H, W) bool - True for major river cells
    """
    elevation = world["elevation"].astype(np.float64)
    precipitation = world["precipitation"].astype(np.float64)
    land_mask = world["land_mask"]

    H, W = world.height, world.width

    # --- Compute D8 flow direction ---
    # For each cell, find the steepest downhill neighbor
    # Build padded elevation (wrap longitude, block latitude edges)
    padded = np.full((H + 2, W + 2), np.inf, dtype=np.float64)
    padded[1:-1, 1:-1] = elevation
    # Wrap longitude
    padded[1:-1, 0] = elevation[:, -1]
    padded[1:-1, -1] = elevation[:, 0]

    # Compute slope to each of 8 neighbors
    flow_dr = np.zeros((H, W), dtype=np.int32)
    flow_dc = np.zeros((H, W), dtype=np.int32)
    best_slope = np.full((H, W), -np.inf, dtype=np.float64)

    for k in range(8):
        dr, dc = int(_DR[k]), int(_DC[k])
        neighbor = padded[1 + dr : H + 1 + dr, 1 + dc : W + 1 + dc]
        slope = (elevation - neighbor) / _DIST[k]
        better = slope > best_slope
        flow_dr[better] = dr
        flow_dc[better] = dc
        best_slope[better] = slope[better]

    # Cells with no downhill neighbor: mark as sinks (flow to self)
    is_sink = best_slope <= 0
    flow_dr[is_sink] = 0
    flow_dc[is_sink] = 0

    # --- Flow accumulation ---
    # Sort land cells by elevation (highest first) and route flow
    accumulation = np.where(land_mask, precipitation, 0.0)

    # Get land cell indices sorted by descending elevation
    land_rows, land_cols = np.where(land_mask)
    land_elev = elevation[land_rows, land_cols]
    sort_idx = np.argsort(-land_elev)
    sorted_rows = land_rows[sort_idx]
    sorted_cols = land_cols[sort_idx]

    # Route flow from high to low
    for i in range(len(sorted_rows)):
        r, c = sorted_rows[i], sorted_cols[i]
        dr, dc = flow_dr[r, c], flow_dc[r, c]
        if dr == 0 and dc == 0:
            continue  # sink
        nr = r + dr
        nc = (c + dc) % W  # wrap longitude
        if 0 <= nr < H:
            accumulation[nr, nc] += accumulation[r, c]

    # --- Identify major rivers ---
    max_acc = accumulation.max()
    if max_acc > 0:
        threshold = params.river_threshold * max_acc
        river_mask = (accumulation > threshold) & land_mask
    else:
        river_mask = np.zeros((H, W), dtype=bool)

    world["flow_accumulation"] = accumulation.astype(np.float32)
    world["river_mask"] = river_mask
