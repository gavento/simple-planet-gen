"""River network generation using D8 flow accumulation with pit filling."""

from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import gaussian_filter

from worldgen.world import WorldData, WorldParams

# 8 neighbor offsets: N, NE, E, SE, S, SW, W, NW
_DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
_DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
_DIST = np.array(
    [1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)]
)


def _priority_flood(elevation, land_mask, H, W, epsilon=0.01):
    """Fill pits so every land cell drains to the ocean.

    Uses Wang & Liu (2006) priority-flood algorithm.
    Seeds from ocean cells and boundary rows; raises land cells
    just enough to ensure monotonic drainage toward the coast.
    """
    filled = elevation.copy()
    resolved = np.zeros((H, W), dtype=bool)
    heap = []  # (elevation, row, col)

    # Seed: all ocean cells (they are already at correct elevation)
    ocean_rows, ocean_cols = np.where(~land_mask)
    for i in range(len(ocean_rows)):
        r, c = int(ocean_rows[i]), int(ocean_cols[i])
        resolved[r, c] = True
        heapq.heappush(heap, (float(filled[r, c]), r, c))

    # Also seed pole boundary rows (flow cannot exit the poles)
    for c in range(W):
        for r in (0, H - 1):
            if not resolved[r, c]:
                resolved[r, c] = True
                heapq.heappush(heap, (float(filled[r, c]), r, c))

    # Flood inward
    while heap:
        elev, r, c = heapq.heappop(heap)
        for k in range(8):
            nr = r + _DR[k]
            nc = (c + _DC[k]) % W  # wrap longitude
            if nr < 0 or nr >= H:
                continue
            if resolved[nr, nc]:
                continue
            resolved[nr, nc] = True
            new_elev = max(elevation[nr, nc], elev + epsilon)
            filled[nr, nc] = new_elev
            heapq.heappush(heap, (new_elev, nr, nc))

    return filled


def _compute_d8_flow(elevation, H, W):
    """Compute D8 flow direction for each cell.

    Returns flow_dr, flow_dc arrays indicating the direction
    of steepest descent for each cell.
    """
    padded = np.full((H + 2, W + 2), np.inf, dtype=np.float64)
    padded[1:-1, 1:-1] = elevation
    padded[1:-1, 0] = elevation[:, -1]  # wrap longitude
    padded[1:-1, -1] = elevation[:, 0]

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

    # Sinks: no downhill neighbor
    is_sink = best_slope <= 0
    flow_dr[is_sink] = 0
    flow_dc[is_sink] = 0

    return flow_dr, flow_dc


def _accumulate_flow(flow_dr, flow_dc, precipitation, land_mask, elevation, H, W):
    """Route flow from high to low cells, accumulating precipitation."""
    accumulation = np.where(land_mask, precipitation, 0.0)

    land_rows, land_cols = np.where(land_mask)
    land_elev = elevation[land_rows, land_cols]
    sort_idx = np.argsort(-land_elev)
    sorted_rows = land_rows[sort_idx]
    sorted_cols = land_cols[sort_idx]

    for i in range(len(sorted_rows)):
        r, c = sorted_rows[i], sorted_cols[i]
        dr, dc = flow_dr[r, c], flow_dc[r, c]
        if dr == 0 and dc == 0:
            continue
        nr = r + dr
        nc = (c + dc) % W
        if 0 <= nr < H:
            accumulation[nr, nc] += accumulation[r, c]

    return accumulation


def generate_rivers(world: WorldData, params: WorldParams):
    """Generate river network using pit-filled D8 flow + valley carving.

    Two-pass approach:
    1. Fill pits -> compute flow on filled surface -> carve valleys
    2. Recompute flow on carved surface for final river network

    Produces:
        flow_accumulation: (H, W) float32 - accumulated flow
        river_mask: (H, W) bool - True for major river cells
    Also modifies:
        elevation: overwritten with valley-carved version
        elevation_raw: original pre-carve elevation preserved
    """
    elevation = world["elevation"].astype(np.float64)
    precipitation = world["precipitation"].astype(np.float64)
    land_mask = world["land_mask"]
    lake_mask = world["lake_mask"] if "lake_mask" in world else np.zeros_like(land_mask)
    H, W = world.height, world.width

    # For river routing, treat lakes as land (not ocean sinks).
    # Pit-fill seeds from true ocean only; flow routes through flat
    # lake surfaces to the pour point and out.
    routing_mask = land_mask | lake_mask  # "not ocean" for pit-fill seeding

    # Preserve original elevation
    world["elevation_raw"] = world["elevation"].copy()

    # === PASS 1: pit fill + initial flow ===
    filled = _priority_flood(elevation, routing_mask, H, W, params.pit_fill_epsilon)
    flow_dr, flow_dc = _compute_d8_flow(filled, H, W)
    accumulation = _accumulate_flow(
        flow_dr, flow_dc, precipitation, routing_mask, filled, H, W
    )

    # === Valley carving ===
    # Use log of flow accumulation to carve valleys proportionally
    # Only carve land, not lake surfaces
    max_acc = accumulation.max()
    if max_acc > 0 and params.valley_carve_strength > 0:
        log_flow = np.log1p(accumulation / max_acc * 1000.0)
        # Normalize to [0, 1]
        log_flow_norm = log_flow / max(log_flow.max(), 1e-10)
        carve_depth = params.valley_carve_strength * log_flow_norm * land_mask
        # Smooth to create V-shaped valleys, not pixel-thin cuts
        carve_depth = gaussian_filter(carve_depth, sigma=2.0)
        elevation_carved = elevation - carve_depth
    else:
        elevation_carved = elevation

    # === PASS 2: recompute flow on carved surface ===
    filled2 = _priority_flood(elevation_carved, routing_mask, H, W, params.pit_fill_epsilon)
    flow_dr2, flow_dc2 = _compute_d8_flow(filled2, H, W)
    accumulation2 = _accumulate_flow(
        flow_dr2, flow_dc2, precipitation, routing_mask, filled2, H, W
    )

    # Store carved elevation
    world["elevation"] = elevation_carved.astype(np.float32)

    # --- Identify major rivers ---
    max_acc2 = accumulation2.max()
    if max_acc2 > 0:
        threshold = params.river_threshold * max_acc2
        river_mask = (accumulation2 > threshold) & routing_mask
    else:
        river_mask = np.zeros((H, W), dtype=bool)

    world["flow_accumulation"] = accumulation2.astype(np.float32)
    world["river_mask"] = river_mask
