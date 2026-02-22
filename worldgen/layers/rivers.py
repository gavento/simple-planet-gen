"""River network generation using D8 flow accumulation with pit filling."""

from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label

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

    # Seed: all non-land cells (ocean + lakes)
    ocean_rows, ocean_cols = np.where(~land_mask)
    for i in range(len(ocean_rows)):
        r, c = int(ocean_rows[i]), int(ocean_cols[i])
        resolved[r, c] = True
        heapq.heappush(heap, (float(filled[r, c]), r, c))

    # Also seed pole boundary rows
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


# --- Lake-aware flow routing ---


def _find_lake_pour_points(lake_mask, land_mask, elevation, H, W):
    """Find pour point (lowest rim cell) for each lake.

    Returns (lake_labels, pour_points) where pour_points maps
    label -> (row, col).
    """
    lake_labels, n_lakes = label(lake_mask)
    pour_points = {}

    for lbl in range(1, n_lakes + 1):
        water = lake_labels == lbl
        padded = np.pad(water, ((0, 0), (1, 1)), mode="wrap")
        dilated = binary_dilation(padded, structure=np.ones((3, 3)))[:, 1:-1]
        rim = dilated & ~water & land_mask

        rim_r, rim_c = np.where(rim)
        if len(rim_r) == 0:
            continue

        idx = np.argmin(elevation[rim_r, rim_c])
        pour_points[lbl] = (int(rim_r[idx]), int(rim_c[idx]))

    return lake_labels, pour_points


def _fix_pour_point_d8(pour_points, flow_dr, flow_dc, lake_mask, elevation):
    """Override D8 at pour points to flow away from lakes.

    Without this, pour points flow INTO the lake (downhill to lake surface).
    We redirect them to the steepest non-lake neighbor instead.
    """
    H, W = flow_dr.shape
    for _lbl, (pp_r, pp_c) in pour_points.items():
        best_slope = -np.inf
        best_dr, best_dc = 0, 0
        for k in range(8):
            nr = pp_r + int(_DR[k])
            nc = (pp_c + int(_DC[k])) % W
            if nr < 0 or nr >= H:
                continue
            if lake_mask[nr, nc]:
                continue  # must not flow into lake
            slope = (elevation[pp_r, pp_c] - elevation[nr, nc]) / _DIST[k]
            if slope > best_slope:
                best_slope = slope
                best_dr = int(_DR[k])
                best_dc = int(_DC[k])

        flow_dr[pp_r, pp_c] = best_dr
        flow_dc[pp_r, pp_c] = best_dc


def _inject_lake_outflow(
    lake_labels, pour_points, accumulation,
    flow_dr, flow_dc, lake_mask, precipitation, elevation, H, W,
):
    """Route flow through lakes: sum inflow, inject outflow at pour point.

    For each lake, total outflow = (flow draining in from land rim cells)
    + (precipitation falling on lake surface). This outflow is added to
    every cell downstream of the pour point.

    Lakes are processed highest-first so cascading lakes work correctly.
    """
    # Sort by pour point elevation (highest first → drains into lower lakes)
    pp_order = sorted(
        pour_points.items(),
        key=lambda item: elevation[item[1][0], item[1][1]],
        reverse=True,
    )

    for lbl, (pp_r, pp_c) in pp_order:
        water = lake_labels == lbl

        # Inflow: accumulation on lake cells (land rim cells drained into them)
        inflow = float(np.sum(accumulation[water]))
        # Lake's own precipitation
        lake_precip = float(np.sum(precipitation[water]))
        outflow = inflow + lake_precip

        if outflow <= 0:
            continue

        # Trace downstream from pour point, adding outflow to each cell
        r, c = pp_r, pp_c
        visited = set()
        while 0 <= r < H:
            if (r, c) in visited:
                break  # D8 cycle detected
            visited.add((r, c))
            accumulation[r, c] += outflow
            dr = int(flow_dr[r, c])
            dc = int(flow_dc[r, c])
            if dr == 0 and dc == 0:
                break
            nr, nc = r + dr, (c + dc) % W
            if nr < 0 or nr >= H:
                break
            if lake_mask[nr, nc]:
                # Entering another lake — deposit flow there so the
                # downstream lake picks it up when it's processed
                accumulation[nr, nc] += outflow
                break
            r, c = nr, nc


def _enforce_max_slope(elevation, max_slope, H, W):
    """Lower terrain so no slope exceeds max_slope (meters per pixel).

    Iterative relaxation: each pass enforces that no cell is more than
    max_slope * distance above any of its 8 neighbors.  Only lowers
    terrain, never raises it.  Converges when no cell changes.
    """
    result = elevation.copy()
    for iteration in range(200):
        padded = np.full((H + 2, W + 2), np.inf, dtype=np.float64)
        padded[1:-1, 1:-1] = result
        padded[1:-1, 0] = result[:, -1]   # longitude wrap
        padded[1:-1, -1] = result[:, 0]

        prev = result.copy()
        for k in range(8):
            dr, dc = int(_DR[k]), int(_DC[k])
            dist = float(_DIST[k])
            neighbor = padded[1 + dr : H + 1 + dr, 1 + dc : W + 1 + dc]
            max_allowed = neighbor + max_slope * dist
            result = np.minimum(result, max_allowed)

        if np.allclose(result, prev, atol=0.01):
            break

    return result


def generate_rivers(world: WorldData, params: WorldParams):
    """Generate river network using pit-filled D8 flow + valley carving.

    Lakes are treated as collection basins: land rivers drain into them,
    then total outflow is injected at the pour point and traced downstream.
    Micro-noise is added to the filled surface to break grid-aligned
    artifacts in flat regions.

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

    # Preserve original elevation
    world["elevation_raw"] = world["elevation"].copy()

    has_lakes = np.any(lake_mask)
    rng = np.random.RandomState(params.seed + 800)

    # === PASS 1: pit fill + initial flow ===
    # Lakes are sinks (part of ~land_mask): rivers drain into them
    filled = _priority_flood(elevation, land_mask, H, W, params.pit_fill_epsilon)
    # Micro-noise breaks grid-aligned artifacts in flat regions
    filled += rng.uniform(0, params.pit_fill_epsilon * 0.1, (H, W))

    flow_dr, flow_dc = _compute_d8_flow(filled, H, W)

    if has_lakes:
        lake_labels, pour_points = _find_lake_pour_points(
            lake_mask, land_mask, filled, H, W
        )
        _fix_pour_point_d8(pour_points, flow_dr, flow_dc, lake_mask, filled)

    accumulation = _accumulate_flow(
        flow_dr, flow_dc, precipitation, land_mask, filled, H, W
    )

    if has_lakes:
        _inject_lake_outflow(
            lake_labels, pour_points, accumulation,
            flow_dr, flow_dc, lake_mask, precipitation, filled, H, W,
        )

    # === Valley carving ===
    # Only carve land, not lake surfaces.
    # Multi-scale smoothing: narrow gorges + broad floodplains.
    # In flat terrain, the broad pass prevents 1-pixel trenches.
    max_acc = accumulation.max()
    if max_acc > 0 and params.valley_carve_strength > 0:
        log_flow = np.log1p(accumulation / max_acc * 1000.0)
        log_flow_norm = log_flow / max(log_flow.max(), 1e-10)
        carve_raw = params.valley_carve_strength * log_flow_norm * land_mask
        carve_narrow = gaussian_filter(carve_raw, sigma=2.0)
        carve_broad = gaussian_filter(carve_raw, sigma=5.0) * 0.6
        carve_depth = np.maximum(carve_narrow, carve_broad)
        elevation_carved = elevation - carve_depth
    else:
        elevation_carved = elevation

    # === PASS 2: recompute flow on carved surface ===
    filled2 = _priority_flood(elevation_carved, land_mask, H, W, params.pit_fill_epsilon)
    filled2 += rng.uniform(0, params.pit_fill_epsilon * 0.1, (H, W))

    flow_dr2, flow_dc2 = _compute_d8_flow(filled2, H, W)

    if has_lakes:
        lake_labels2, pour_points2 = _find_lake_pour_points(
            lake_mask, land_mask, filled2, H, W
        )
        _fix_pour_point_d8(pour_points2, flow_dr2, flow_dc2, lake_mask, filled2)

    accumulation2 = _accumulate_flow(
        flow_dr2, flow_dc2, precipitation, land_mask, filled2, H, W
    )

    if has_lakes:
        _inject_lake_outflow(
            lake_labels2, pour_points2, accumulation2,
            flow_dr2, flow_dc2, lake_mask, precipitation, filled2, H, W,
        )

    # --- Enforce maximum terrain slope (only lowers, never raises) ---
    if params.max_terrain_slope > 0:
        elevation_carved = _enforce_max_slope(
            elevation_carved, params.max_terrain_slope, H, W,
        )

    # Store carved elevation
    world["elevation"] = elevation_carved.astype(np.float32)

    # --- Identify major rivers (land only, not lake surfaces) ---
    max_acc2 = accumulation2.max()
    if max_acc2 > 0:
        threshold = params.river_threshold * max_acc2
        river_mask = (accumulation2 > threshold) & land_mask
    else:
        river_mask = np.zeros((H, W), dtype=bool)

    world["flow_accumulation"] = accumulation2.astype(np.float32)
    world["river_mask"] = river_mask
