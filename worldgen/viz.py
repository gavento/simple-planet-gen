"""Visualization functions for world map layers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from worldgen.layers.biomes import BIOME_COLORS, BIOME_NAMES
from worldgen.world import WorldData

# --- Custom colormaps ---


def _make_terrain_cmap():
    """Earth-like terrain colormap: ocean bathymetry → land elevation."""
    levels = [
        (-8000, "#071e3d"),
        (-4000, "#0a3161"),
        (-1000, "#1a6faa"),
        (-100, "#63b3d4"),
        (0, "#8ec7a6"),
        (100, "#5baa5e"),
        (500, "#3d8b3d"),
        (1000, "#a0a040"),
        (2000, "#c0a030"),
        (3000, "#b08060"),
        (4500, "#a07050"),
        (6000, "#e0d0c0"),
        (8000, "#ffffff"),
    ]
    lo = levels[0][0]
    hi = levels[-1][0]
    norm_levels = [(v - lo) / (hi - lo) for v, _ in levels]
    colors = [c for _, c in levels]
    return mcolors.LinearSegmentedColormap.from_list(
        "terrain_world", list(zip(norm_levels, colors))
    ), lo, hi


TERRAIN_CMAP, TERRAIN_VMIN, TERRAIN_VMAX = _make_terrain_cmap()


def _make_biome_cmap():
    """Discrete colormap for biomes."""
    n = len(BIOME_NAMES)
    colors = [BIOME_COLORS.get(i, "#888888") for i in range(n)]
    cmap = mcolors.ListedColormap(colors, name="biomes")
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n, 1), cmap.N)
    return cmap, norm


def _extent(world: WorldData):
    """Get (left, right, bottom, top) for imshow extent."""
    dlon = 360.0 / world.width / 2
    dlat = 180.0 / world.height / 2
    return [-180 - dlon, 180 + dlon, -90 - dlat, 90 + dlat]


def _setup_ax(ax, title: str, world: WorldData):
    """Configure axis for map display."""
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")


# --- Individual layer plot functions ---


def plot_elevation(world: WorldData, ax=None, show_colorbar=True):
    """Plot elevation with terrain colormap."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    elevation = world["elevation"]
    im = ax.imshow(
        elevation,
        extent=_extent(world),
        cmap=TERRAIN_CMAP,
        vmin=TERRAIN_VMIN,
        vmax=TERRAIN_VMAX,
        interpolation="bilinear",
        origin="upper",
    )
    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)
    _setup_ax(ax, "Elevation", world)
    return im


def _make_land_cmap():
    """Colormap for land-only elevation: 0m to 8000m."""
    levels = [
        (0, "#8ec7a6"),
        (100, "#5baa5e"),
        (500, "#3d8b3d"),
        (1000, "#a0a040"),
        (2000, "#c0a030"),
        (3000, "#b08060"),
        (4500, "#a07050"),
        (6000, "#e0d0c0"),
        (8000, "#ffffff"),
    ]
    hi = levels[-1][0]
    norm_levels = [v / hi for v, _ in levels]
    colors = [c for _, c in levels]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "land_elev", list(zip(norm_levels, colors))
    )
    cmap.set_bad(alpha=0)
    return cmap


LAND_CMAP = _make_land_cmap()


def plot_elevation_land(world: WorldData, ax=None, show_colorbar=True):
    """Plot pre-carve elevation for land only, with uniform ocean color."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    # Use raw (pre-river-carve) elevation if available
    elevation = world["elevation_raw"] if "elevation_raw" in world else world["elevation"]
    land_mask = world["land_mask"]

    ax.set_facecolor("#2266aa")
    display = np.ma.masked_where(~land_mask, elevation)
    im = ax.imshow(
        display,
        extent=_extent(world),
        cmap=LAND_CMAP,
        vmin=0,
        vmax=8000,
        interpolation="nearest",
        origin="upper",
    )
    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)
    _setup_ax(ax, "Land Elevation", world)
    return im


def plot_terrain_carved(world: WorldData, ax=None, show_colorbar=True):
    """Plot post-river carved terrain (land only).

    Shows the elevation after valley carving by rivers.
    Compare with elevation_land (pre-carve) to see river erosion.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    # elevation is the carved version (rivers.py overwrites it)
    # elevation_raw is the pre-carve version
    elevation = world["elevation"]
    land_mask = world["land_mask"]

    ax.set_facecolor("#2266aa")
    display = np.ma.masked_where(~land_mask, elevation)
    im = ax.imshow(
        display,
        extent=_extent(world),
        cmap=LAND_CMAP,
        vmin=0,
        vmax=8000,
        interpolation="nearest",
        origin="upper",
    )
    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)
    _setup_ax(ax, "Terrain (post-river carving)", world)
    return im


def plot_temperature_land(world: WorldData, ax=None, show_colorbar=True):
    """Plot temperature for land only, with uniform ocean color."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    temp = world["temperature"]
    land_mask = world["land_mask"]

    ax.set_facecolor("#2266aa")
    display = np.ma.masked_where(~land_mask, temp)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(alpha=0)
    im = ax.imshow(
        display,
        extent=_extent(world),
        cmap=cmap,
        vmin=-40,
        vmax=35,
        interpolation="nearest",
        origin="upper",
    )
    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Temperature (°C)", shrink=0.7)
    _setup_ax(ax, "Land Temperature", world)
    return im


def plot_plates(world: WorldData, ax=None):
    """Plot tectonic plates with boundaries and motion vectors."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    plate_ids = world["plate_ids"]
    num_plates = world.metadata.get("num_plates", plate_ids.max() + 1)

    cmap = plt.get_cmap("tab20", num_plates)
    im = ax.imshow(
        plate_ids,
        extent=_extent(world),
        cmap=cmap,
        interpolation="nearest",
        origin="upper",
    )

    # Overlay boundaries
    boundary = world["boundary_distance"]
    convergence = world["boundary_convergence"]
    boundary_mask = boundary < 2.0

    # Show convergent (red) and divergent (blue) boundaries
    overlay = np.full((*plate_ids.shape, 4), 0.0)
    conv_boundary = boundary_mask & (convergence > 0.1)
    div_boundary = boundary_mask & (convergence < -0.1)
    overlay[conv_boundary] = [1, 0, 0, 0.7]  # red
    overlay[div_boundary] = [0, 0.3, 1, 0.7]  # blue
    ax.imshow(overlay, extent=_extent(world), interpolation="nearest", origin="upper")

    # Overlay plate motion vectors
    if "plate_centers" in world and "plate_velocities" in world:
        centers = world["plate_centers"]
        vels = world["plate_velocities"]
        for i in range(len(centers)):
            px, py, pz = centers[i]
            # Convert 3D center to lat/lon
            lat_c = np.degrees(np.arcsin(np.clip(pz, -1, 1)))
            lon_c = np.degrees(np.arctan2(py, px))
            # Project 3D velocity to local east/north at this point
            # East unit vector: (-sin(lon), cos(lon), 0)
            # North unit vector: (-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat))
            lat_r = np.radians(lat_c)
            lon_r = np.radians(lon_c)
            east = np.array([-np.sin(lon_r), np.cos(lon_r), 0.0])
            north = np.array([
                -np.sin(lat_r) * np.cos(lon_r),
                -np.sin(lat_r) * np.sin(lon_r),
                np.cos(lat_r),
            ])
            vel_3d = vels[i]
            u = np.dot(vel_3d, east)   # eastward component
            v = np.dot(vel_3d, north)  # northward component
            ax.quiver(
                lon_c, lat_c, u, v,
                color="white", scale=5, width=0.005,
                headwidth=3, headlength=4,
                edgecolor="black", linewidth=0.5,
                zorder=10,
            )

    _setup_ax(ax, "Tectonic Plates (red=convergent, blue=divergent)", world)
    return im


def plot_land_mask(world: WorldData, ax=None):
    """Plot land/ocean mask."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    land_mask = world["land_mask"]
    cmap = mcolors.ListedColormap(["#2266aa", "#55aa55"])
    im = ax.imshow(
        land_mask.astype(int),
        extent=_extent(world),
        cmap=cmap,
        interpolation="nearest",
        origin="upper",
    )
    frac = world.metadata.get("land_fraction", 0)
    _setup_ax(ax, f"Land/Ocean Mask (land: {frac:.1%})", world)
    return im


def plot_temperature(world: WorldData, ax=None, show_colorbar=True):
    """Plot temperature map."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    temp = world["temperature"]
    im = ax.imshow(
        temp,
        extent=_extent(world),
        cmap="RdBu_r",
        vmin=-40,
        vmax=35,
        interpolation="bilinear",
        origin="upper",
    )
    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Temperature (°C)", shrink=0.7)
    _setup_ax(ax, "Annual Mean Temperature", world)
    return im


def plot_winds(world: WorldData, ax=None):
    """Plot wind patterns as quiver plot over land mask."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Background: land/ocean
    plot_land_mask(world, ax=ax)

    # Subsample for quiver readability - target ~25 rows x 50 cols
    step_y = max(1, world.height // 25)
    step_x = max(1, world.width // 50)
    lat_sub = world.lat_grid[::step_y, ::step_x]
    lon_sub = world.lon_grid[::step_y, ::step_x]
    u_sub = world["wind_u"][::step_y, ::step_x]
    v_sub = world["wind_v"][::step_y, ::step_x]

    speed = np.sqrt(u_sub**2 + v_sub**2)
    ax.quiver(
        lon_sub,
        lat_sub,
        u_sub,
        v_sub,
        speed,
        cmap="YlOrRd",
        scale=20,
        alpha=0.7,
        width=0.002,
        headwidth=4,
        headlength=5,
    )
    _setup_ax(ax, "Prevailing Winds", world)


def plot_precipitation(world: WorldData, ax=None, show_colorbar=True):
    """Plot precipitation map."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    precip = world["precipitation"]
    im = ax.imshow(
        precip,
        extent=_extent(world),
        cmap="YlGnBu",
        vmin=0,
        vmax=3000,
        interpolation="bilinear",
        origin="upper",
    )
    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Precipitation (mm/yr)", shrink=0.7)
    _setup_ax(ax, "Annual Precipitation", world)
    return im


def plot_rivers(world: WorldData, ax=None, show_colorbar=True):
    """Plot rivers over terrain using continuous flow accumulation."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Background: elevation
    plot_elevation(world, ax=ax, show_colorbar=show_colorbar)

    flow = world["flow_accumulation"]
    land_mask = world["land_mask"]

    if flow.max() > 0:
        # Use continuous log-scaled flow for smooth river rendering
        land_flow = np.where(land_mask, flow, 0)
        log_flow = np.log1p(land_flow)

        # Find a good threshold: show top ~15% of land cells by flow
        land_log = log_flow[land_mask]
        threshold = np.percentile(land_log, 85)

        # Subtract threshold so only significant flows show
        river_intensity = np.clip((log_flow - threshold) / (log_flow.max() - threshold), 0, 1)

        # Create RGBA overlay: blue rivers with flow-proportional alpha (no dilation)
        overlay = np.zeros((*flow.shape, 4))
        visible = river_intensity > 0.01
        overlay[visible, 0] = 0.0
        overlay[visible, 1] = 0.15
        overlay[visible, 2] = 0.85
        # Non-linear alpha: faint tributaries, bold main rivers
        overlay[visible, 3] = np.clip(river_intensity[visible] ** 0.4 * 0.9, 0.1, 0.9)

        ax.imshow(
            overlay, extent=_extent(world), interpolation="nearest", origin="upper"
        )

    _setup_ax(ax, "Rivers", world)


def plot_rivers_land(world: WorldData, ax=None, show_colorbar=True):
    """Plot rivers over land-only terrain."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Background: land-only elevation
    plot_terrain_carved(world, ax=ax, show_colorbar=show_colorbar)

    flow = world["flow_accumulation"]
    land_mask = world["land_mask"]

    if flow.max() > 0:
        land_flow = np.where(land_mask, flow, 0)
        log_flow = np.log1p(land_flow)

        land_log = log_flow[land_mask]
        threshold = np.percentile(land_log, 85)

        river_intensity = np.clip((log_flow - threshold) / (log_flow.max() - threshold), 0, 1)

        overlay = np.zeros((*flow.shape, 4))
        visible = river_intensity > 0.01
        overlay[visible, 0] = 0.0
        overlay[visible, 1] = 0.15
        overlay[visible, 2] = 0.85
        overlay[visible, 3] = np.clip(river_intensity[visible] ** 0.4 * 0.9, 0.1, 0.9)

        ax.imshow(
            overlay, extent=_extent(world), interpolation="nearest", origin="upper"
        )

    _setup_ax(ax, "Rivers (land)", world)


def plot_biomes(world: WorldData, ax=None):
    """Plot biome classification."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))

    biome_cmap, biome_norm = _make_biome_cmap()
    biome_id = world["biome_id"]
    im = ax.imshow(
        biome_id,
        extent=_extent(world),
        cmap=biome_cmap,
        norm=biome_norm,
        interpolation="nearest",
        origin="upper",
    )

    # Legend
    unique_biomes = np.unique(biome_id)
    handles = []
    for b in sorted(unique_biomes):
        color = BIOME_COLORS.get(b, "#888888")
        name = BIOME_NAMES.get(b, f"Biome {b}")
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=name))
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=7,
        ncol=2,
        framealpha=0.8,
    )
    _setup_ax(ax, "Biomes", world)
    return im


def plot_ocean_currents(world: WorldData, ax=None, show_colorbar=True):
    """Plot sea surface temperature anomaly from ocean currents."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    sst = world["sst_anomaly"]
    land_mask = world["land_mask"]

    # Show SST anomaly over ocean, grey out land
    display = np.where(land_mask, np.nan, sst)
    im = ax.imshow(
        display,
        extent=_extent(world),
        cmap="RdBu_r",
        vmin=-10,
        vmax=10,
        interpolation="bilinear",
        origin="upper",
    )
    # Grey land overlay
    land_overlay = np.full((*land_mask.shape, 4), 0.0)
    land_overlay[land_mask] = [0.5, 0.5, 0.5, 0.8]
    ax.imshow(land_overlay, extent=_extent(world), interpolation="nearest", origin="upper")

    if show_colorbar:
        plt.colorbar(im, ax=ax, label="SST Anomaly (°C)", shrink=0.7)
    _setup_ax(ax, "Ocean Current SST Anomaly", world)
    return im


# --- Dispatcher ---

PLOT_FUNCTIONS = {
    "plates": plot_plates,
    "elevation": plot_elevation,
    "elevation_land": plot_elevation_land,
    "land_mask": plot_land_mask,
    "ocean_currents": plot_ocean_currents,
    "temperature": plot_temperature,
    "temperature_land": plot_temperature_land,
    "winds": plot_winds,
    "precipitation": plot_precipitation,
    "rivers": plot_rivers,
    "rivers_land": plot_rivers_land,
    "terrain_carved": plot_terrain_carved,
    "biomes": plot_biomes,
}


def plot_layer(world: WorldData, layer_name: str, ax=None):
    """Plot a single layer by name."""
    if layer_name not in PLOT_FUNCTIONS:
        raise ValueError(
            f"Unknown plot: {layer_name}. Available: {list(PLOT_FUNCTIONS.keys())}"
        )
    return PLOT_FUNCTIONS[layer_name](world, ax=ax)


def plot_all(world: WorldData, output_dir: str | Path = "output", dpi: int | None = None):
    """Generate and save all available plots.

    Args:
        dpi: Dots per inch. If None, auto-scales so image width ≈ data width.
    """
    if dpi is None:
        # Auto: ~2 pixels per data cell in the plot area
        dpi = max(150, round(world.width / 6))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (name, plot_fn) in enumerate(PLOT_FUNCTIONS.items(), 1):
        # Check if required data exists
        try:
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            plot_fn(world, ax=ax)
            path = output_dir / f"{idx:02d}-{name}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")
        except KeyError as e:
            print(f"  Skipping {name}: missing data {e}")
            plt.close(fig)
