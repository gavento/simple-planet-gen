"""Visualization functions for world map layers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, label

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


def _make_biome_cmap():
    """Discrete colormap for biomes."""
    n = len(BIOME_NAMES)
    colors = [BIOME_COLORS.get(i, "#888888") for i in range(n)]
    cmap = mcolors.ListedColormap(colors, name="biomes")
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n, 1), cmap.N)
    return cmap, norm


# --- Layout helpers ---


def _make_fig_axes():
    """Create figure with fixed map axes and colorbar axes.

    Always reserves the same space so all plots have identical map size.
    Returns (fig, map_ax, cbar_ax).
    """
    fig = plt.figure(figsize=(16, 8))
    map_ax = fig.add_axes([0.06, 0.08, 0.78, 0.85])
    cbar_ax = fig.add_axes([0.86, 0.08, 0.02, 0.85])
    return fig, map_ax, cbar_ax


def _colorbar(im, cbar_ax, label):
    """Add colorbar to the dedicated cbar_ax, or hide it."""
    if cbar_ax is not None:
        plt.colorbar(im, cax=cbar_ax, label=label)
    else:
        plt.colorbar(im, ax=im.axes, label=label, shrink=0.7)


def _hide_cbar(cbar_ax):
    """Hide colorbar axes when plot has no colorbar."""
    if cbar_ax is not None:
        cbar_ax.set_visible(False)


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


def _hillshade(elevation, azimuth=315, altitude=45):
    """Compute hillshade from elevation array. Returns values in [0, 1].

    Shading is attenuated on gentle slopes so flat plains stay clean
    while steep mountain terrain gets full relief.
    """
    from scipy.ndimage import sobel
    dx = sobel(elevation.astype(np.float64), axis=1)
    dy = sobel(elevation.astype(np.float64), axis=0)
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    shade = (
        np.sin(alt)
        + np.cos(alt) * (dx * np.sin(az) - dy * np.cos(az))
        / (np.sqrt(dx**2 + dy**2 + 1) + 1e-10)
    )
    shade = np.clip(shade, 0, 1)
    # Stretch contrast
    lo, hi = np.percentile(shade, [2, 98])
    shade = np.clip((shade - lo) / (hi - lo + 1e-10), 0, 1)

    # Attenuate shading on gentle slopes — emphasize mountains
    slope_mag = np.sqrt(dx**2 + dy**2)
    slope_norm = slope_mag / (np.percentile(slope_mag, 98) + 1e-10)
    slope_norm = np.clip(slope_norm, 0, 1)
    # Flat terrain → shade pushed toward neutral (0.5)
    # Steep terrain → full shade range preserved
    shade = 0.5 + (shade - 0.5) * np.clip(slope_norm ** 0.5, 0.05, 1.0)
    return shade


# --- Individual layer plot functions ---
# All accept ax=None, cbar_ax=None for consistent layout.


def plot_elevation(world: WorldData, ax=None, cbar_ax=None, **_kw):
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
    _colorbar(im, cbar_ax, "Elevation (m)")
    _setup_ax(ax, "Elevation", world)
    return im


def plot_elevation_land(world: WorldData, ax=None, cbar_ax=None, **_kw):
    """Plot pre-carve elevation for land only, with uniform ocean color."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
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
    _render_lake_overlay(world, ax)
    _colorbar(im, cbar_ax, "Elevation (m)")
    _setup_ax(ax, "Land Elevation", world)
    return im


def plot_terrain_carved(world: WorldData, ax=None, cbar_ax=None, **_kw):
    """Plot post-river carved terrain (land only)."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
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
    _render_lake_overlay(world, ax)
    _colorbar(im, cbar_ax, "Elevation (m)")
    _setup_ax(ax, "Terrain (post-river carving)", world)
    return im


def plot_temperature_land(world: WorldData, ax=None, cbar_ax=None, **_kw):
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
    _colorbar(im, cbar_ax, "Temperature (°C)")
    _setup_ax(ax, "Land Temperature", world)
    return im


def plot_plates(world: WorldData, ax=None, cbar_ax=None, **_kw):
    """Plot tectonic plates with boundaries and motion vectors."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    _hide_cbar(cbar_ax)
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
            lat_c = np.degrees(np.arcsin(np.clip(pz, -1, 1)))
            lon_c = np.degrees(np.arctan2(py, px))
            lat_r = np.radians(lat_c)
            lon_r = np.radians(lon_c)
            east = np.array([-np.sin(lon_r), np.cos(lon_r), 0.0])
            north = np.array([
                -np.sin(lat_r) * np.cos(lon_r),
                -np.sin(lat_r) * np.sin(lon_r),
                np.cos(lat_r),
            ])
            vel_3d = vels[i]
            u = np.dot(vel_3d, east)
            v = np.dot(vel_3d, north)
            ax.quiver(
                lon_c, lat_c, u, v,
                color="white", scale=10, width=0.005,
                headwidth=3, headlength=4,
                edgecolor="black", linewidth=0.5,
                zorder=10,
            )

    _setup_ax(ax, "Tectonic Plates (red=convergent, blue=divergent)", world)
    return im


def plot_land_mask(world: WorldData, ax=None, cbar_ax=None, **_kw):
    """Plot land/ocean/lake mask."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    _hide_cbar(cbar_ax)
    land_mask = world["land_mask"]
    lake_mask = world["lake_mask"] if "lake_mask" in world else np.zeros_like(land_mask)

    # 0=ocean, 1=land, 2=lake
    display = np.zeros_like(land_mask, dtype=int)
    display[land_mask] = 1
    display[lake_mask] = 2
    cmap = mcolors.ListedColormap(["#2266aa", "#55aa55", LAKE_COLOR])
    im = ax.imshow(
        display,
        extent=_extent(world),
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
        origin="upper",
    )
    frac = world.metadata.get("land_fraction", 0)
    n_lakes = world.metadata.get("n_lakes", 0)
    title = f"Land/Ocean Mask (land: {frac:.1%}"
    if n_lakes > 0:
        title += f", {n_lakes} lakes"
    title += ")"
    _setup_ax(ax, title, world)
    return im


def plot_temperature(world: WorldData, ax=None, cbar_ax=None, **_kw):
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
    _colorbar(im, cbar_ax, "Temperature (°C)")
    _setup_ax(ax, "Annual Mean Temperature", world)
    return im


def plot_winds(world: WorldData, ax=None, cbar_ax=None, **_kw):
    """Plot wind patterns as quiver plot over land mask."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    _hide_cbar(cbar_ax)

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


def plot_precipitation(world: WorldData, ax=None, cbar_ax=None, **_kw):
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
    _colorbar(im, cbar_ax, "Precipitation (mm/yr)")
    _setup_ax(ax, "Annual Precipitation", world)
    return im


LAKE_COLOR = "#4a90b8"


def _render_lake_overlay(world, ax):
    """Render lake cells as a distinct lighter blue. Shared by land/biome plots."""
    if "lake_mask" not in world:
        return
    lake_mask = world["lake_mask"]
    if not np.any(lake_mask):
        return
    lake_rgba = np.zeros((*lake_mask.shape, 4))
    r, g, b = mcolors.to_rgb(LAKE_COLOR)
    lake_rgba[lake_mask] = [r, g, b, 1.0]
    ax.imshow(
        lake_rgba, extent=_extent(world), interpolation="nearest", origin="upper"
    )


def _render_lake_labels(world, ax):
    """Annotate each lake with its surface elevation."""
    if "lake_mask" not in world:
        return
    lake_mask = world["lake_mask"]
    if not np.any(lake_mask):
        return
    elevation = world["elevation"]
    H, W = world.height, world.width
    labels, n_lakes = label(lake_mask)

    for lbl in range(1, n_lakes + 1):
        rows, cols = np.where(labels == lbl)
        if len(rows) < 4:
            continue
        # Centroid in pixel coords → lon/lat
        cr = rows.mean()
        cc = cols.mean()
        lat = 90.0 - (cr + 0.5) * 180.0 / H
        lon = -180.0 + (cc + 0.5) * 360.0 / W
        elev = float(elevation[rows[0], cols[0]])
        ax.text(
            lon, lat, f"{elev:.0f}m",
            ha="center", va="center",
            fontsize=5, color="white",
            fontweight="bold",
            path_effects=[mpe.withStroke(linewidth=1.5, foreground="black")],
        )


def _render_river_overlay(world, ax):
    """Render river overlay with flow-proportional width.

    Uses multi-scale compositing: high-flow rivers are blurred wider,
    low-flow rivers stay thin (1 pixel). Thickness is soft-scaled by
    the global maximum flow.
    """
    flow = world["flow_accumulation"]
    land_mask = world["land_mask"]
    lake_mask = world["lake_mask"] if "lake_mask" in world else np.zeros_like(land_mask)
    routing_mask = land_mask | lake_mask

    if flow.max() <= 0:
        return

    # Show rivers on land only (not on lake surfaces — lakes render as clean water)
    land_flow = np.where(land_mask, flow, 0)
    log_flow = np.log1p(land_flow)

    land_log = log_flow[land_mask]
    threshold = np.percentile(land_log, 85)
    max_log = log_flow.max()
    if max_log <= threshold:
        return

    river_intensity = np.clip((log_flow - threshold) / (max_log - threshold), 0, 1)

    # Multi-scale width: blur intensity at increasing sigma.
    # High-flow rivers survive wider blurs → appear thicker.
    narrow = river_intensity
    medium = gaussian_filter(river_intensity, sigma=1.5)
    wide = gaussian_filter(river_intensity, sigma=3.0)

    # Composite: take the max across scales, each attenuated
    composite = np.maximum(np.maximum(wide * 0.5, medium * 0.7), narrow)

    overlay = np.zeros((*flow.shape, 4))
    visible = composite > 0.01
    overlay[visible, 0] = 0.0
    overlay[visible, 1] = 0.15
    overlay[visible, 2] = 0.85
    overlay[visible, 3] = np.clip(composite[visible] ** 0.5 * 0.85, 0.05, 0.85)

    ax.imshow(
        overlay, extent=_extent(world), interpolation="nearest", origin="upper"
    )


def plot_rivers(world: WorldData, ax=None, cbar_ax=None, **_kw):
    """Plot rivers over terrain using continuous flow accumulation."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Background: elevation (with colorbar)
    plot_elevation(world, ax=ax, cbar_ax=cbar_ax)
    _render_river_overlay(world, ax)
    _setup_ax(ax, "Rivers", world)


def plot_rivers_land(world: WorldData, ax=None, cbar_ax=None, **_kw):
    """Plot rivers over land-only terrain with lakes."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Background: land-only elevation (with colorbar, includes lake overlay)
    plot_terrain_carved(world, ax=ax, cbar_ax=cbar_ax)
    _render_river_overlay(world, ax)
    _render_lake_labels(world, ax)
    _setup_ax(ax, "Rivers (land)", world)


def plot_biomes(world: WorldData, ax=None, cbar_ax=None, **_kw):
    """Plot biome classification."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    _hide_cbar(cbar_ax)

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
    _render_lake_overlay(world, ax)

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


def plot_biomes_terrain(world: WorldData, ax=None, cbar_ax=None, **_kw):
    """Plot biomes with hillshaded elevation relief.

    Biome colors are darkened/lightened by a hillshade computed from
    elevation, making mountain ranges and terrain features visible.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    _hide_cbar(cbar_ax)

    biome_cmap, biome_norm = _make_biome_cmap()
    biome_id = world["biome_id"]
    elevation = world["elevation"]
    land_mask = world["land_mask"]

    # Render biome colors as RGBA
    mapper = plt.cm.ScalarMappable(norm=biome_norm, cmap=biome_cmap)
    biome_rgba = mapper.to_rgba(biome_id)  # (H, W, 4) float in [0,1]

    # Compute hillshade from elevation
    shade = _hillshade(elevation)

    # Blend: modulate biome RGB by hillshade (land only)
    # shade ~0.5 = neutral, <0.5 = shadow, >0.5 = lit
    blend = 0.4 + 0.6 * shade  # range [0.4, 1.0]
    for c in range(3):  # RGB channels
        biome_rgba[:, :, c] = np.where(
            land_mask,
            np.clip(biome_rgba[:, :, c] * blend, 0, 1),
            biome_rgba[:, :, c],
        )

    # Uniform ocean; lakes get distinct color
    ax.set_facecolor("#2266aa")
    lake_mask = world["lake_mask"] if "lake_mask" in world else np.zeros_like(land_mask)
    ocean_mask = ~land_mask & ~lake_mask
    biome_rgba[ocean_mask, 3] = 0.0  # transparent ocean → shows facecolor
    # Lakes: render as lake color with hillshade
    r, g, b = mcolors.to_rgb(LAKE_COLOR)
    biome_rgba[lake_mask, :3] = [r, g, b]
    biome_rgba[lake_mask, 3] = 1.0

    ax.imshow(
        biome_rgba,
        extent=_extent(world),
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
    _render_river_overlay(world, ax)
    _setup_ax(ax, "Biomes + Terrain", world)


def plot_ocean_currents(world: WorldData, ax=None, cbar_ax=None, **_kw):
    """Plot sea surface temperature anomaly from ocean currents."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8))
    sst = world["sst_anomaly"]
    land_mask = world["land_mask"]
    lake_mask = world["lake_mask"] if "lake_mask" in world else np.zeros_like(land_mask)
    not_ocean = land_mask | lake_mask

    # Show SST anomaly over ocean, grey out land and lakes
    display = np.where(not_ocean, np.nan, sst)
    im = ax.imshow(
        display,
        extent=_extent(world),
        cmap="RdBu_r",
        vmin=-10,
        vmax=10,
        interpolation="bilinear",
        origin="upper",
    )
    # Grey land+lake overlay
    land_overlay = np.full((*land_mask.shape, 4), 0.0)
    land_overlay[land_mask] = [0.5, 0.5, 0.5, 0.8]
    land_overlay[lake_mask] = [0.5, 0.5, 0.5, 0.8]
    ax.imshow(land_overlay, extent=_extent(world), interpolation="nearest", origin="upper")

    _colorbar(im, cbar_ax, "SST Anomaly (°C)")
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
    "biomes_terrain": plot_biomes_terrain,
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
            fig, map_ax, cbar_ax = _make_fig_axes()
            plot_fn(world, ax=map_ax, cbar_ax=cbar_ax)
            path = output_dir / f"{idx:02d}-{name}.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            print(f"  Saved {path}")
        except KeyError as e:
            print(f"  Skipping {name}: missing data {e}")
            plt.close(fig)
