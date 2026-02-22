"""Projection registry and figure factory for map plots."""

from __future__ import annotations

import matplotlib.pyplot as plt

PROJECTION_NAMES = [
    "equirectangular",
    "mercator",
    "mollweide",
    "robinson",
    "orthographic",
    "homolosine",
]


def make_projected_fig(projection: str = "equirectangular"):
    """Create figure with map axes and colorbar for the given projection.

    Returns (fig, map_axes_list, cbar_ax, transform):
        map_axes_list: list of axes (1 for most projections, 2 for orthographic)
        cbar_ax: dedicated colorbar axes
        transform: cartopy CRS for data coordinates, or None for equirectangular
    """
    if projection == "equirectangular":
        fig = plt.figure(figsize=(16, 8))
        map_ax = fig.add_axes([0.06, 0.08, 0.78, 0.85])
        cbar_ax = fig.add_axes([0.86, 0.08, 0.02, 0.85])
        return fig, [map_ax], cbar_ax, None

    # All other projections need cartopy
    import cartopy.crs as ccrs

    transform = ccrs.PlateCarree()

    if projection == "orthographic":
        crs_west = ccrs.Orthographic(-90, 0)
        crs_east = ccrs.Orthographic(90, 0)
        fig = plt.figure(figsize=(18, 8))
        ax_west = fig.add_subplot(1, 2, 1, projection=crs_west)
        ax_east = fig.add_subplot(1, 2, 2, projection=crs_east)
        ax_west.set_global()
        ax_east.set_global()
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        return fig, [ax_west, ax_east], cbar_ax, transform

    if projection == "mercator":
        crs = ccrs.Mercator()
    elif projection == "mollweide":
        crs = ccrs.Mollweide()
    elif projection == "robinson":
        crs = ccrs.Robinson()
    elif projection == "homolosine":
        crs = ccrs.InterruptedGoodeHomolosine()
    else:
        raise ValueError(
            f"Unknown projection: {projection}. "
            f"Choose from: {PROJECTION_NAMES}"
        )

    fig = plt.figure(figsize=(16, 8))
    map_ax = fig.add_axes([0.05, 0.05, 0.80, 0.88], projection=crs)
    map_ax.set_global()
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    return fig, [map_ax], cbar_ax, transform
