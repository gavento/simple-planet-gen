# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Experimental/toy procedural world map generator. Generates Earth-like planets with tectonic plates, elevation, climate, rivers, and biomes on an equirectangular grid. This is a playground for iterating on terrain realism — visual output quality matters more than code polish.

## Commands

```bash
uv sync                                          # Install dependencies
uv run python generate.py                        # Full pipeline, random seed, all plots
uv run python generate.py --seed 42 --dpi 300    # Reproducible run, high-res plots
uv run python generate.py --up-to elevation      # Stop after a specific layer
uv run python generate.py --input output/world_data.npz --layer precipitation  # Re-run one layer
uv run python generate.py --input output/world_data.npz --plot elevation       # Plot from saved data
```

No test suite exists. Verification is visual: inspect output PNGs after generation.

## Architecture

**Entry point:** `generate.py` — CLI with argparse. Parses args, creates `WorldParams` + `WorldData`, runs pipeline, saves `.npz`, generates plots.

**Core types (`worldgen/world.py`):**
- `WorldParams` — dataclass with all tunable parameters (plates, elevation, climate, etc.). Serialized into `.npz` metadata.
- `WorldData` — container for named numpy arrays (`world["elevation"]`, etc.) + precomputed grids (lat/lon, sphere xyz, cell area). Dict-like interface. Saves/loads as compressed `.npz`.

**Layer pipeline (`worldgen/layers/__init__.py`):**
- `PIPELINE` list of `(name, function, dependencies)` tuples. Each layer function has signature `generate_X(world: WorldData, params: WorldParams)` and stores results via `world["key"] = array`.
- `run_pipeline()` executes layers in order. `run_layer()` runs a single layer with dependency checks.
- Pipeline order: tectonics → elevation → land_mask → circulation → ocean_currents → temperature → precipitation → rivers → biomes

**Noise system (`worldgen/noise.py`):**
- Vectorized 3D Perlin noise operating on numpy arrays.
- `spherical_fbm()` and `spherical_fbm_warped()` generate seamless noise on the unit sphere. Domain warping offsets coordinates before evaluation for organic, non-repetitive features.
- All layer generators use these — the sphere_{x,y,z} coordinates from WorldData are the standard input.

**Visualization (`worldgen/viz.py`):**
- 14 plot functions, all with signature `plot_X(world, ax=None, cbar_ax=None, **kw)`.
- `_make_fig_axes()` creates fixed-layout figures (consistent map + colorbar positioning across all plots).
- `plot_all()` generates numbered PNGs (`01-plates.png` through `14-biomes_terrain.png`).
- `PLOT_FUNCTIONS` dict maps names to functions; add new plots there.

## Conventions

- Use `uv` for all package management (not pip/pip3).
- Layer functions read prerequisites from `world["key"]` and write results back. Adding a new layer: implement the function, add it to `PIPELINE` in `layers/__init__.py`, add a plot function to `viz.py` and register in `PLOT_FUNCTIONS`.
- All spatial computation uses the precomputed unit-sphere coordinates (`world.sphere_x/y/z`). Grid shape is `(height, width)` where `height = resolution // 2`, `width = resolution`.
- Noise seeds are derived from `params.seed + offset` to keep layers reproducible but independent.
- Parameter changes go in `WorldParams` with sensible defaults; CLI overrides go in `generate.py`.
