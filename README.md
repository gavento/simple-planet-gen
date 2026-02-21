# worldgen

Realistic random world map generator with tectonic plates, elevation, climate, rivers, and biomes.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync
```

## Usage

```bash
# Full pipeline (random seed)
uv run python generate.py

# Custom resolution and seed
uv run python generate.py --resolution 2000 --seed 123

# Generate all plots
uv run python generate.py --plot-all

# Custom output directory
uv run python generate.py --output-dir maps/ --plot-all

# Plot a single layer
uv run python generate.py --plot elevation

# Load existing world and add a layer
uv run python generate.py --input output/world_data.npz --layer precipitation

# Custom DPI for plots
uv run python generate.py --plot-all --dpi 300
```

## Output

All output goes to `--output-dir` (default: `output/`):

```
output/
  world_data.npz          # world data
  01-plates.png
  02-elevation.png
  03-elevation_land.png
  04-land_mask.png
  05-ocean_currents.png
  06-temperature.png
  07-temperature_land.png
  08-winds.png
  09-precipitation.png
  10-rivers.png
  11-rivers_land.png
  12-terrain_carved.png
  13-biomes.png
  14-biomes_terrain.png
```
