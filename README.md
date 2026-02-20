# worldgen

Realistic random world map generator with tectonic plates, elevation, climate, rivers, and biomes.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync
```

## Usage

```bash
# Full pipeline with default settings
uv run python generate.py

# Custom resolution and seed
uv run python generate.py --resolution 2000 --seed 123

# Generate all plots
uv run python generate.py --plot-all --seed 42

# Plot a single layer
uv run python generate.py --plot elevation

# Generate up to a specific layer
uv run python generate.py --up-to elevation

# Load existing world and add a layer
uv run python generate.py --input world.npz --layer precipitation

# Custom DPI for plots
uv run python generate.py --plot-all --dpi 300
```

## Output

Plots are saved to `output/` with numbered filenames:

```
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
11-biomes.png
```

World data is saved to `world.npz`.
