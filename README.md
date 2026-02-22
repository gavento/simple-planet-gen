# worldgen

A simple toy procedural world map generator. Produces Earth-like maps with tectonic plates, elevation, climate, rivers, lakes, and biomes — not physically accurate, but visually plausible.

## Pipeline

The generator runs layers in sequence, each building on the previous:

1. **Tectonics** — Random Voronoi plates on a sphere with assigned velocities and continental/oceanic types.
2. **Elevation** — Plate collisions create mountain ranges at convergent boundaries; continental plates are raised above oceanic ones; multi-octave spherical noise adds terrain detail.
3. **Land mask** — Threshold elevation at a computed sea level to achieve ~30% land fraction.
4. **Lakes** — Connected-component analysis finds inland seas (disconnected ocean patches) and converts them to lakes. Minimax-path outlet channels are carved to the ocean. Glacial depressions are carved at high latitudes/altitudes.
5. **Circulation** — Latitude-band wind model (trades, westerlies, polar easterlies) deflected by topography via pressure-gradient steering.
6. **Ocean currents** — Heuristic SST anomalies from western-boundary warm currents, eastern-boundary cold upwelling, equatorial upwelling, and circumpolar cooling.
7. **Temperature** — Base from latitude + elevation lapse rate, modified by SST anomalies and continentality.
8. **Precipitation** — Moisture advection along wind field with orographic lift/rain shadow, evaporation from warm ocean, and moisture decay over land.
9. **Rivers** — Priority-flood pit filling, D8 flow accumulation, lake pour-point outflow injection, two-pass valley carving.
10. **Biomes** — Whittaker-style classification from temperature and precipitation.

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

# Custom output directory
uv run python generate.py --output-dir maps/

# Plot a single layer
uv run python generate.py --plot elevation

# Custom DPI for plots
uv run python generate.py --dpi 300
```

## Output

All output goes to `--output-dir` (default: `output/`):

```
output/
  world_data.npz          # world data (all layers as numpy arrays)
  01-plates.png           14-biomes_terrain.png
  02-elevation.png        ...
```
