"""Layer pipeline registry and runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worldgen.world import WorldData, WorldParams

from worldgen.layers.biomes import generate_biomes
from worldgen.layers.circulation import generate_circulation
from worldgen.layers.elevation import generate_elevation
from worldgen.layers.land_mask import generate_land_mask
from worldgen.layers.ocean_currents import generate_ocean_currents
from worldgen.layers.precipitation import generate_precipitation
from worldgen.layers.rivers import generate_rivers
from worldgen.layers.tectonics import generate_tectonics
from worldgen.layers.temperature import generate_temperature

# (name, function, dependency layer names)
PIPELINE = [
    ("tectonics", generate_tectonics, []),
    ("elevation", generate_elevation, ["tectonics"]),
    ("land_mask", generate_land_mask, ["elevation"]),
    ("circulation", generate_circulation, ["land_mask"]),
    ("ocean_currents", generate_ocean_currents, ["circulation", "land_mask"]),
    ("temperature", generate_temperature, ["elevation", "land_mask", "ocean_currents"]),
    ("precipitation", generate_precipitation, ["temperature", "circulation", "elevation", "land_mask"]),
    ("rivers", generate_rivers, ["elevation", "precipitation", "land_mask"]),
    ("biomes", generate_biomes, ["temperature", "precipitation", "land_mask"]),
]

# Primary array produced by each layer (for dependency checking)
LAYER_PRIMARY_KEY = {
    "tectonics": "plate_ids",
    "elevation": "elevation",
    "land_mask": "land_mask",
    "circulation": "wind_u",
    "ocean_currents": "sst_anomaly",
    "temperature": "temperature",
    "precipitation": "precipitation",
    "rivers": "flow_accumulation",
    "biomes": "biome_id",
}

LAYER_NAMES = [name for name, _, _ in PIPELINE]


def check_dependencies(world: WorldData, layer_name: str) -> list[str]:
    """Return list of missing dependency layers."""
    missing = []
    for _, _, deps in PIPELINE:
        break
    for name, _, deps in PIPELINE:
        if name == layer_name:
            for dep in deps:
                key = LAYER_PRIMARY_KEY[dep]
                if key not in world:
                    missing.append(dep)
            break
    return missing


def run_layer(world: WorldData, layer_name: str, params: WorldParams):
    """Run a single layer, checking dependencies first."""
    missing = check_dependencies(world, layer_name)
    if missing:
        raise ValueError(
            f"Layer '{layer_name}' requires {missing} to be generated first"
        )
    for name, fn, _ in PIPELINE:
        if name == layer_name:
            print(f"  Generating {name}...")
            fn(world, params)
            print(f"  Done: {name}")
            return
    raise ValueError(f"Unknown layer: {layer_name}")


def run_pipeline(world: WorldData, params: WorldParams, up_to: str | None = None):
    """Run the full pipeline (or up to a specific layer)."""
    for name, fn, _ in PIPELINE:
        print(f"  Generating {name}...")
        fn(world, params)
        print(f"  Done: {name}")
        if name == up_to:
            break
