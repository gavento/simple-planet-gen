"""Biome classification using Whittaker diagram approach."""

from __future__ import annotations

import numpy as np

from worldgen.world import WorldData, WorldParams

# Biome definitions: (id, name, color)
BIOME_DEFS = {
    0: ("Ocean", "#1a5276"),
    1: ("Ice/Glacier", "#d5e8f0"),
    2: ("Tundra", "#a3c1d0"),
    3: ("Boreal Forest", "#2d5a3d"),
    4: ("Temperate Rainforest", "#1e6b3a"),
    5: ("Temperate Forest", "#4a8c5c"),
    6: ("Temperate Grassland", "#b8cc6e"),
    7: ("Mediterranean Shrub", "#c8a860"),
    8: ("Subtropical Forest", "#3d8b4f"),
    9: ("Tropical Rainforest", "#1b5e32"),
    10: ("Tropical Savanna", "#c4b648"),
    11: ("Hot Desert", "#e8d5a3"),
    12: ("Cold Desert/Steppe", "#c4b89a"),
}

BIOME_NAMES = {k: v[0] for k, v in BIOME_DEFS.items()}
BIOME_COLORS = {k: v[1] for k, v in BIOME_DEFS.items()}


def generate_biomes(world: WorldData, params: WorldParams):
    """Classify biomes from temperature and precipitation.

    Uses a simplified Whittaker-style diagram.

    Produces:
        biome_id: (H, W) int32 - biome classification
    """
    temperature = world["temperature"]
    precipitation = world["precipitation"]
    land_mask = world["land_mask"]

    biome = np.zeros_like(temperature, dtype=np.int32)

    # Ocean
    biome[~land_mask] = 0

    # --- Land biomes based on temperature and precipitation ---
    T = temperature
    P = precipitation

    # Ice / Glacier: very cold
    ice = land_mask & (T < -10)
    biome[ice] = 1

    # Tundra: cold, any precipitation
    tundra = land_mask & (T >= -10) & (T < -2)
    biome[tundra] = 2

    # Boreal Forest: cool, moderate+ precipitation
    boreal = land_mask & (T >= -2) & (T < 5) & (P >= 300)
    biome[boreal] = 3

    # Cold Desert/Steppe: cool, dry
    cold_desert = land_mask & (T >= -2) & (T < 5) & (P < 300)
    biome[cold_desert] = 12

    # Temperate zone (5-15°C)
    temp_zone = land_mask & (T >= 5) & (T < 15)

    # Temperate Rainforest: cool, very wet
    biome[temp_zone & (P >= 1500)] = 4

    # Temperate Forest: moderate precipitation
    biome[temp_zone & (P >= 500) & (P < 1500)] = 5

    # Temperate Grassland: drier
    biome[temp_zone & (P >= 200) & (P < 500)] = 6

    # Cold Desert/Steppe: very dry
    biome[temp_zone & (P < 200)] = 12

    # Warm zone (15-22°C)
    warm_zone = land_mask & (T >= 15) & (T < 22)

    # Mediterranean: warm, moderate-dry
    biome[warm_zone & (P < 500)] = 7

    # Subtropical Forest: warm, wet
    biome[warm_zone & (P >= 500) & (P < 1500)] = 8

    # Temperate Rainforest: warm, very wet
    biome[warm_zone & (P >= 1500)] = 4

    # Hot zone (22°C+)
    hot_zone = land_mask & (T >= 22)

    # Hot Desert: hot and dry
    biome[hot_zone & (P < 250)] = 11

    # Tropical Savanna: hot, seasonal rain
    biome[hot_zone & (P >= 250) & (P < 1500)] = 10

    # Tropical Rainforest: hot and very wet
    biome[hot_zone & (P >= 1500)] = 9

    world["biome_id"] = biome
