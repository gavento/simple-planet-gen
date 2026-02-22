"""Core WorldData container and WorldParams configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WorldParams:
    """All parameters for world generation, with reasonable Earth-like defaults."""

    # General
    resolution: int = 1000
    seed: int = 42

    # Tectonics
    num_major_plates: int = 7
    num_minor_plates: int = 20
    continental_ratio: float = 0.4
    plate_noise_weight: float = 0.5
    plate_noise_frequency: float = 6.0
    plate_tilt_strength: float = 800.0  # meters of tilt across a plate
    plate_warp_strength: float = 0.15  # domain warp for plate boundaries
    plate_warp_frequency: float = 3.0  # warp noise frequency

    # Elevation
    elevation_noise_octaves: int = 6
    elevation_noise_frequency: float = 4.0
    elevation_noise_persistence: float = 0.50
    mountain_height: float = 8000.0  # meters
    ocean_depth: float = -8000.0  # meters
    continental_base: float = 200.0  # base continental elevation (m)
    oceanic_base: float = -3500.0  # base oceanic elevation (m)
    boundary_mountain_width: float = 7.0  # degrees
    boundary_mountain_height: float = 4000.0  # meters

    # Land mask
    sea_level: float | None = None  # None = auto-adjust
    target_land_fraction: float = 0.30

    # Temperature
    equator_temperature: float = 30.0  # °C
    pole_temperature: float = -30.0  # °C
    lapse_rate: float = 6.5  # °C per 1000m
    ocean_moderation: float = 0.3  # how much ocean moderates temperature

    # Circulation
    max_wind_speed: float = 1.0  # normalized units
    wind_terrain_drag: float = 0.4  # wind speed reduction over high terrain
    wind_thermal_contrast: float = 0.15  # onshore flow from land/ocean thermal contrast
    wind_deflection_strength: float = 0.3  # topographic deflection of wind

    # Precipitation
    base_precipitation: float = 2000.0  # mm/year at equator over ocean
    orographic_factor: float = 3.0  # rainfall multiplier for windward slopes
    rain_shadow_factor: float = 0.15  # rainfall multiplier for leeward slopes
    moisture_decay_land: float = 0.002  # moisture decay per km over land

    # Ocean currents
    western_boundary_warmth: float = 8.0  # °C max SST anomaly for warm currents
    eastern_boundary_cooling: float = 5.0  # °C max SST anomaly for cold currents
    current_width: float = 10.0  # degrees of lat/lon for current influence

    # Temperature - continentality
    continentality_strength: float = 0.25  # amplification of temp departure inland
    coast_moderation_strength: float = 0.4  # how strongly coast temps follow ocean

    # Lakes
    min_lake_cells: int = 4  # minimum pixel count to keep a lake (smaller → removed)
    glacial_carve_strength: float = 30.0  # meters, glacial valley deepening
    glacial_lake_depth: float = 15.0  # meters, max glacial depression depth

    # Rivers
    river_threshold: float = 0.02  # fraction of max accumulation to show
    pit_fill_epsilon: float = 0.01  # meters, minimum slope in filled terrain
    valley_carve_strength: float = 50.0  # meters, max valley depth scaling
    max_terrain_slope: float = 0.0  # meters/pixel, max slope (0 = disabled)

    # Biomes
    # (uses temperature + precipitation, no extra params)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> WorldParams:
        # Only use keys that are actual fields
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


class WorldData:
    """Container for world map data arrays.

    Stores named layers as numpy arrays, plus metadata.
    Supports save/load to .npz files for incremental layer building.
    """

    def __init__(self, params: WorldParams):
        self.params = params
        self.width = params.resolution
        self.height = params.resolution // 2
        self.seed = params.seed

        # Precompute lat/lon grids (cell centers)
        lat_spacing = 180.0 / self.height
        lon_spacing = 360.0 / self.width
        self.lat = np.linspace(
            90 - lat_spacing / 2, -90 + lat_spacing / 2, self.height
        )
        self.lon = np.linspace(
            -180 + lon_spacing / 2, 180 - lon_spacing / 2, self.width
        )
        self.lat_grid, self.lon_grid = np.meshgrid(self.lat, self.lon, indexing="ij")

        # Precompute 3D unit sphere coordinates
        lat_rad = np.radians(self.lat_grid)
        lon_rad = np.radians(self.lon_grid)
        self.sphere_x = np.cos(lat_rad) * np.cos(lon_rad)
        self.sphere_y = np.cos(lat_rad) * np.sin(lon_rad)
        self.sphere_z = np.sin(lat_rad)

        # Cell area weight (proportional to cos(latitude))
        self.cell_area = np.cos(lat_rad)

        # Named data layers
        self._layers: dict[str, np.ndarray] = {}

        # Extra metadata (per-layer params, etc.)
        self.metadata: dict[str, Any] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        return self._layers[key]

    def __setitem__(self, key: str, value: np.ndarray):
        self._layers[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._layers

    def __delitem__(self, key: str):
        del self._layers[key]

    @property
    def layer_names(self) -> list[str]:
        return list(self._layers.keys())

    def save(self, path: str | Path):
        """Save world data to a compressed .npz file."""
        path = Path(path)
        arrays = dict(self._layers)
        # Store params + metadata as JSON in a special key
        meta = {
            "params": self.params.to_dict(),
            "metadata": self.metadata,
        }
        arrays["__meta__"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
        print(f"Saved world ({len(self._layers)} layers) to {path}")

    @classmethod
    def load(cls, path: str | Path) -> WorldData:
        """Load world data from a .npz file."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["__meta__"]))
        params = WorldParams.from_dict(meta["params"])
        world = cls(params)
        world.metadata = meta.get("metadata", {})
        for key in data.files:
            if key != "__meta__":
                world[key] = data[key]
        print(f"Loaded world ({len(world._layers)} layers) from {path}")
        return world
