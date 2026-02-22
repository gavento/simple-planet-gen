#!/usr/bin/env python3
"""World map generator - CLI entry point.

Usage:
    # Full pipeline at development resolution (random seed)
    python generate.py

    # Custom resolution and seed
    python generate.py --resolution 2000 --seed 123

    # Generate only up to a specific layer
    python generate.py --up-to elevation

    # Add a layer to an existing world file
    python generate.py --input output/world_data.npz --layer precipitation

    # Plot from an existing world file
    python generate.py --input output/world_data.npz --plot elevation
    python generate.py --input output/world_data.npz --plot-all

    # Full pipeline with plots
    python generate.py --resolution 1000 --seed 42 --plot-all

    # Custom output directory
    python generate.py --output-dir maps/ --plot-all
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from worldgen.layers import LAYER_NAMES, run_layer, run_pipeline
from worldgen.viz import PLOT_FUNCTIONS, plot_all, plot_layer
from worldgen.world import WorldData, WorldParams

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Realistic random world map generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # General
    parser.add_argument(
        "--resolution", type=int, default=1000, help="Grid width (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: random)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .npz file path (default: <output-dir>/world_data.npz)",
    )

    # Pipeline control
    parser.add_argument(
        "--input", type=str, default=None, help="Load existing world file"
    )
    parser.add_argument(
        "--layer",
        type=str,
        choices=LAYER_NAMES,
        default=None,
        help="Generate only this layer (requires --input with dependencies)",
    )
    parser.add_argument(
        "--up-to",
        type=str,
        choices=LAYER_NAMES,
        default=None,
        help="Run pipeline up to this layer (inclusive)",
    )

    # Output / plotting
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for all output files and plots (default: output)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=list(PLOT_FUNCTIONS.keys()),
        default=None,
        help="Plot a specific layer",
    )
    parser.add_argument(
        "--plot-all", action="store_true", help="Generate all plots (default unless --plot is given)"
    )
    parser.add_argument(
        "--dpi", type=int, default=None,
        help="Plot DPI (default: auto-scale to match data resolution)",
    )

    # Parameter overrides
    parser.add_argument(
        "--num-major-plates", type=int, default=None,
        help="Number of major tectonic plates (default: 7)",
    )
    parser.add_argument(
        "--num-minor-plates", type=int, default=None,
        help="Number of minor tectonic plates (default: 20)",
    )
    parser.add_argument(
        "--land-fraction", type=float, default=None,
        help="Target land fraction 0-1 (default: 0.30)",
    )
    parser.add_argument(
        "--sea-level", type=float, default=None,
        help="Fixed sea level in meters; overrides --land-fraction",
    )
    parser.add_argument(
        "--max-slope", type=float, default=None,
        help="Max terrain slope in meters/pixel; smooths steep cliffs (0 = disabled)",
    )

    args = parser.parse_args()

    # Resolve seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    # Resolve output path: --output wins, otherwise <output-dir>/world_data.npz
    output_dir = Path(args.output_dir)
    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = output_dir / "world_data.npz"

    # --- Load or create world ---
    if args.input:
        print(f"Loading world from {args.input}...")
        world = WorldData.load(args.input)
        params = world.params
    else:
        params = WorldParams(resolution=args.resolution, seed=seed)
        # Apply overrides
        if args.num_major_plates is not None:
            params.num_major_plates = args.num_major_plates
        if args.num_minor_plates is not None:
            params.num_minor_plates = args.num_minor_plates
        if args.land_fraction is not None:
            params.target_land_fraction = args.land_fraction
        if args.sea_level is not None:
            params.sea_level = args.sea_level
        if args.max_slope is not None:
            params.max_terrain_slope = args.max_slope

        world = WorldData(params)

    # --- Generate layers ---
    if args.layer:
        # Single layer mode
        print(f"Running layer: {args.layer}")
        t0 = time.time()
        run_layer(world, args.layer, params)
        print(f"  Completed in {time.time() - t0:.1f}s")
    elif not args.input or (not args.plot and not args.plot_all):
        # Full pipeline (or up to a specific layer)
        print(f"Running pipeline (resolution={params.resolution}, seed={params.seed})...")
        t0 = time.time()
        run_pipeline(world, params, up_to=args.up_to)
        total = time.time() - t0
        print(f"Pipeline completed in {total:.1f}s")

    # --- Save ---
    if args.layer or (not args.plot and not args.plot_all) or not args.input:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        world.save(output_path)

    # --- Plot ---
    dpi = args.dpi if args.dpi else max(150, round(params.resolution / 6))

    if args.plot:
        print(f"Plotting {args.plot} (dpi={dpi})...")
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        plot_layer(world, args.plot, ax=ax)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{args.plot}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")
    else:
        print(f"Generating all plots (dpi={dpi})...")
        plot_all(world, output_dir=output_dir, dpi=dpi)


if __name__ == "__main__":
    main()
