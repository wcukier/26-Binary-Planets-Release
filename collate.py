"""
Collate per-run output files into a single collated_results.npz for each planet.

Scans output/<run_name>/<planet_name>/ and (re)generates collated_results.npz
whenever it is missing or any numbered run subdirectory is newer than it.

Usage:
    python collate.py <run_name>
    python collate.py bin_inj
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from tqdm import tqdm


def needs_update(planet_dir: pathlib.Path, collated_path: pathlib.Path) -> bool:
    if not collated_path.exists():
        return True
    collated_mtime = collated_path.stat().st_mtime
    return any(
        run_dir.stat().st_mtime > collated_mtime
        for run_dir in planet_dir.iterdir()
        if run_dir.is_dir() and run_dir.name.isdigit()
    )


def collate_planet(planet_dir: pathlib.Path, is_binary: bool) -> int:
    summaries, cfgs, elements, distances = [], [], [], []

    run_dirs = sorted(
        (d for d in planet_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    for run_dir in tqdm(run_dirs, desc="  runs", leave=False):
        try:
            with open(run_dir / "summary.json") as f:
                summary = json.load(f)
            with open(run_dir / "config.json") as f:
                cfg = json.load(f)
            element = np.load(run_dir / "elements.npy")
            if is_binary:
                distance = np.load(run_dir / "distances.npy")

            summaries.append(summary)
            cfgs.append(cfg)
            elements.append(element)
            if is_binary:
                distances.append(distance)
        except FileNotFoundError:
            pass

    np.savez(
        str(planet_dir / "collated_results.npz"),
        summaries=summaries,
        cfgs=cfgs,
        elements=elements,
        distances=distances,
    )
    return len(summaries)


def main():
    parser = argparse.ArgumentParser(description="Collate simulation results.")
    parser.add_argument("run_name", help="Subdirectory of output/ to collate (e.g. bin_inj)")
    args = parser.parse_args()

    run_dir = pathlib.Path("output") / args.run_name
    if not run_dir.is_dir():
        print(f"Error: '{run_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    is_binary = "bin" in args.run_name

    planet_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir())

    for planet_dir in tqdm(planet_dirs, desc="planets"):
        collated_path = planet_dir / "collated_results.npz"
        if not needs_update(planet_dir, collated_path):
            tqdm.write(f"  skip  {planet_dir.name}")
            continue
        tqdm.write(f"collate {planet_dir.name}")
        try:
            n = collate_planet(planet_dir, is_binary)
            tqdm.write(f"   -> {n} runs")
        except Exception as e:
            tqdm.write(f"  ERROR {planet_dir.name}: {e}")


if __name__ == "__main__":
    main()
