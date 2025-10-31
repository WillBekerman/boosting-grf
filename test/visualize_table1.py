"""Plot Table 1 benchmarking results stored in JSON format.

This utility consumes the JSON artefact produced by ``benchmark_table1.py``
(which records both the published reference MSEs and the estimates obtained with
Algorithm 1) and renders a multi-panel comparison illustrating how close the
current implementation is to the original numbers.

Usage:
    python test/visualize_table1.py --input table1_results.json \
        --output plots/table1_comparison.png

The script emits a PNG (or any Matplotlib-supported format) containing one
subplot per scenario (12 in total for the AW1/AW2/AW3 settings).
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('matplotlibrc')


def load_table1_results(path: Path) -> Dict[Tuple[str, int, int], Dict[str, Dict[str, float]]]:
    """Load and normalise the Table 1 JSON output.

    Args:
        path: Path to the JSON file produced by ``benchmark_table1.py``.

    Returns:
        A dictionary keyed by scenario tuples ``(dgp, p, n)`` whose values each
        contain ``"reference"`` and ``"ours"`` sub-dictionaries.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    results: Dict[Tuple[str, int, int], Dict[str, Dict[str, float]]] = {}
    for key_str, payload in raw.items():
        scenario = ast.literal_eval(key_str)
        ours = payload.get("ours", {})
        # Backwards compatibility: older JSON used keys "GRF"/"C.GRF"
        # if "GRFBoost" not in ours and "GRF" in ours:
        #     ours = {
        #         "GRFBoost": ours.get("GRF"),
        #         "C.GRFBoost": ours.get("C.GRF"),
        #     }
        #     payload["ours"] = ours
        results[scenario] = payload
    return results


def plot_table1(results: Dict[Tuple[str, int, int], Dict[str, Dict[str, float]]], output: Path) -> None:
    """Render a grid of bar plots comparing reference vs. reproduced MSEs.

    Args:
        results: Parsed results from :func:`load_table1_results`.
        output: Destination path for the generated figure.
    """
    scenarios = sorted(results.keys())
    n_scenarios = len(scenarios)
    ncols = 4
    nrows = int(np.ceil(n_scenarios / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.4, nrows * 3.4), sharey=True)
    axes_iter = axes.flatten()

    width = 0.2
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {
        "WA": default_colors[0],
        "GRF": default_colors[1],
        "BOOST": default_colors[2],
    }
    offsets = {
        "WA": -width,
        "GRF": 0.0,
        "BOOST": width,
    }
    label_map = {
        "WA": "WA baselines",
        "GRF": "GRF (paper)",
        "BOOST": "GRFBoost (this repo)",
    }
    seen_labels = set()
    for ax, scenario in zip(axes_iter, scenarios):
        payload = results[scenario]
        ref = payload["reference"]
        ours = payload["ours"]
        groups = ["WA-1", "WA-2", "GRF", "C.GRF", "GRFBoost", "C.GRFBoost"]
        ref_vals = np.array([ref.get(g, np.nan) for g in groups], dtype=float)
        ours_vals = np.array([
            ref.get(g, np.nan) if g in ("WA-1", "WA-2", "GRF", "C.GRF") else ours.get(g, np.nan)
            for g in groups
        ], dtype=float)
        x = np.arange(len(groups))

        for i, label in enumerate(groups):
            family = (
                "WA"
                if label.startswith("WA")
                else "GRF"
                if label in ("GRF", "C.GRF")
                else "BOOST"
            )
            value = ours_vals[i]
            if np.isnan(value):
                continue
            offset = offsets[family]
            color = color_map[family]
            legend_label = label_map[family] if family not in seen_labels else None
            ax.bar(
                x[i] + offset,
                value,
                width,
                color=color,
                label=legend_label,
            )
            seen_labels.add(family)

        dgp, p, n = scenario
        ax.set_title(f"{dgp.upper()}, p={p}, n={n}")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_ylabel("MSE ×10")
        ax.set_yscale('log')
        ax.grid(True, axis="y", alpha=0.3)

    # Hide any unused axes if scenarios < grid size
    for ax in axes_iter[n_scenarios:]:
        ax.set_visible(False)

    handles, labels = axes_iter[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    #plt.subplots_adjust(top=0.92, right=0.94, hspace=0.4, wspace=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    """Command-line entrypoint to visualise the Table 1 benchmark results."""
    parser = argparse.ArgumentParser(description="Visualise Table 1 benchmark results.")
    parser.add_argument("--input", type=Path, default=Path("table1_results.json"), help="Path to JSON results file")
    parser.add_argument("--output", type=Path, default=Path("plots") / "table1_comparison.png",
                        help="Destination for the generated plot")
    args = parser.parse_args()

    results = load_table1_results(args.input)
    plot_table1(results, args.output)
    print(f"Saved figure to {args.output.resolve()}")


if __name__ == "__main__":
    main()
