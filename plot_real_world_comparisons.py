#!/usr/bin/env python

import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-" + str(os.getpid())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditioned-csv",
        default=("/experiment_data/real_world_density_probe/evaluations/"
                 "rw_standard_conditioned_L5/evaluation_raw.csv"))
    parser.add_argument(
        "--baseline-csv",
        default=("/experiment_data/final_evaluations_fixed_real_world/real_world/"
                 "real_world_from_er_L5/evaluation_raw.csv"))
    parser.add_argument("--output-dir", default="/experiment_data/real_world_figures")
    parser.add_argument("--min-n", type=int)
    parser.add_argument("--max-n", type=int)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--graph", action="append", default=[])
    return parser.parse_args()


def load_data(args):
    if not os.path.isfile(args.conditioned_csv):
        raise ValueError("missing conditioned evaluation: " + args.conditioned_csv)
    if not os.path.isfile(args.baseline_csv):
        raise ValueError("missing specialist/greedy evaluation: " + args.baseline_csv)
    conditioned = pd.read_csv(args.conditioned_csv, low_memory=False)
    conditioned["comparison_method"] = "Normal conditioned"
    baselines = pd.read_csv(args.baseline_csv, low_memory=False)
    specialist = baselines[baselines["model_kind"].astype(str) == "specialist"].copy()
    specialist["comparison_method"] = "Specialist"
    greedy = baselines[baselines["algorithm"].astype(str) == "greedy"].copy()
    greedy["comparison_method"] = "Greedy"
    data = pd.concat([conditioned, specialist, greedy], ignore_index=True, sort=False)
    for column in ["n", "initial_edges", "weight", "agent_seed",
                   "final_efficiency", "final_robustness"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["network_generator"] = data["network_generator"].astype(str).str.lower()
    data["graph_id"] = data["graph_id"].astype(str).str.lower()
    if args.min_n is not None:
        data = data[data["n"] >= args.min_n]
    if args.max_n is not None:
        data = data[data["n"] <= args.max_n]
    if args.dataset:
        data = data[data["network_generator"].isin(args.dataset)]
    if args.graph:
        data = data[data["graph_id"].isin(args.graph)]
    return data


def draw_front(axis, frame, method, marker):
    front = frame[frame["comparison_method"] == method].groupby(
        "weight", as_index=False)[["final_efficiency", "final_robustness"]].mean()
    front = front.sort_values("weight")
    if front.empty:
        raise ValueError("missing rows for " + method)
    colours = plt.get_cmap("viridis")(front["weight"].astype(float).values)
    axis.scatter(front["final_efficiency"], front["final_robustness"],
                 c=colours, marker=marker, s=55, linewidths=1.0,
                 edgecolors="black" if marker == "o" else None, zorder=3)
    axis.plot(front["final_efficiency"], front["final_robustness"],
              color="#8a8a8a", linewidth=0.7, alpha=0.45, zorder=1)


def graph_title(frame, dataset, graph):
    n = int(frame["n"].dropna().mode().iloc[0])
    edges = int(frame["initial_edges"].dropna().mode().iloc[0])
    return "{} {}, N={}, m={}, density={:.3f}".format(
        dataset.upper(), graph.upper(), n, edges,
        2.0 * edges / float(n * (n - 1)))


def save_comparison(frame, dataset, graph, output_root):
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), sharex=True, sharey=True)
    for axis, method, title in zip(
            axes, ["Normal conditioned", "Specialist"],
            ["Conditioned mean across 5 seeds", "Specialist mean across 5 seeds"]):
        draw_front(axis, frame, method, "o")
        draw_front(axis, frame, "Greedy", "x")
        axis.set_title(title)
        axis.set_xlabel("Final global efficiency")
        axis.set_ylabel("Final targeted robustness")
        axis.grid(alpha=0.17)
    figure.suptitle(graph_title(frame, dataset, graph), fontsize=15)
    figure.subplots_adjust(left=0.09, right=0.90, bottom=0.18, top=0.84)
    scalar = plt.cm.ScalarMappable(
        norm=plt.Normalize(0.0, 1.0), cmap=plt.get_cmap("viridis"))
    scalar.set_array([])
    bar = figure.colorbar(scalar, ax=list(axes), fraction=0.025, pad=0.025)
    bar.set_label("Efficiency weight w")
    figure.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="black", label="Five-seed learned-policy mean"),
        Line2D([], [], marker="x", linestyle="none", color="black", label="Greedy")],
        loc="lower center", bbox_to_anchor=(0.48, 0.03), ncol=2, frameon=False)
    output = output_root / "specialist" / (
        dataset + "_" + graph + "_conditioned_vs_specialist.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output), bbox_inches="tight")
    plt.close(figure)
    print("Saved " + str(output))


def main():
    args = parse_args()
    data = load_data(args)
    groups = data.groupby(["network_generator", "graph_id"])
    if len(groups) == 0:
        raise ValueError("no real-world graphs match the requested filters")
    for (dataset, graph), frame in groups:
        save_comparison(frame, dataset, graph, Path(args.output_dir))


if __name__ == "__main__":
    main()
