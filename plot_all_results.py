#!/usr/bin/env python

import argparse
import glob
import json
import os
import re

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-" + str(os.getpid())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


GENERATORS = ["barabasi_albert", "random_network"]
LABELS = {"barabasi_albert": "BA", "random_network": "ER"}
BASELINES = {
    "random": ("#6b6b6b", "Random"),
    "lowest_degree_product": ("#e67e22", "Lowest-degree product"),
    "fiedler_vector": ("#9b59b6", "Fiedler vector"),
    "effective_resistance": ("#2a9d8f", "Effective resistance"),
}
CMAP, NORM = plt.get_cmap("viridis"), plt.Normalize(0.0, 1.0)
XY = ["final_efficiency", "final_robustness"]
OUTCOME_PATTERN = re.compile(
    r"outcomes_(?P<experiment>.+)_(?P<generator>random_network|barabasi_albert)_"
    r"seed(?P<seed>\d+)_mc(?P<mc>\d+)\.csv$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["final-status", "final-pdfs"])
    parser.add_argument("--project-root", default="/experiment_data/single_graph_corrected")
    parser.add_argument("--evaluation-root", default="/experiment_data/final_evaluations")
    parser.add_argument("--figure-root", default="/experiment_data/final_figures")
    parser.add_argument("--exact-root", default="")
    parser.add_argument("--network-seed", action="append", type=int, default=[])
    parser.add_argument("--agent-seed", type=int, default=168)
    return parser.parse_args()


def load_evaluations(root):
    frames = []
    pattern = os.path.join(os.path.abspath(root), "**", "evaluation_manifest.json")
    for manifest_path in sorted(glob.glob(pattern, recursive=True)):
        raw_path = os.path.join(os.path.dirname(manifest_path), "evaluation_raw.csv")
        if not os.path.isfile(raw_path) or not os.path.getsize(raw_path):
            continue
        with open(manifest_path) as handle:
            spec = json.load(handle).get("experiment", {})
        frame = pd.read_csv(raw_path)
        if frame.empty:
            continue
        frame["_family"] = str(spec.get("family", ""))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def select(frame, name=None, generator=None, family=None):
    mask = pd.Series(True, index=frame.index)
    for column, value in [("evaluation_name", name),
                          ("network_generator", generator),
                          ("_family", family)]:
        if value is not None:
            mask &= frame[column].astype(str) == str(value)
    return frame[mask]


def numeric(frame):
    frame = frame.copy()
    for column in ["weight", "agent_seed", "network_seed"] + XY:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def mean_by_weight(frame):
    if frame.empty:
        return frame
    return frame.groupby("weight", as_index=False)[XY].mean().sort_values("weight")


def handle(marker, label, **style):
    return Line2D([0], [0], marker=marker, linestyle="None", label=label, **style)


def unique_handles(handles):
    unique = {}
    for item in handles:
        unique.setdefault(item.get_label(), item)
    return list(unique.values())


def seed_count(frame, kind):
    rows = frame[frame["model_kind"].astype(str) == kind]
    return int(pd.to_numeric(rows["agent_seed"], errors="coerce").dropna().nunique())


def mid_weight(frame):
    weights = pd.to_numeric(frame["weight"], errors="coerce")
    if weights.dropna().empty:
        return frame
    chosen = min(weights.dropna().unique(), key=lambda value: abs(float(value) - 0.5))
    return frame[np.isclose(weights, float(chosen))]


def draw_front(axis, frame):
    frame, handles = numeric(frame), []
    means = mean_by_weight(frame[frame["model_kind"].astype(str) == "specialist"])
    if not means.empty:
        axis.scatter(means[XY[0]], means[XY[1]], c=means["weight"], cmap=CMAP,
                     norm=NORM, marker="o", s=56, edgecolors="black",
                     linewidths=0.55, zorder=8)
        handles.append(handle(
            "o", "Specialist DQN mean (%d seeds)" % seed_count(frame, "specialist"),
            markerfacecolor="#777777", markeredgecolor="black", markersize=7.0))
    rows = frame[(frame["model_kind"].astype(str) == "preference_conditioned") |
                 (frame["algorithm"].astype(str) == "rnet_dqn_pref")]
    means = mean_by_weight(rows)
    if not means.empty:
        colours = [CMAP(NORM(float(weight))) for weight in means["weight"]]
        axis.scatter(means[XY[0]], means[XY[1]], marker="o", s=56,
                     facecolors="none", edgecolors=colours, linewidths=1.65, zorder=9)
        handles.append(handle(
            "o", "Preference-conditioned DQN mean (%d seeds)" %
            seed_count(frame, "preference_conditioned"), markerfacecolor="none",
            markeredgecolor="#555555", markeredgewidth=1.5, markersize=7.0))
    means = mean_by_weight(frame[frame["algorithm"].astype(str) == "greedy"])
    if not means.empty:
        axis.scatter(means[XY[0]], means[XY[1]], c=means["weight"], cmap=CMAP,
                     norm=NORM, marker="x", s=68, linewidths=1.8, zorder=10)
        handles.append(handle("x", "Greedy", color="#444444",
                              markeredgewidth=1.8, markersize=7.0))
    for algorithm, (colour, label) in BASELINES.items():
        rows = frame[frame["algorithm"].astype(str) == algorithm]
        if rows.empty:
            continue
        point = mid_weight(rows)[XY].mean()
        axis.scatter(point.iloc[0], point.iloc[1], marker="^", s=54, color=colour,
                     edgecolors="black", linewidths=0.45, zorder=7)
        handles.append(handle("^", label, color=colour, markeredgecolor="black",
                              markeredgewidth=0.45, markersize=6.5))
    means = mean_by_weight(frame[frame["model_kind"].astype(str) == "oracle"])
    if not means.empty:
        axis.scatter(means[XY[0]], means[XY[1]], c=means["weight"], cmap=CMAP,
                     norm=NORM, marker="*", s=105, edgecolors="black",
                     linewidths=0.55, zorder=11)
        handles.append(handle("*", "Exact", color="#f4c542",
                              markeredgecolor="black", markersize=10))
    return handles


def tighten(axis, fraction=0.055):
    points = [np.asarray(item.get_offsets()) for item in axis.collections]
    points = [item[np.isfinite(item).all(axis=1)] for item in points
              if item.ndim == 2 and item.shape[1] == 2 and item.size]
    if not points:
        return
    values = np.vstack(points)
    low, high = values.min(axis=0), values.max(axis=0)
    span = np.maximum(high - low, np.maximum(np.abs(low), 1.0) * 0.012)
    axis.set_xlim(low[0] - fraction * span[0], high[0] + fraction * span[0])
    axis.set_ylim(low[1] - fraction * span[1], high[1] + fraction * span[1])


def style_axis(axis, title, row, column, exact=False):
    axis.set_title(title, fontsize=14)
    if row == 2:
        axis.set_xlabel("Final global efficiency" if exact else "Mean efficiency",
                        fontsize=12)
    if column == 0:
        axis.set_ylabel("Final targeted robustness" if exact else "Mean robustness",
                        fontsize=12)
    if not exact:
        axis.tick_params(axis="both", labelsize=10.5)
    axis.grid(alpha=0.16 if exact else 0.17)
    tighten(axis)


def add_colourbar(figure, rectangle):
    scalar = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    scalar.set_array([])
    bar = figure.colorbar(scalar, cax=figure.add_axes(rectangle))
    bar.set_label("Efficiency weight $w$", fontsize=12)
    bar.set_ticks(np.linspace(0.0, 1.0, 11))
    bar.ax.tick_params(labelsize=10.5)


def save_figure(figure, args, family, filename):
    path = os.path.join(os.path.abspath(args.figure_root), "grouped", family,
                        filename + ".pdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.035,
                   metadata={"Creator": "plot_all_results.py",
                             "Title": filename.replace("_", " ")})
    plt.close(figure)
    print("Saved " + path)
    return path


def exact_evaluation_root(project_root):
    for root in [project_root, os.path.dirname(project_root.rstrip(os.sep))]:
        for relative in ["evaluations_exact_n10",
                         os.path.join("exact_stage", "evaluations_exact_n10")]:
            path = os.path.abspath(os.path.join(root, relative))
            if os.path.isdir(path):
                return path


def add_oracle(args, panel, generator, budget):
    root = exact_evaluation_root(args.project_root)
    path = os.path.join(root or "", "exact_n10_L%d" % budget, "evaluation_raw.csv")
    if not root or not os.path.isfile(path) or not os.path.getsize(path):
        return panel
    oracle = pd.read_csv(path)
    oracle = oracle[(oracle["model_kind"].astype(str) == "oracle") &
                    (oracle["network_generator"].astype(str) == generator)].copy()
    if oracle.empty:
        return panel
    keys = ["network_generator", "network_seed"]
    if "graph_id" in panel and "graph_id" in oracle and oracle["graph_id"].notna().any():
        keys.append("graph_id")
    identities = oracle[keys].drop_duplicates()
    paired = panel.merge(identities, on=keys, how="inner")
    return pd.concat([paired, oracle], ignore_index=True, sort=False) if not paired.empty else pd.DataFrame()


def front_grid(args, raw, n, budgets):
    frame = select(raw, family="standard")
    figure, axes = plt.subplots(3, 2, figsize=(12.7, 14.0), squeeze=False)
    handles = []
    for row, budget in enumerate(budgets):
        for column, generator in enumerate(GENERATORS):
            panel = select(frame, "n%d_L%d" % (n, budget), generator)
            if n == 10:
                panel = add_oracle(args, panel, generator, budget)
            if panel.empty:
                plt.close(figure)
                raise ValueError("missing data for n=%d, L=%d, %s" %
                                 (n, budget, generator))
            handles += draw_front(axes[row][column], panel)
            style_axis(axes[row][column], "%s, L=%d" %
                       (LABELS[generator], budget), row, column)
    handles = unique_handles(handles)
    figure.suptitle("Efficiency and robustness fronts, n=%d" % n,
                    fontsize=18, y=0.986)
    figure.subplots_adjust(left=0.08, right=0.89, bottom=0.13, top=0.94,
                           wspace=0.25, hspace=0.35)
    add_colourbar(figure, [0.92, 0.21, 0.014, 0.62])
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.49, 0.018),
                  ncol=min(4, len(handles)), frameon=False, fontsize=10.8,
                  columnspacing=1.35, handletextpad=0.5, labelspacing=0.7)
    return save_figure(figure, args, "standard", "front_grid_n%d" % n)


def draw_cloud(axis, frame):
    model = numeric(frame)
    model = model[model["model_kind"].astype(str).isin(
        ["specialist", "preference_conditioned"])]
    graph_means = model.groupby(["weight", "network_seed"], as_index=False)[XY].mean()
    means = mean_by_weight(model)
    axis.scatter(graph_means[XY[0]], graph_means[XY[1]], c=graph_means["weight"],
                 cmap=CMAP, norm=NORM, marker="o", s=7, alpha=0.11,
                 linewidths=0, zorder=1)
    axis.scatter(means[XY[0]], means[XY[1]], c=means["weight"], cmap=CMAP,
                 norm=NORM, marker="o", s=64, edgecolors="black",
                 linewidths=0.6, zorder=5)
    count = pd.to_numeric(model["agent_seed"], errors="coerce").dropna().nunique()
    return [
        handle("o", "Held-out graph mean across %d seeds" % count,
               color="#9c9c9c", markersize=4, alpha=0.45),
        handle("o", "Mean across held-out graphs", markerfacecolor="#777777",
               markeredgecolor="#444444", markersize=7),
    ]


def draw_dense(axis, panel, raw):
    panel = numeric(panel)
    conditioned = panel[(panel["model_kind"].astype(str) == "preference_conditioned") |
                        (panel["algorithm"].astype(str) == "rnet_dqn_pref")]
    means = mean_by_weight(conditioned)
    name = str(panel["evaluation_name"].iloc[0]).replace("dense_", "", 1)
    generator = str(panel["network_generator"].iloc[0])
    graph_seeds = pd.to_numeric(
        conditioned["network_seed"], errors="coerce").dropna().astype(int)
    comparison = numeric(raw[(raw["_family"].astype(str) == "standard") &
                             (raw["evaluation_name"].astype(str) == name) &
                             (raw["network_generator"].astype(str) == generator) &
                             pd.to_numeric(raw["network_seed"], errors="coerce").isin(
                                 set(graph_seeds))])
    handles = []
    for algorithm, (colour, label) in BASELINES.items():
        rows = comparison[comparison["algorithm"].astype(str) == algorithm]
        if rows.empty:
            continue
        point = mid_weight(rows)[XY].mean()
        axis.scatter(point.iloc[0], point.iloc[1], marker="^", s=29, color=colour,
                     edgecolors="black", linewidths=0.35, zorder=6)
        handles.append(handle("^", label, color=colour, markeredgecolor="black",
                              markeredgewidth=0.35, markersize=5.5))
    axis.scatter(means[XY[0]], means[XY[1]], c=means["weight"], cmap=CMAP,
                 norm=NORM, marker="o", s=23, edgecolors="black",
                 linewidths=0.30, zorder=8)
    handles.append(handle(
        "o", "Conditioned DQN mean (101 weights; %d seeds)" %
        seed_count(conditioned, "preference_conditioned"),
        markerfacecolor="#777777", markeredgecolor="black", markersize=5.5))
    greedy = mean_by_weight(comparison[comparison["algorithm"].astype(str) == "greedy"])
    if not greedy.empty:
        axis.scatter(greedy[XY[0]], greedy[XY[1]], c=greedy["weight"], cmap=CMAP,
                     norm=NORM, marker="x", s=58, linewidths=1.8, zorder=10)
        handles.append(handle("x", "Greedy", color="#444444",
                              markeredgewidth=1.8, markersize=6))
    return handles


def cloud_grid(args, raw, family, n, budgets):
    prefix = "dense" if family == "dense_101" else "cloud"
    frame = select(raw, family=family)
    figure, axes = plt.subplots(3, 2, figsize=(12.6, 14.2), squeeze=False)
    handles = []
    for index, (budget, generator) in enumerate(
            (budget, generator) for budget in budgets for generator in GENERATORS):
        panel = select(frame, "%s_n%d_L%d" % (prefix, n, budget), generator)
        if panel.empty:
            plt.close(figure)
            raise ValueError("missing %s data for n=%d, L=%d, %s" %
                             (family, n, budget, generator))
        handles += (draw_dense(axes.flat[index], panel, raw)
                    if family == "dense_101" else draw_cloud(axes.flat[index], panel))
        axes.flat[index].set_title("%s, n=%d, L=%d" %
                                   (LABELS[generator], n, budget), fontsize=14)
        axes.flat[index].set_xlabel("Final global efficiency")
        if index % 2 == 0:
            axes.flat[index].set_ylabel("Final targeted robustness")
        axes.flat[index].grid(alpha=0.16)
        tighten(axes.flat[index])
    figure.suptitle(
        ("Dense conditioned evaluations (101 weights), n=" if family == "dense_101"
         else "Graph-level specialist outcomes, n=") + str(n), fontsize=18, y=0.98)
    figure.subplots_adjust(left=0.075, right=0.89, bottom=0.14, top=0.94,
                           wspace=0.24, hspace=0.28)
    add_colourbar(figure, [0.92, 0.17, 0.014, 0.70])
    figure.legend(handles=unique_handles(handles), loc="lower center",
                  bbox_to_anchor=(0.49, 0.018), ncol=3, frameon=False,
                  fontsize=10.5)
    return save_figure(figure, args, family,
                       ("dense_all_n" if family == "dense_101" else "cloud_grid_n") + str(n))


def draw_size(axis, frame, metric):
    frame = numeric(frame)
    conditioned = frame[frame["model_kind"].astype(str) == "preference_conditioned"]
    weights = pd.to_numeric(conditioned["weight"], errors="coerce")
    keep = np.logical_or.reduce([np.isclose(weights.values, value)
                                 for value in (0.0, 0.6, 1.0)])
    for weight, rows in conditioned.loc[keep].groupby("weight"):
        grouped = rows.groupby("n")[metric]
        mean = grouped.mean().sort_index()
        lower = grouped.quantile(0.10).reindex(mean.index)
        upper = grouped.quantile(0.90).reindex(mean.index)
        colour = CMAP(NORM(float(weight)))
        axis.fill_between(mean.index.values, lower.values, upper.values,
                          color=colour, alpha=0.07, linewidth=0)
        axis.plot(mean.index.values, mean.values, color=colour, marker="o",
                  markersize=4.2, linewidth=1.65)
    for algorithm, (colour, unused) in BASELINES.items():
        rows = frame[frame["algorithm"].astype(str) == algorithm]
        if not rows.empty:
            mean = rows.groupby("n")[metric].mean().sort_index()
            axis.plot(mean.index.values, mean.values, color=colour, marker="^",
                      markersize=4.5, linewidth=1.25)


def size_grid(args, raw, tau, label):
    frame = select(raw, family="size_generalization")
    selected = frame[frame["evaluation_name"].astype(str).str.endswith(tau)]
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), squeeze=False)
    for row, generator in enumerate(GENERATORS):
        rows = select(selected, generator=generator)
        for column, metric in enumerate(XY):
            draw_size(axes[row][column], rows, metric)
            axes[row][column].set_title("%s — %s" %
                (LABELS[generator], "efficiency" if column == 0 else "robustness"),
                fontsize=14)
            axes[row][column].set_xlabel("Evaluation graph size n")
            axes[row][column].set_ylabel("Final " +
                ("global efficiency" if column == 0 else "targeted robustness"))
            axes[row][column].set_xticks([20, 30, 40])
            axes[row][column].grid(alpha=0.17)
    figure.suptitle("Zero-shot size transfer — edge budget " + label,
                    fontsize=18, y=0.98)
    figure.subplots_adjust(left=0.08, right=0.89, bottom=0.19, top=0.90,
                           wspace=0.25, hspace=0.34)
    add_colourbar(figure, [0.92, 0.24, 0.013, 0.60])
    bar = figure.axes[-1]
    bar.set_yticks([0.0, 0.6, 1.0])
    bar.set_yticklabels(["0.0 (robustness)", "0.6 (mid)", "1.0 (efficiency)"])
    handles = [Line2D([0], [0], color="#555555", marker="o", markersize=4,
                      label="Conditioned: w = 0.0, 0.6, 1.0; 10–90% graph band")]
    handles += [Line2D([0], [0], color=BASELINES[name][0], marker="^",
                       markersize=5, label=BASELINES[name][1]) for name in BASELINES
                if (selected["algorithm"].astype(str) == name).any()]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.49, 0.025),
                  ncol=3, frameon=False, fontsize=10.8)
    return save_figure(figure, args, "size_generalization",
                       "size_generalization_" + tau)


def pareto_front(points):
    values = np.unique(np.round(np.asarray(points, dtype=float), 12), axis=0)
    ordered = values[np.lexsort((-values[:, 1], -values[:, 0]))]
    keep, best_y = [], -np.inf
    for point in ordered:
        if point[1] > best_y + 1e-10:
            keep.append(point)
            best_y = point[1]
    return np.asarray(sorted(keep, key=lambda point: (point[0], -point[1])))


def concentric(axis, frame, marker, largest, smallest, zorder):
    grouped = {}
    for row in frame.to_dict("records"):
        point = tuple(round(float(row[column]), 12) for column in XY)
        grouped.setdefault(point, []).append(float(row["weight"]))
    for point, weights in grouped.items():
        weights = sorted(set(weights))
        for index, weight in enumerate(weights):
            size = largest if len(weights) == 1 else largest - index * (
                largest - smallest) / max(1, len(weights) - 1)
            axis.scatter(*[[value] for value in point], s=size, marker=marker,
                         facecolors="none", edgecolors=[CMAP(NORM(weight))],
                         linewidths=1.25, zorder=zorder + index * 0.01)


def draw_exact(axis, outcomes, learned, agent_seed):
    outcomes, learned = numeric(outcomes), numeric(learned)
    front = pareto_front(outcomes[XY].values)
    axis.scatter(outcomes[XY[0]], outcomes[XY[1]], s=8, color="#c7c7c7",
                 alpha=0.38, linewidths=0, zorder=1)
    if len(front):
        front = front[np.argsort(front[:, 0])]
        axis.plot(front[:, 0], front[:, 1], color="#c62828", linewidth=1.0, zorder=4)
        axis.scatter(front[:, 0], front[:, 1], s=24, color="#d32f2f",
                     edgecolors="black", linewidths=0.35, zorder=5)
    specialist = learned[(learned["model_kind"].astype(str) == "specialist") |
                         (learned["algorithm"].astype(str) == "rnet_dqn")]
    specialist = specialist[pd.to_numeric(
        specialist["agent_seed"], errors="coerce") == int(agent_seed)]
    means = mean_by_weight(specialist)
    if not means.empty:
        concentric(axis, means, "o", 32.0, 10.0, 7.0)
    greedy = mean_by_weight(learned[learned["algorithm"].astype(str) == "greedy"])
    if not greedy.empty:
        concentric(axis, greedy, "D", 34.0, 11.0, 8.0)
    return [
        handle(".", "All attainable outcomes", color="#c7c7c7", markersize=8),
        Line2D([0], [0], marker="o", linestyle="-", color="#c62828",
               markerfacecolor="#d32f2f", markeredgecolor="black", linewidth=1.0,
               markersize=5, label="Exact nondominated outcomes"),
        handle("o", "DQN weights (seed %d)" % int(agent_seed),
               markerfacecolor="none", markeredgecolor="#2a9d8f", markersize=6),
        handle("D", "Greedy weights", markerfacecolor="none",
               markeredgecolor="#555555", markersize=6),
    ]


def outcomes(args):
    roots = [os.path.abspath(args.exact_root) if args.exact_root else None,
             os.path.join(args.project_root, "exact_attainable_diagnostic"),
             os.path.join(os.path.dirname(args.project_root), "exact_attainable_diagnostic")]
    root = next((path for path in roots if path and os.path.isdir(path)), None)
    items = []
    for path in sorted(glob.glob(os.path.join(root or "", "outcomes_*.csv"))):
        match = OUTCOME_PATTERN.match(os.path.basename(path))
        if not match:
            continue
        item = match.groupdict()
        budget = re.search(r"_L(\d+)$", item["experiment"])
        item.update(seed=int(item["seed"]), budget=int(budget.group(1)), path=path)
        items.append(item)
    return items


def exact_grids(args, raw):
    frame, metadata = select(raw, family="exact"), outcomes(args)
    seeds = sorted(set(args.network_seed)) if args.network_seed else sorted(
        {item["seed"] for item in metadata})
    outputs = []
    for seed in seeds:
        figure, axes = plt.subplots(3, 2, figsize=(12.7, 14.0), squeeze=False)
        handles = []
        for row, budget in enumerate([2, 3, 5]):
            for column, generator in enumerate(GENERATORS):
                item = next((value for value in metadata
                             if value["generator"] == generator
                             and value["seed"] == seed
                             and value["budget"] == budget), None)
                if item is None:
                    plt.close(figure)
                    raise ValueError("missing exact outcomes for seed %d" % seed)
                learned = select(frame, "exact_n10_L%d" % budget, generator)
                learned = learned[pd.to_numeric(
                    learned["network_seed"], errors="coerce") == seed]
                handles += draw_exact(axes[row][column], pd.read_csv(item["path"]),
                                      learned, args.agent_seed)
                style_axis(axes[row][column], "%s, L=%d" %
                           (LABELS[generator], budget), row, column, exact=True)
        figure.suptitle(
            "Exact attainable outcomes: n=10, test seed %d, DQN seed %d" %
            (seed, args.agent_seed), fontsize=18, y=0.986)
        figure.subplots_adjust(left=0.08, right=0.89, bottom=0.13, top=0.94,
                               wspace=0.25, hspace=0.35)
        add_colourbar(figure, [0.92, 0.21, 0.014, 0.62])
        figure.legend(handles=unique_handles(handles), loc="lower center",
                      bbox_to_anchor=(0.49, 0.018), ncol=4, frameon=False,
                      fontsize=9.5, columnspacing=1.15, handletextpad=0.45)
        outputs.append(save_figure(
            figure, args, "exact", "exact_grid_seed%d_dqnseed%d" %
            (seed, args.agent_seed)))
    return outputs


def main():
    args = parse_args()
    raw = load_evaluations(args.evaluation_root)
    if args.target == "final-status":
        print("Evaluation rows: %d" % len(raw))
        if not raw.empty:
            print(raw.groupby("_family").size().to_string())
        return
    if raw.empty:
        raise ValueError("no completed evaluations found")
    outputs = [front_grid(args, raw, 10, [2, 3, 5]),
               front_grid(args, raw, 20, [2, 5, 10])]
    outputs += [cloud_grid(args, raw, family, n, budgets)
                for family in ["clouds", "dense_101"]
                for n, budgets in [(10, [2, 3, 5]), (20, [2, 5, 10])]]
    outputs += [size_grid(args, raw, tau, label) for tau, label in
                [("tau1", "1%"), ("tau2p5", "2.5%"), ("tau5", "5%")]]
    outputs += exact_grids(args, raw)
    print("Saved %d dissertation figures." % len(outputs))


if __name__ == "__main__":
    main()
