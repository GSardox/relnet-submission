#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


METHODS = [
    ("random", "Random"),
    ("lowest_degree_product", "LDP"),
    ("fiedler_vector", "FV"),
    ("effective_resistance", "ERes"),
    ("greedy", "Greedy"),
    ("specialist", "Specialist"),
    ("preference_conditioned", "Conditioned"),
]
METHOD_NAMES = [name for name, unused in METHODS]
FIXED_METHODS = METHOD_NAMES[:5]
STOCHASTIC = {"random", "specialist", "preference_conditioned"}
GENERATOR_LABELS = {
    "barabasi_albert": "BA",
    "random_network": "ER",
}
REAL_WORLD_METHODS = ["greedy", "specialist", "preference_conditioned"]
DATASET_LABELS = {
    "euroroad": "EuroRoad",
    "scigrid": "SciGRID",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation_dir",
                        default="/experiment_data/final_evaluations/standard")
    parser.add_argument("--paper_output_dir",
                        default="/experiment_data/final_tables")
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--real_world", action="store_true")
    parser.add_argument("--checkpoint_generator",
                        choices=sorted(GENERATOR_LABELS),
                        default="random_network")
    return parser.parse_args()


def method_rows(frame, method):
    if method in ("specialist", "preference_conditioned"):
        return frame[frame.model_kind == method]
    return frame[(frame.model_kind == "baseline") &
                 (frame.algorithm == method)]


def rows_at_weight(frame, weight):
    return frame[np.isclose(pd.to_numeric(frame.weight, errors="coerce"), weight)]


def summarize(frame, column, method, confidence):
    frame = frame.copy()
    frame["agent_seed"] = pd.to_numeric(
        frame.agent_seed, errors="coerce").fillna(0).astype(int)
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame.groupby(["agent_seed", "network_seed"])[column].mean()
    values = values.groupby(level="agent_seed").mean().dropna()
    interval = np.nan
    if method in STOCHASTIC and len(values) > 1:
        sem = values.std(ddof=1) / np.sqrt(len(values))
        interval = float(stats.t.ppf((1. + confidence) / 2.,
                                     len(values) - 1) * sem)
    return {"mean": float(values.mean()), "ci": interval,
            "seeds": len(values)}


def check_same_graphs(frames, label):
    graph_sets = [set(frame.network_seed.astype(int))
                  for frame in frames if not frame.empty]
    if not graph_sets or any(item != graph_sets[0] for item in graph_sets[1:]):
        raise ValueError("methods use different graphs for " + label)


def normalization_reference(frame, column, weight):
    endpoint = rows_at_weight(frame, weight).copy()
    endpoint[column] = pd.to_numeric(endpoint[column], errors="coerce")
    reward = pd.to_numeric(endpoint.combined_reward, errors="coerce")
    ratios = (endpoint[column] / reward).replace(
        [np.inf, -np.inf], np.nan).dropna()
    ratios = ratios[ratios > 0.]
    if ratios.empty:
        raise ValueError("cannot infer normalization for " + column)
    reference = float(ratios.median())
    if float(((ratios - reference).abs() / reference).quantile(.95)) > 1e-5:
        raise ValueError("inconsistent normalization for " + column)
    return reference


def pareto_front(points):
    points = np.asarray(points, dtype=float)
    points = np.unique(points[np.isfinite(points).all(axis=1)], axis=0)
    keep = []
    for index, point in enumerate(points):
        others = np.delete(points, index, axis=0)
        dominated = len(others) and np.any(
            np.all(others >= point, axis=1) & np.any(others > point, axis=1))
        if not dominated:
            keep.append(point)
    return np.asarray(keep, dtype=float).reshape((-1, 2))


def hypervolume(points):
    front = pareto_front(points)
    front = pareto_front(front[(front[:, 0] > 0.) & (front[:, 1] > 0.)])
    if not len(front):
        return 0.
    front = front[np.argsort(front[:, 0])]
    area, previous_x = 0., 0.
    for x_value, y_value in front:
        area += (x_value - previous_x) * y_value
        previous_x = x_value
    return float(area)


def mean_front(frame, agent_seed=None):
    if agent_seed is not None:
        seeds = pd.to_numeric(frame.agent_seed, errors="coerce")
        frame = frame[seeds == agent_seed]
    return frame.groupby("weight")[[
        "normalized_efficiency", "normalized_robustness"
    ]].mean().values


def collect_setting(name, frame, generator, confidence):
    frame = frame[frame.network_generator == generator].copy()
    if frame.empty:
        return None
    first = frame.iloc[0]
    result = {"experiment": name, "generator": generator,
              "n": int(first.n), "edge_budget": int(first.edge_budget)}
    by_method = {method: method_rows(frame, method) for method in METHOD_NAMES}
    check_same_graphs(list(by_method.values()), name + "/" + generator)

    result["endpoints"] = {}
    for objective, weight, column in [
            ("Efficiency", 1., "final_efficiency"),
            ("Robustness", 0., "final_robustness")]:
        values = {}
        endpoint = rows_at_weight(frame, weight)
        for method in METHOD_NAMES:
            rows = method_rows(endpoint, method)
            if not rows.empty:
                values[method] = summarize(rows, column, method, confidence)
        result["endpoints"][objective] = values

    frame["normalized_efficiency"] = (
        pd.to_numeric(frame.delta_efficiency, errors="coerce") /
        normalization_reference(frame, "delta_efficiency", 1.))
    frame["normalized_robustness"] = (
        pd.to_numeric(frame.delta_robustness, errors="coerce") /
        normalization_reference(frame, "delta_robustness", 0.))
    by_method = {method: method_rows(frame, method) for method in METHOD_NAMES}
    heuristic_points = []
    for method in FIXED_METHODS:
        if not by_method[method].empty:
            heuristic_points.extend(mean_front(by_method[method]).tolist())
    result["hypervolume"] = {
        "heuristic_union": {
            "mean": hypervolume(heuristic_points), "ci": np.nan, "seeds": 1
        }
    }
    for method in ("specialist", "preference_conditioned"):
        rows = by_method[method]
        seeds = sorted(pd.to_numeric(
            rows.agent_seed, errors="coerce").dropna().astype(int).unique())
        volumes = [hypervolume(mean_front(rows, seed)) for seed in seeds]
        interval = np.nan
        if len(volumes) > 1:
            sem = np.std(volumes, ddof=1) / np.sqrt(len(volumes))
            interval = float(stats.t.ppf((1. + confidence) / 2.,
                                         len(volumes) - 1) * sem)
        result["hypervolume"][method] = {
            "mean": float(np.mean(volumes)), "ci": interval,
            "seeds": len(volumes),
        }
    return result


def load_settings(args):
    selected, settings = set(args.experiment), []
    pattern = os.path.join(os.path.abspath(args.evaluation_dir),
                           "*", "evaluation_manifest.json")
    for manifest_path in sorted(glob.glob(pattern)):
        with open(manifest_path) as handle:
            name = json.load(handle)["experiment"]["name"]
        if selected and name not in selected:
            continue
        frame = pd.read_csv(os.path.join(
            os.path.dirname(manifest_path), "evaluation_raw.csv"),
            low_memory=False)
        for generator in sorted(frame.network_generator.dropna().unique()):
            setting = collect_setting(name, frame, generator, args.confidence)
            if setting:
                settings.append(setting)
    if not settings:
        raise ValueError("no matching completed evaluations found")
    return settings


def format_value(values, method, decimals, include_ci=True):
    if method not in values:
        return "--"
    item = values[method]
    value = ("%." + str(decimals) + "f") % item["mean"]
    if include_ci and np.isfinite(item["ci"]):
        value += r" $\pm$ " + (("%." + str(decimals) + "f") % item["ci"])
    best = max(candidate["mean"] for candidate in values.values())
    if round(item["mean"], decimals) == round(best, decimals):
        value = r"\textbf{" + value + "}"
    return value


def table_header(columns):
    return [r"\begin{table}[H]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{" + columns + "}", r"\hline"]


def settings_for_n(settings, n):
    return sorted([item for item in settings if item["n"] == n],
                  key=lambda item: (GENERATOR_LABELS.get(
                      item["generator"], item["generator"]),
                      item["edge_budget"]))


def endpoint_table(settings, n, decimals):
    lines = table_header("lllccccccc")
    lines += [
        r"Objective & $G$ & $L$ & Random & LDP & FV & ERes & Greedy & "
        r"\multicolumn{2}{c}{RNet--DQN} \\",
        r"\cline{9-10}", r" & & & & & & & & Specialist & Conditioned \\",
        r"\hline",
    ]
    for objective in ("Efficiency", "Robustness"):
        for index, setting in enumerate(settings_for_n(settings, n)):
            row = [objective if index == 0 else "",
                   GENERATOR_LABELS.get(setting["generator"],
                                        setting["generator"]),
                   str(setting["edge_budget"])]
            values = setting["endpoints"][objective]
            row += [format_value(values, method, decimals)
                    for method in METHOD_NAMES]
            lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\hline", r"\end{tabular}%", r"}",
        (r"\caption{Pure-objective endpoint performance for $N=%d$. "
         r"Efficiency uses $w=1$ and robustness uses $w=0$. Values for "
         r"Random and both RNet--DQN variants are means $\pm$ 95\%% "
         r"Student-$t$ confidence intervals across their independent seeds "
         r"(50 action seeds for Random and five training seeds for each "
         r"learned model); deterministic methods are means over 100 held-out "
         r"graphs. The highest displayed mean in each row is bold.}" % n),
        r"\label{tab:pure-objective-n%d}" % n, r"\end{table}", "",
    ]
    return lines


def hypervolume_table(settings, n, decimals):
    lines = table_header("llccc")
    lines += [
        r"$G$ & $L$ & Heuristic front & \multicolumn{2}{c}{RNet--DQN} \\",
        r"\cline{4-5}", r" & & & Specialist & Conditioned \\", r"\hline",
    ]
    methods = ["heuristic_union", "specialist", "preference_conditioned"]
    for setting in settings_for_n(settings, n):
        values = setting["hypervolume"]
        row = [GENERATOR_LABELS.get(setting["generator"], setting["generator"]),
               str(setting["edge_budget"])]
        row += [format_value(values, method, decimals) for method in methods]
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\hline", r"\end{tabular}%", r"}",
        (r"\caption{Two-objective hypervolume for $N=%d$, computed from "
         r"each method's nondominated mean front in normalized gain space "
         r"relative to $(0,0)$. The heuristic front is the nondominated "
         r"union of Random, LDP, FV, ERes and Greedy. RNet--DQN values are "
         r"means $\pm$ 95\%% Student-$t$ confidence intervals across five "
         r"training seeds.}" % n),
        r"\label{tab:hypervolume-n%d}" % n, r"\end{table}", "",
    ]
    return lines


def write_tables(settings, args):
    output_dir = os.path.join(os.path.abspath(args.paper_output_dir), "tables")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    lines = []
    for n in sorted(set(item["n"] for item in settings)):
        lines += endpoint_table(settings, n, args.decimals)
        lines += hypervolume_table(settings, n, args.decimals)
    output_path = os.path.join(output_dir, "synthetic_results_tables.tex")
    with open(output_path, "w") as handle:
        handle.write("\n".join(lines))

    audit = []
    for setting in settings:
        details = {key: setting[key] for key in
                   ("experiment", "generator", "n", "edge_budget")}
        for objective, values in setting["endpoints"].items():
            for method, value in values.items():
                audit.append(dict(details, table="endpoint",
                    objective=objective, method=method, mean=value["mean"],
                    ci=value["ci"], agent_seeds=value["seeds"]))
        for method, value in setting["hypervolume"].items():
            audit.append(dict(details, table="hypervolume", objective="front",
                method=method, mean=value["mean"], ci=value["ci"],
                agent_seeds=value["seeds"]))
    audit_path = os.path.join(output_dir, "synthetic_results_tables.csv")
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    print("Saved " + output_path)
    print("Saved " + audit_path)


def real_world_settings(args):
    selected, frames = set(args.experiment), []
    pattern = os.path.join(os.path.abspath(args.evaluation_dir),
                           "*", "evaluation_raw.csv")
    for csv_path in sorted(glob.glob(pattern)):
        frame = pd.read_csv(csv_path, low_memory=False)
        name = (str(frame.evaluation_name.iloc[0])
                if "evaluation_name" in frame and not frame.empty
                else os.path.basename(os.path.dirname(csv_path)))
        if selected and name not in selected:
            continue
        frame["experiment"] = name
        frames.append(frame)
    if not frames:
        raise ValueError("no matching real-world evaluations found")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "network_generator", "graph_id", "n", "initial_edges",
        "edge_budget", "weight", "network_seed", "agent_seed",
        "algorithm", "model_kind", "checkpoint_generator",
        "delta_efficiency", "delta_robustness",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("real-world CSV is missing: " + ", ".join(missing))
    frame = frame[frame.checkpoint_generator == args.checkpoint_generator].copy()
    for column in ["n", "initial_edges", "edge_budget", "weight",
                   "network_seed", "agent_seed", "delta_efficiency",
                   "delta_robustness"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame.edge_budget == 5].copy()
    graph_keys = ["network_generator", "graph_id", "n"]
    reference_rows = frame[
        frame.model_kind != "preference_conditioned"
    ]
    references = reference_rows.groupby(
        graph_keys, dropna=False
    ).initial_edges.first().rename("reference_initial_edges")
    frame = frame.join(references, on=graph_keys)
    frame["initial_edges"] = frame.reference_initial_edges.fillna(
        frame.initial_edges)
    keys = ["network_generator", "graph_id", "n", "initial_edges",
            "edge_budget"]
    settings = []
    for values, rows in frame.groupby(keys, dropna=False):
        item = dict(zip(keys, values))
        item["endpoints"] = {}
        for objective, weight, column in [
                ("Efficiency", 1., "delta_efficiency"),
                ("Robustness", 0., "delta_robustness")]:
            endpoint, summary = rows_at_weight(rows, weight), {}
            for method in REAL_WORLD_METHODS:
                method_frame = method_rows(endpoint, method)
                if not method_frame.empty:
                    summary[method] = summarize(
                        method_frame, column, method, args.confidence)
            item["endpoints"][objective] = summary
        settings.append(item)
    if not settings:
        raise ValueError("no rows use checkpoint generator " +
                         args.checkpoint_generator)
    return sorted(settings, key=lambda item: (
        item["network_generator"], item["n"], item["initial_edges"],
        item["graph_id"]))


def real_world_table(settings, objective, args):
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{0.88\textwidth}{!}{%",
        r"\begin{tabular}{llrrrccc}", r"\hline",
        r"Dataset & Network & $N$ & $m_0$ & $L$ & Greedy & "
        r"\multicolumn{2}{c}{RNet--DQN} \\",
        r"\cline{7-8}", r" & & & & & & Specialist & Conditioned \\",
        r"\hline",
    ]
    previous_dataset = None
    for setting in settings:
        dataset = setting["network_generator"]
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append(r"\hline")
        graph = str(setting["graph_id"])
        values = setting["endpoints"][objective]
        row = [
            (DATASET_LABELS.get(dataset, str(dataset))
             if dataset != previous_dataset else ""),
            graph.upper() if len(graph) == 2 else graph.replace("_", r"\_"),
            str(int(setting["n"])), str(int(setting["initial_edges"])), "5",
        ]
        row += [format_value(values, method, args.decimals)
                for method in REAL_WORLD_METHODS]
        lines.append(" & ".join(row) + r" \\")
        previous_dataset = dataset
    lower = objective.lower()
    weight = 1 if objective == "Efficiency" else 0
    source = GENERATOR_LABELS[args.checkpoint_generator]
    lines += [
        r"\hline", r"\end{tabular}%", r"}",
        (r"\caption{%s gain on the real networks at $w=%d$. The learned "
         r"models were trained on $N=20$ %s graphs and evaluated with $L=5$. "
         r"Learned values are means $\pm$ 95\%% Student-$t$ confidence "
         r"intervals across five training seeds; Greedy is deterministic. "
         r"The highest displayed mean for each network is bold.}" %
         (objective, weight, source)),
        r"\label{tab:real-world-%s-gain}" % lower,
        r"\end{table}", "",
    ]
    return lines


def write_real_world_tables(settings, args):
    output_dir = os.path.join(os.path.abspath(args.paper_output_dir), "tables")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    lines = []
    for objective in ("Efficiency", "Robustness"):
        lines += real_world_table(settings, objective, args)
    output_path = os.path.join(output_dir, "real_world_results_tables.tex")
    with open(output_path, "w") as handle:
        handle.write("\n".join(lines))
    print("Saved " + output_path)


def main():
    args = parse_args()
    if args.real_world:
        write_real_world_tables(real_world_settings(args), args)
    else:
        write_tables(load_settings(args), args)


if __name__ == "__main__":
    main()
