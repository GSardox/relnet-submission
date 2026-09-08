#!/usr/bin/env python

import argparse
import csv
import json
import math
import os
from copy import deepcopy

from relnet.evaluation.experiment_profiles import (
    BASELINES, PROFILES, checkpoint, load_profile, model_seeds,
)


RAW_COLUMNS = [
    "evaluation_name", "which", "objective", "n", "edge_percentage",
    "edge_budget", "initial_edges", "initial_density", "final_edges",
    "final_density", "weight", "network_generator", "checkpoint_generator",
    "network_seed", "graph_id", "algorithm", "model_label", "model_kind",
    "experiment_id", "agent_seed", "initial_efficiency", "final_efficiency",
    "delta_efficiency", "initial_robustness", "final_robustness",
    "delta_robustness", "combined_reward", "initial_connected",
    "final_connected", "initial_largest_component_size",
    "final_largest_component_size", "initial_robustness_canonical",
    "final_robustness_canonical", "initial_robustness_lcc",
    "final_robustness_lcc",
]
INHERITED = [
    "parent_dir", "output_dir", "weights", "num_graphs", "random_repeats",
    "baselines", "real_world_data_root", "normalization_n",
    "normalization_edge_percentage", "normalization_generator",
    "normalization_references", "normalization_manifest",
    "evaluation_batch_size",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--experiment", action="append", dest="experiments")
    parser.add_argument("--family", action="append", dest="families")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--baseline", action="append", dest="baselines",
                        choices=sorted(BASELINES))
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def experiments(profile):
    config = load_profile(profile)
    specs = []
    for experiment in config["experiments"]:
        base = {key: deepcopy(config[key]) for key in INHERITED if key in config}
        base.update(deepcopy(experiment))
        for n in base.pop("target_sizes", [None]):
            spec = deepcopy(base)
            if n is not None:
                spec["n"] = int(n)
                spec["name"] = spec["name"].format(n=int(n))
            spec["weights"] = [float(value) for value in spec.get(
                "weights", [value / 10.0 for value in range(11)])]
            spec["num_graphs"] = int(spec.get("num_graphs", 100))
            spec["random_repeats"] = int(spec.get("random_repeats", 50))
            spec["generator_params"] = spec.get("generator_params", {})
            spec["edge_budget"] = int(math.ceil(
                int(spec["n"]) * (int(spec["n"]) - 1) / 2.0 *
                float(spec["edge_percentage"]) / 100.0))
            if spec.get("normalization_manifest"):
                with open(spec["normalization_manifest"], "r") as handle:
                    values = json.load(handle)
                spec["normalization_references"] = values.get(
                    "normalization", values)
            specs.append(spec)
    return specs


def methods(spec, args):
    requested = set(args.models or [])
    models = [model for model in spec["models"]
              if not requested or model["label"] in requested]
    if requested and not models:
        raise ValueError("no requested model label exists in " + spec["name"])
    baselines = [] if args.no_baselines else list(
        args.baselines or spec.get("baselines", BASELINES))
    return models, baselines


def check_checkpoints(spec, models):
    paths, missing = [], []
    for model in models:
        weights = spec["weights"] if model["kind"] == "specialist" else [None]
        for generator in spec["generators"]:
            for weight in weights:
                for seed in model_seeds(model):
                    unused, unused, path = checkpoint(
                        spec, model, generator, seed, weight)
                    paths.append(path)
                    if not os.path.isfile(path) or not os.path.getsize(path):
                        missing.append(path)
    if missing:
        raise ValueError("missing model checkpoints:\n" + "\n".join(missing))
    return paths


def save_csv(path, rows):
    with open(path, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_manifest(path, profile, spec, models, baselines, checkpoint_count,
                  row_count):
    with open(path, "w") as handle:
        json.dump({
            "source_config": "profile:" + profile,
            "experiment": spec,
            "selected_models": [model["label"] for model in models],
            "selected_baselines": baselines,
            "checkpoint_count": checkpoint_count,
            "evaluation_rows": row_count,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    args = parse_args()
    wanted_names = set(args.experiments or [])
    wanted_families = set(args.families or [])
    selected = [spec for spec in experiments(args.profile)
                if (not wanted_names or spec["name"] in wanted_names)
                and (not wanted_families or
                     spec.get("family", "") in wanted_families)]
    if not selected:
        raise ValueError("no experiments selected")

    for spec in selected:
        models, baselines = methods(spec, args)
        paths = check_checkpoints(spec, models)
        print("Validated %d checkpoint paths for %s" %
              (len(paths), spec["name"]))
        if args.validate_only:
            continue

        from relnet.evaluation.model_evaluation import evaluate_experiment

        output_dir = os.path.join(spec["output_dir"], spec["name"])
        raw_path = os.path.join(output_dir, "evaluation_raw.csv")
        if args.reuse and os.path.isfile(raw_path):
            print("Reusing " + raw_path)
            continue
        os.makedirs(output_dir, exist_ok=True)
        rows = evaluate_experiment(spec, models, baselines)
        save_csv(raw_path, rows)
        save_manifest(os.path.join(output_dir, "evaluation_manifest.json"),
                      args.profile, spec, models, baselines, len(paths), len(rows))
        print("Saved %d rows to %s" % (len(rows), raw_path))
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
