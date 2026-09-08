#!/usr/bin/env python

from __future__ import print_function

import argparse
import csv
import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime

import numpy as np

from relnet.agent.baseline.baseline_agent import GreedyAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.eval_utils import get_values_for_g_list
from relnet.evaluation.experiment_conditions import get_exp_conditions, NORMALIZATION_FILE
from relnet.objective_functions.objective_functions import CriticalFractionTargeted, GlobalEfficiency
from relnet.state.network_generators import GNMNetworkGenerator, NetworkGenerator


FIELDS = [
    'network_generator', 'n', 'edge_budget', 'edge_percentage',
    'num_graphs', 'efficiency_reference_gain', 'robustness_reference_gain',
    'efficiency_initial', 'robustness_initial',
]


def measure(graphs, objective, edge_percentage, objective_kwargs):
    environment = GraphEdgeEnv(objective, objective_kwargs, edge_percentage)
    agent = GreedyAgent(environment)
    agent.setup({'random_seed': 42, 'log_progress': False}, {})
    initial, final = get_values_for_g_list(
        agent, [deepcopy(graph) for graph in graphs], None, True, {})
    return float(np.mean(final - initial)), float(np.mean(initial))


def atomic_json(path, value):
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    fd, temporary_path = tempfile.mkstemp(prefix=os.path.basename(path) + '.',
                                           suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def load_csv():
    if not os.path.exists(NORMALIZATION_FILE):
        return {}
    with open(NORMALIZATION_FILE, 'r') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError('normalization CSV schema mismatch')
        rows = {}
        for row in reader:
            key = (row['network_generator'], int(row['n']), int(row['edge_budget']))
            if key in rows:
                raise ValueError('duplicate normalization row for ' + str(key))
            rows[key] = row
        return rows


def save_csv(rows):
    values = sorted(rows.values(), key=lambda row: (
        int(row['n']), row['network_generator'], int(row['edge_budget'])))
    directory = os.path.dirname(NORMALIZATION_FILE)
    fd, temporary_path = tempfile.mkstemp(prefix='normalization.',
                                           suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'w') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, NORMALIZATION_FILE)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def calibrate_standard(args):
    rows = load_csv()
    for edge_percentage in args.edge_percentages:
        conditions = get_exp_conditions(args.which, edge_percentage, False, .5)
        conditions.set_graph_size(args.n)
        conditions.set_generator_seeds()
        edge_budget = NetworkGenerator.compute_number_edges(args.n, edge_percentage)
        seeds = conditions.train_seeds[:1 if args.which == 'empty' else args.num_graphs]
        objective_kwargs = {
            'random_seed': conditions.obj_fun_seed,
            'num_mc_sims': conditions.num_mc_sims,
        }

        for generator_class in conditions.network_generators:
            key = (generator_class.name, args.n, edge_budget)
            if key in rows and not args.replace_existing:
                print('Skipping existing normalization for ' + str(key))
                continue
            generator = generator_class(store_graphs=False)
            graphs = generator.generate_many(conditions.gen_params, seeds)
            efficiency_gain, efficiency_initial = measure(
                graphs, GlobalEfficiency(), edge_percentage, objective_kwargs)
            robustness_gain, robustness_initial = measure(
                graphs, CriticalFractionTargeted(), edge_percentage, objective_kwargs)
            if efficiency_gain <= 0. or robustness_gain <= 0.:
                raise ValueError('normalization gains must be positive for ' + str(key))
            rows[key] = {
                'network_generator': generator.name,
                'n': args.n,
                'edge_budget': edge_budget,
                'edge_percentage': edge_percentage,
                'num_graphs': len(seeds),
                'efficiency_reference_gain': efficiency_gain,
                'robustness_reference_gain': robustness_gain,
                'efficiency_initial': efficiency_initial,
                'robustness_initial': robustness_initial,
            }
            save_csv(rows)
            print('Saved complete normalization row for ' + str(key))


def calibrate_density(args):
    computed_budget = NetworkGenerator.compute_number_edges(args.n, args.edge_percentage)
    if computed_budget != args.edge_budget:
        raise ValueError('edge configuration gives L=%d, not L=%d' %
                         (computed_budget, args.edge_budget))
    if args.initial_edges < args.n - 1:
        raise ValueError('connected graphs require at least n-1 edges')
    if args.initial_edges + args.edge_budget > args.n * (args.n - 1) // 2:
        raise ValueError('initial graph does not leave enough non-edges')
    if os.path.exists(args.output):
        raise ValueError('refusing to overwrite ' + args.output)

    generator = GNMNetworkGenerator(store_graphs=False)
    params = {'n': args.n, 'm': args.initial_edges, 'uniform_tree': True}
    seeds = list(range(args.num_graphs))
    graphs = generator.generate_many(params, seeds)
    objective_kwargs = {'random_seed': args.random_seed,
                        'num_mc_sims': args.num_mc_sims}
    efficiency_gain, efficiency_initial = measure(
        graphs, GlobalEfficiency(), args.edge_percentage, objective_kwargs)
    robustness_gain, robustness_initial = measure(
        graphs, CriticalFractionTargeted(), args.edge_percentage, objective_kwargs)
    if efficiency_gain <= 0. or robustness_gain <= 0.:
        raise ValueError('normalization reference gains must be positive')

    possible_edges = args.n * (args.n - 1) // 2
    result = {
        'created_at': datetime.now().isoformat(),
        'calibration_method': 'greedy_endpoint_gain',
        'network_generator': GNMNetworkGenerator.name,
        'n': args.n,
        'initial_edges': args.initial_edges,
        'initial_density': float(args.initial_edges) / possible_edges,
        'edge_percentage': args.edge_percentage,
        'edge_budget': args.edge_budget,
        'final_edges': args.initial_edges + args.edge_budget,
        'num_graphs': args.num_graphs,
        'graph_seeds': seeds,
        'num_mc_sims': args.num_mc_sims,
        'random_seed': args.random_seed,
        'efficiency_reference_gain': efficiency_gain,
        'robustness_reference_gain': robustness_gain,
        'efficiency_initial': efficiency_initial,
        'robustness_initial': robustness_initial,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print('Saved ' + args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    standard = subparsers.add_parser('standard')
    standard.add_argument('--which', choices=['synth', 'empty'], required=True)
    standard.add_argument('--n', type=int, required=True)
    standard.add_argument('--edge_percentages', type=float, nargs='+', required=True)
    standard.add_argument('--num_graphs', type=int, default=100)
    standard.add_argument('--replace_existing', action='store_true')

    density = subparsers.add_parser('density')
    density.add_argument('--n', type=int, required=True)
    density.add_argument('--initial_edges', type=int, required=True)
    density.add_argument('--edge_percentage', type=float, required=True)
    density.add_argument('--edge_budget', type=int, required=True)
    density.add_argument('--num_graphs', type=int, default=100)
    density.add_argument('--num_mc_sims', type=int, default=40)
    density.add_argument('--random_seed', type=int, default=42)
    density.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.mode is None:
        parser.error('choose standard or density')
    return args


def main():
    args = parse_args()
    calibrate_standard(args) if args.mode == 'standard' else calibrate_density(args)


if __name__ == '__main__':
    main()
