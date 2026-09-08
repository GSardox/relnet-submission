import argparse
import json
import os
import re
import tempfile
from datetime import datetime

import torch

from relnet.agent.rnet_dqn.preference_agent import PreferenceConditionedRNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.experiment_conditions import get_exp_conditions
from relnet.evaluation.file_paths import FilePaths
from relnet.objective_functions.objective_functions import LinearCombinedObjective
from relnet.state.network_generators import (
    BANetworkGenerator,
    GNMNetworkGenerator,
    NetworkGenerator,
)


NETWORK_GENERATORS = {
    GNMNetworkGenerator.name: GNMNetworkGenerator,
    BANetworkGenerator.name: BANetworkGenerator,
}
ALLOWED_MODEL_SEEDS = (0, 42, 84, 126, 168)


def parse_weights(value, minimum=2):
    value = value.replace(':', ',')
    weights = [float(item) for item in value.split(',') if item.strip()]
    if len(weights) < minimum:
        raise ValueError("provide at least " + str(minimum) + " weights")
    if len(set(weights)) != len(weights):
        raise ValueError("weights must be unique")
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("weights must lie in [0, 1]")
    return weights


def get_hyperparams(experiment_conditions):
    grid = experiment_conditions.hyperparam_grids[
        LinearCombinedObjective.name
    ]['rnet_dqn']
    hyperparams = {}
    for name, values in grid.items():
        if len(values) != 1:
            raise ValueError("expected exactly one value for " + name)
        hyperparams[name] = values[0]
    return hyperparams


def validate_args(args):
    if args.n < 2:
        raise ValueError("n must be at least 2")
    if args.model_seed not in ALLOWED_MODEL_SEEDS:
        raise ValueError(
            "model_seed must be one of " +
            ', '.join(str(seed) for seed in ALLOWED_MODEL_SEEDS)
        )
    if args.agent_budget <= 0 or args.agent_budget > 400000:
        raise ValueError("agent_budget must lie in [1, 400000]")
    if args.validation_check_interval <= 0:
        raise ValueError("validation_check_interval must be positive")
    if re.match(r'^relnet_cpu_pref_[A-Za-z0-9_]+$', args.experiment_id) is None:
        raise ValueError("experiment_id must use the relnet_cpu_pref_ prefix")

    parent_basename = os.path.basename(os.path.abspath(args.parent_dir))
    if parent_basename not in (
            'preference_data', 'relnet_preference', 'density_data'):
        raise ValueError("parent_dir must be a dedicated preference data root")

    computed_budget = NetworkGenerator.compute_number_edges(
        args.n,
        args.edge_percentage,
    )
    if computed_budget != args.edge_budget:
        raise ValueError(
            "edge configuration gives L=" + str(computed_budget) +
            ", not L=" + str(args.edge_budget)
        )

    if args.network_generator not in (
            GNMNetworkGenerator.name, BANetworkGenerator.name):
        raise ValueError("synth requires random_network or barabasi_albert")

    density_mode = args.initial_edges is not None
    if density_mode:
        if args.which != 'synth':
            raise ValueError("initial_edges is supported only for synth")
        if args.network_generator != GNMNetworkGenerator.name:
            raise ValueError("initial_edges is supported only for random_network")
        if args.normalization_file is None:
            raise ValueError("initial_edges requires normalization_file")
        if args.initial_edges < args.n - 1:
            raise ValueError("connected graphs require at least n-1 initial edges")
        total_possible_edges = args.n * (args.n - 1) // 2
        if args.initial_edges + args.edge_budget > total_possible_edges:
            raise ValueError("initial graph does not leave enough non-edges")
    elif args.normalization_file is not None:
        raise ValueError("normalization_file requires initial_edges")


def load_density_normalization(path, args):
    with open(path, 'r') as handle:
        details = json.load(handle)

    expected = {
        'network_generator': GNMNetworkGenerator.name,
        'n': args.n,
        'initial_edges': args.initial_edges,
        'edge_budget': args.edge_budget,
    }
    for key, expected_value in expected.items():
        if details.get(key) != expected_value:
            raise ValueError(
                "normalization mismatch for " + key +
                ": expected " + str(expected_value) +
                ", found " + str(details.get(key))
            )

    required = (
        'efficiency_reference_gain',
        'robustness_reference_gain',
        'efficiency_initial',
        'robustness_initial',
    )
    for key in required:
        if key not in details:
            raise ValueError("normalization file is missing " + key)
    if details['efficiency_reference_gain'] <= 0.0:
        raise ValueError("efficiency reference gain must be positive")
    if details['robustness_reference_gain'] <= 0.0:
        raise ValueError("robustness reference gain must be positive")
    return details


def atomic_json(path, details):
    fd, temporary_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + '.',
        suffix='.tmp',
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(details, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', choices=['synth'], required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--edge_percentage', type=float, required=True)
    parser.add_argument('--edge_budget', type=int, required=True)
    parser.add_argument(
        '--network_generator',
        choices=sorted(NETWORK_GENERATORS),
        required=True,
    )
    parser.add_argument('--agent_budget', type=int, required=True)
    parser.add_argument('--model_seed', type=int, required=True)
    parser.add_argument('--experiment_id', required=True)
    parser.add_argument('--parent_dir', default='/preference_data')
    parser.add_argument(
        '--weights',
        default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
    )
    parser.add_argument('--validation_check_interval', type=int, default=5000)
    parser.add_argument('--initial_edges', type=int)
    parser.add_argument('--normalization_file')
    args = parser.parse_args()

    validate_args(args)
    training_weights = parse_weights(args.weights)
    density_mode = args.initial_edges is not None

    torch.set_num_threads(1)
    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(1)

    experiment_conditions = get_exp_conditions(
        args.which,
        args.edge_percentage,
        False,
        0.5,
    )
    experiment_conditions.set_graph_size(args.n)
    experiment_conditions.set_generator_seeds()

    normalization = None
    if density_mode:
        experiment_conditions.gen_params['m'] = args.initial_edges
        experiment_conditions.gen_params['uniform_tree'] = True
        normalization = load_density_normalization(
            args.normalization_file,
            args,
        )
        generator = GNMNetworkGenerator(store_graphs=False)
        objective = LinearCombinedObjective(
            weight=0.5,
            efficiency_reference_gain=normalization[
                'efficiency_reference_gain'
            ],
            robustness_reference_gain=normalization[
                'robustness_reference_gain'
            ],
            efficiency_initial=normalization['efficiency_initial'],
            robustness_initial=normalization['robustness_initial'],
        )
    else:
        generator = NETWORK_GENERATORS[args.network_generator](store_graphs=False)
        objective = experiment_conditions.create_objective_function(
            LinearCombinedObjective,
            generator.name,
        )

    objective_kwargs = {
        'random_seed': experiment_conditions.obj_fun_seed,
        'num_mc_sims': experiment_conditions.num_mc_sims,
    }
    environment = GraphEdgeEnv(
        objective,
        objective_kwargs,
        args.edge_percentage,
    )
    agent = PreferenceConditionedRNetDQNAgent(environment)
    file_paths = FilePaths(args.parent_dir, args.experiment_id)

    model_identifier_prefix = file_paths.construct_model_identifier_prefix(
        agent.algorithm_name,
        LinearCombinedObjective.name,
        generator.name,
        args.model_seed,
        0,
    )
    seed_log = os.path.join(
        str(file_paths.logs_dir),
        args.experiment_id + '_seed' + str(args.model_seed) + '.log',
    )
    run_options = {
        'random_seed': args.model_seed,
        'models_path': file_paths.models_dir,
        'log_progress': True,
        'log_filename': seed_log,
        'model_identifier_prefix': model_identifier_prefix,
        'restore_model': False,
        'log_tf_summaries': False,
        'training_weights': training_weights,
        'validation_check_interval': args.validation_check_interval,
        'max_validation_consecutive_steps': args.agent_budget,
    }
    agent.setup(run_options, get_hyperparams(experiment_conditions))

    train_graphs = generator.generate_many(
        experiment_conditions.gen_params,
        experiment_conditions.train_seeds,
    )
    validation_graphs = generator.generate_many(
        experiment_conditions.gen_params,
        experiment_conditions.validation_seeds,
    )

    if density_mode:
        if any(graph.num_edges != args.initial_edges for graph in train_graphs):
            raise ValueError("a training graph has the wrong initial edge count")
        if any(graph.num_edges != args.initial_edges for graph in validation_graphs):
            raise ValueError("a validation graph has the wrong initial edge count")

    print("Starting one preference-conditioned RelNet model")
    print("experiment_id=" + args.experiment_id)
    print("network_generator=" + generator.name)
    if density_mode:
        print("initial_edges=" + str(args.initial_edges))
    print("model_seed=" + str(args.model_seed))
    print("training_weights=" + ','.join(
        str(weight) for weight in training_weights
    ))
    print("agent_budget=" + str(args.agent_budget))
    print("validation_check_interval=" + str(
        args.validation_check_interval
    ))

    agent.train(train_graphs, validation_graphs, args.agent_budget)
    validation_rewards = agent.evaluate_preferences(
        validation_graphs,
        training_weights,
        validation=True,
    )
    completed_steps = int(agent.step)
    agent.finalize()
    if getattr(agent, 'hist_out', None) is not None and not agent.hist_out.closed:
        agent.hist_out.flush()
        agent.hist_out.close()

    manifest_filename = (
        'complete_preference_' + generator.name +
        '_seed' + str(args.model_seed) + '.json'
    )
    manifest_path = os.path.join(
        str(file_paths.experiment_dir),
        manifest_filename,
    )
    manifest = {
        'status': 'complete',
        'completed_at': datetime.now().isoformat(),
        'which': args.which,
        'n': args.n,
        'edge_percentage': args.edge_percentage,
        'edge_budget': args.edge_budget,
        'network_generator': generator.name,
        'model_seed': args.model_seed,
        'agent_budget': args.agent_budget,
        'completed_steps': completed_steps,
        'experiment_id': args.experiment_id,
        'model_identifier_prefix': model_identifier_prefix,
        'training_weights': training_weights,
        'validation_check_interval': args.validation_check_interval,
        'max_validation_consecutive_steps': args.agent_budget,
        'validation_rewards': {
            str(weight): float(reward)
            for weight, reward in validation_rewards.items()
        },
    }
    if density_mode:
        total_possible_edges = args.n * (args.n - 1) // 2
        manifest.update({
            'study': 'er_starting_density',
            'initial_edges': args.initial_edges,
            'initial_density': float(args.initial_edges) / total_possible_edges,
            'final_edges': args.initial_edges + args.edge_budget,
            'normalization_file': os.path.basename(args.normalization_file),
            'normalization': normalization,
        })
    atomic_json(manifest_path, manifest)

    print("Completed preference-conditioned training safely")
    print("validation_rewards=" + json.dumps(
        manifest['validation_rewards'],
        sort_keys=True,
    ))
    print("results=" + str(file_paths.experiment_dir))


if __name__ == '__main__':
    main()
