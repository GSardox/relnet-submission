import argparse
import json
import os
import tempfile
from datetime import datetime

import torch

from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.experiment_conditions import get_exp_conditions
from relnet.evaluation.file_paths import FilePaths
from relnet.objective_functions.objective_functions import LinearCombinedObjective
from relnet.state.network_generators import NetworkGenerator, GNMNetworkGenerator, \
    BANetworkGenerator


NETWORK_GENERATORS = {
    GNMNetworkGenerator.name: GNMNetworkGenerator,
    BANetworkGenerator.name: BANetworkGenerator,
}


def get_hyperparams(experiment_conditions):
    grid = experiment_conditions.hyperparam_grids[LinearCombinedObjective.name][RNetDQNAgent.algorithm_name]
    hyperparams = {}

    for name, values in grid.items():
        if len(values) != 1:
            raise ValueError(
                "single-task runner requires exactly one value for hyperparameter " + name
            )
        hyperparams[name] = values[0]

    return hyperparams


def validate_args(args):
    if args.n < 2:
        raise ValueError("n must be at least 2")

    if args.weight < 0. or args.weight > 1.:
        raise ValueError("weight must lie in [0, 1]")

    if args.agent_budget <= 0 or args.agent_budget > 200000:
        raise ValueError("agent_budget must lie in [1, 200000]")

    if args.model_seed < 0 or args.model_seed % 42 != 0:
        raise ValueError(
            "model_seed must be a non-negative multiple of 42 "
            "to follow the original repository convention"
        )

    computed_edge_budget = NetworkGenerator.compute_number_edges(args.n, args.edge_percentage)
    if computed_edge_budget != args.edge_budget:
        raise ValueError(
            "edge configuration mismatch: n=" + str(args.n) +
            ", edge_percentage=" + str(args.edge_percentage) +
            " gives L=" + str(computed_edge_budget) +
            ", not L=" + str(args.edge_budget)
        )

    if args.network_generator not in NETWORK_GENERATORS:
        raise ValueError("unknown network generator " + args.network_generator)

    if args.network_generator not in (
            GNMNetworkGenerator.name, BANetworkGenerator.name):
        raise ValueError("synth jobs require random_network or barabasi_albert")
    expected_budget = 20000 * args.edge_budget
    if args.agent_budget != expected_budget:
        raise ValueError(
            "synthetic L=" + str(args.edge_budget) +
            " must use " + str(expected_budget) + " training steps"
        )


def save_manifest(file_paths, args, average_reward, model_identifier_prefix):
    manifest = {
        "status": "complete",
        "completed_at": datetime.now().isoformat(),
        "which": args.which,
        "n": args.n,
        "edge_percentage": args.edge_percentage,
        "edge_budget": args.edge_budget,
        "weight": args.weight,
        "objective": "critical",
        "network_generator": args.network_generator,
        "normalization_generator": args.network_generator,
        "model_seed": args.model_seed,
        "agent_budget": args.agent_budget,
        "experiment_id": args.experiment_id,
        "model_identifier_prefix": model_identifier_prefix,
        "validation_reward": float(average_reward),
    }

    if args.model_seed == 0:
        manifest_filename = "complete_" + args.network_generator + ".json"
    else:
        manifest_filename = (
            "complete_" + args.network_generator +
            "_seed" + str(args.model_seed) + ".json"
        )
    manifest_path = file_paths.experiment_dir / manifest_filename

    fd, temporary_path = tempfile.mkstemp(
        prefix=manifest_path.name + ".",
        suffix=".tmp",
        dir=str(manifest_path.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, str(manifest_path))
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["synth"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--edge_percentage", type=float, required=True)
    parser.add_argument("--edge_budget", type=int, required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument(
        "--network_generator",
        choices=sorted(NETWORK_GENERATORS.keys()),
        required=True,
    )
    parser.add_argument("--agent_budget", type=int, required=True)
    parser.add_argument(
        "--model_seed",
        type=int,
        default=0,
        help="independent training seed; original runs use run_number * 42",
    )
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--parent_dir", default="/experiment_data")
    args = parser.parse_args()

    validate_args(args)
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    experiment_conditions = get_exp_conditions(
        args.which,
        args.edge_percentage,
        False,
        args.weight,
    )
    experiment_conditions.set_graph_size(args.n)
    experiment_conditions.set_generator_seeds()

    model_seed = args.model_seed

    generator_class = NETWORK_GENERATORS[args.network_generator]
    generator = generator_class(store_graphs=False)

    efficiency_reference_gain, robustness_reference_gain, efficiency_initial, \
        robustness_initial = experiment_conditions.get_normalization_references(
            generator.name
        )

    file_paths = FilePaths(args.parent_dir, args.experiment_id)
    objective_kwargs = {
        "random_seed": experiment_conditions.obj_fun_seed,
        "num_mc_sims": experiment_conditions.num_mc_sims,
    }
    objective = LinearCombinedObjective(
        weight=args.weight,
        efficiency_reference_gain=efficiency_reference_gain,
        robustness_reference_gain=robustness_reference_gain,
        efficiency_initial=efficiency_initial,
        robustness_initial=robustness_initial,
    )
    environment = GraphEdgeEnv(
        objective,
        objective_kwargs,
        args.edge_percentage,
    )
    agent = RNetDQNAgent(environment)

    model_identifier_prefix = file_paths.construct_model_identifier_prefix(
        RNetDQNAgent.algorithm_name,
        LinearCombinedObjective.name,
        generator.name,
        model_seed,
        0,
    )

    run_options = {
        "random_seed": model_seed,
        "models_path": file_paths.models_dir,
        "log_progress": True,
        "log_filename": str(file_paths.construct_log_filepath()),
        "model_identifier_prefix": model_identifier_prefix,
        "restore_model": False,
        "log_tf_summaries": False,
    }

    hyperparams = get_hyperparams(experiment_conditions)
    agent.setup(run_options, hyperparams)

    print("experiment_id=" + args.experiment_id)
    print("network_generator=" + generator.name)
    print("weight=" + str(args.weight))
    print("edge_budget=" + str(args.edge_budget))
    print("agent_budget=" + str(args.agent_budget))
    print("model_seed=" + str(model_seed))

    train_graphs = generator.generate_many(
        experiment_conditions.gen_params,
        experiment_conditions.train_seeds,
    )
    validation_graphs = generator.generate_many(
        experiment_conditions.gen_params,
        experiment_conditions.validation_seeds,
    )

    agent.train(
        train_graphs,
        validation_graphs,
        args.agent_budget,
    )
    average_reward = agent.eval(validation_graphs)

    hyperopt_result_file = (
        file_paths.hyperopt_results_dir /
        file_paths.construct_best_validation_file_name(model_identifier_prefix)
    )
    with open(str(hyperopt_result_file), "w") as f:
        f.write("%.6f\n" % average_reward)

    agent.finalize()
    if getattr(agent, "hist_out", None) is not None and not agent.hist_out.closed:
        agent.hist_out.flush()
        agent.hist_out.close()
    save_manifest(
        file_paths,
        args,
        average_reward,
        model_identifier_prefix,
    )

    print("Completed safely")
    print("validation_reward=" + str(float(average_reward)))
    print("results=" + str(file_paths.experiment_dir))


if __name__ == "__main__":
    main()
