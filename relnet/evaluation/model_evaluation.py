import os
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np

from relnet.agent.baseline.baseline_agent import (
    EffectiveResistanceAgent, FiedlerVectorAgent, GreedyAgent,
    LowestDegreeProductAgent, RandomAgent,
)
from relnet.agent.rnet_dqn.preference_agent import (
    PreferenceConditionedRNetDQNAgent,
)
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.experiment_conditions import get_exp_conditions
from relnet.evaluation.experiment_profiles import checkpoint, model_seeds
from relnet.objective_functions.objective_functions import (
    CriticalFractionTargeted, GlobalEfficiency, LinearCombinedObjective,
)
from relnet.state.network_generators import (
    BANetworkGenerator, EuroroadNetworkGenerator, GNMNetworkGenerator,
    NetworkGenerator, ScigridNetworkGenerator,
)


GENERATORS = {
    "random_network": GNMNetworkGenerator,
    "barabasi_albert": BANetworkGenerator,
    "euroroad": EuroroadNetworkGenerator,
    "scigrid": ScigridNetworkGenerator,
}
BASELINES = {
    agent.algorithm_name: agent for agent in [
        RandomAgent, GreedyAgent, LowestDegreeProductAgent,
        FiedlerVectorAgent, EffectiveResistanceAgent,
    ]
}


def conditions_for(spec, weight):
    kind = "synth" if spec["which"] == "real_world" else spec["which"]
    conditions = get_exp_conditions(
        kind, float(spec["edge_percentage"]), False, float(weight)
    )
    conditions.set_graph_size(int(spec["n"]))
    conditions.update_size_dependant_params(1)
    conditions.set_generator_seeds()
    conditions.gen_params.update(spec.get("generator_params", {}))
    if spec.get("normalization_references"):
        conditions.explicit_normalization_references = deepcopy(
            spec["normalization_references"]
        )

    source_n = int(spec.get("normalization_n", spec["n"]))
    source_percentage = float(spec.get(
        "normalization_edge_percentage", spec["edge_percentage"]
    ))
    source_generator = spec.get("normalization_generator")
    if (source_n != int(spec["n"])
            or source_percentage != float(spec["edge_percentage"])
            or source_generator is not None):
        source = get_exp_conditions("synth", source_percentage, False, float(weight))
        source.set_graph_size(source_n)
        source.update_size_dependant_params(1)
        conditions.transfer_normalization_conditions = source
        conditions.transfer_normalization_generator = source_generator
    return conditions


def objective_parts(conditions, generator, weight):
    if hasattr(conditions, "explicit_normalization_references"):
        values = conditions.explicit_normalization_references
        values = values.get(generator, values.get("*"))
        if values is None:
            raise ValueError("no normalization references for " + str(generator))
        references = (
            values["efficiency_reference_gain"],
            values["robustness_reference_gain"],
            values["efficiency_initial"],
            values["robustness_initial"],
        )
    elif hasattr(conditions, "transfer_normalization_conditions"):
        source = conditions.transfer_normalization_generator or generator
        references = conditions.transfer_normalization_conditions.\
            get_normalization_references(source)
    else:
        references = conditions.get_normalization_references(generator)
    objective = LinearCombinedObjective(
        weight=float(weight), efficiency_reference_gain=references[0],
        robustness_reference_gain=references[1],
        efficiency_initial=references[2], robustness_initial=references[3],
    )
    kwargs = {
        "random_seed": conditions.obj_fun_seed,
        "num_mc_sims": conditions.num_mc_sims,
    }
    return objective, CriticalFractionTargeted, kwargs


def hyperparams(conditions):
    grid = conditions.hyperparam_grids[
        LinearCombinedObjective.name][RNetDQNAgent.algorithm_name]
    if any(len(values) != 1 for values in grid.values()):
        raise ValueError("evaluation expects one value per hyperparameter")
    return {name: values[0] for name, values in grid.items()}


def graph_diagnostics(graph, robustness):
    nx_graph = graph.to_networkx()
    components = list(nx.connected_components(nx_graph))
    connected = len(components) == 1 if components else False
    largest = max((len(component) for component in components), default=0)
    return connected, largest, float(robustness), (
        float(robustness) if connected else np.nan)


def measure(graphs, robustness_class, kwargs):
    measured = []
    for graph in graphs:
        efficiency = float(GlobalEfficiency.compute(graph, **kwargs))
        robustness = float(robustness_class.compute(graph, **kwargs))
        connected, largest, canonical, lcc = graph_diagnostics(graph, robustness)
        measured.append({
            "efficiency": efficiency, "robustness": robustness,
            "connected": connected, "largest": largest,
            "canonical": canonical, "lcc": lcc,
        })
    return measured


def setting_fields(spec, generator, weight, seed, graph, graph_id=""):
    n = int(graph.num_nodes)
    possible_edges = n * (n - 1) / 2.0
    initial_edges = int(graph.num_edges)
    budget = NetworkGenerator.compute_number_edges(
        n, float(spec["edge_percentage"])
    )
    return {
        "evaluation_name": spec["name"],
        "which": spec["which"],
        "objective": spec.get("objective", "critical"),
        "n": n,
        "edge_percentage": float(spec["edge_percentage"]),
        "edge_budget": budget,
        "initial_edges": initial_edges,
        "initial_density": initial_edges / possible_edges,
        "final_edges": initial_edges + budget,
        "final_density": (initial_edges + budget) / possible_edges,
        "weight": float(weight),
        "network_generator": generator,
        "checkpoint_generator": spec.get("normalization_generator", generator),
        "network_seed": int(seed),
        "graph_id": graph_id,
    }


def episode_rows(spec, generator, weight, seeds, graph_ids,
                 initial_graphs, final_graphs, initial_values, final_values,
                 robustness_class, kwargs, method):
    before = measure(initial_graphs, robustness_class, kwargs)
    after = measure(final_graphs, robustness_class, kwargs)
    rows = []
    for index, seed in enumerate(seeds):
        initial, final = before[index], after[index]
        row = setting_fields(
            spec, generator, weight, seed, initial_graphs[index], graph_ids[index]
        )
        row.update(method)
        row.update({
            "initial_efficiency": initial["efficiency"],
            "final_efficiency": final["efficiency"],
            "delta_efficiency": final["efficiency"] - initial["efficiency"],
            "initial_robustness": initial["robustness"],
            "final_robustness": final["robustness"],
            "delta_robustness": final["robustness"] - initial["robustness"],
            "combined_reward": float(final_values[index] - initial_values[index]),
            "initial_connected": initial["connected"],
            "final_connected": final["connected"],
            "initial_largest_component_size": initial["largest"],
            "final_largest_component_size": final["largest"],
            "initial_robustness_canonical": initial["canonical"],
            "final_robustness_canonical": final["canonical"],
            "initial_robustness_lcc": initial["lcc"],
            "final_robustness_lcc": final["lcc"],
            "final_edges": int(final_graphs[index].num_edges),
            "final_density": (
                float(final_graphs[index].num_edges)
                / (final_graphs[index].num_nodes
                   * (final_graphs[index].num_nodes - 1) / 2.0)
            ),
        })
        rows.append(row)
    return rows

def _agent(agent_class, objective, kwargs, edge_percentage, seed,
           setup_options, conditions, cache, cache_key):
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    agent = agent_class(GraphEdgeEnv(objective, kwargs, edge_percentage))
    options = {"random_seed": int(seed), "log_progress": False}
    options.update(setup_options)
    agent.setup(options, hyperparams(conditions) if agent_class in (
        RNetDQNAgent, PreferenceConditionedRNetDQNAgent
    ) else {})
    if cache is not None:
        cache[cache_key] = agent
    return agent


def evaluate_agent(agent_class, spec, generator, graphs, seeds, graph_ids,
                   weight, seed, label, kind, cache=None, cache_key=None,
                   model=None):
    conditions = conditions_for(spec, weight)
    objective, robustness_class, kwargs = objective_parts(conditions, generator, weight)
    experiment_id, prefix = "", None
    options = {}
    if agent_class is RNetDQNAgent:
        experiment_id, prefix, _ = checkpoint(
            spec, model, generator, seed, weight
        )
        options = {
            "restore_model": True,
            "model_identifier_prefix": prefix,
            "models_path": os.path.join(
                model.get("parent_dir", spec["parent_dir"]),
                experiment_id, "models",
            ),
            "log_tf_summaries": False,
        }
    agent = _agent(
        agent_class, objective, kwargs, float(spec["edge_percentage"]), seed,
        options, conditions, cache, cache_key,
    )
    agent.environment.edge_budget_percentage = float(spec["edge_percentage"])
    agent.set_random_seeds(int(seed))
    initial_values = agent.environment.get_objective_function_values(graphs)
    agent.eval(graphs, initial_obj_values=initial_values, validation=True)
    final_graphs = agent.environment.g_list
    final_values = np.asarray(agent.environment.get_final_values())
    final_kwargs = deepcopy(agent.environment.objective_function_kwargs)
    rows = episode_rows(
        spec, generator, weight, seeds, graph_ids, graphs, final_graphs,
        initial_values, final_values, robustness_class, final_kwargs, {
            "algorithm": agent_class.algorithm_name,
            "model_label": label,
            "model_kind": kind,
            "experiment_id": experiment_id,
            "agent_seed": int(seed),
        },
    )
    if cache is None:
        agent.finalize()
    return rows


def evaluate_conditioned(spec, model, generator, graphs, seeds, graph_ids,
                         seed, cache=None):
    conditions = conditions_for(spec, 0.5)
    objective, robustness_class, kwargs = objective_parts(conditions, generator, 0.5)
    experiment_id, prefix, _ = checkpoint(spec, model, generator, seed)
    cache_key = (
        "conditioned", model["label"], generator,
        model.get("checkpoint_generator", generator), int(seed),
        experiment_id, prefix,
    )
    training_weights = model.get("training_weights", spec["weights"])
    if len(training_weights) < 2:
        training_weights = [0.0, 1.0]
    agent = _agent(
        PreferenceConditionedRNetDQNAgent, objective, kwargs,
        float(spec["edge_percentage"]), seed, {
            "models_path": os.path.join(
                model.get("parent_dir", spec["parent_dir"]),
                experiment_id, "models",
            ),
            "model_identifier_prefix": prefix,
            "restore_model": True,
            "log_tf_summaries": False,
            "training_weights": [float(value) for value in training_weights],
        }, conditions, cache, cache_key,
    )
    agent.environment.edge_budget_percentage = float(spec["edge_percentage"])
    agent.set_random_seeds(int(seed))
    components = agent.compute_components(graphs)
    rows = []
    for weight in spec["weights"]:
        initial_values = agent.combine_components(components, weight)
        agent.eval_at_weight(
            graphs, weight, components=components, validation=True
        )
        rows.extend(episode_rows(
            spec, generator, weight, seeds, graph_ids, graphs,
            agent.environment.g_list, initial_values,
            np.asarray(agent.environment.get_final_values()),
            robustness_class,
            deepcopy(agent.environment.objective_function_kwargs), {
                "algorithm": model.get("algorithm", "rnet_dqn_pref"),
                "model_label": model["label"],
                "model_kind": model["kind"],
                "experiment_id": experiment_id,
                "agent_seed": int(seed),
            },
        ))
    if cache is None:
        agent.finalize()
    return rows


def rescore(rows, spec, generator, weight):
    objective, _, _ = objective_parts(conditions_for(spec, weight), generator, weight)
    rescored = []
    for source in rows:
        row = dict(source)
        row["weight"] = float(weight)
        row["combined_reward"] = (
            float(weight) * row["delta_efficiency"] /
            objective.efficiency_reference_gain +
            (1.0 - float(weight)) * row["delta_robustness"] /
            objective.robustness_reference_gain)
        rescored.append(row)
    return rescored


def real_world_spec(spec, graph, graph_id):
    item = deepcopy(spec)
    item["n"] = int(graph.num_nodes)
    budget = spec.get("fixed_edge_budget")
    if budget is None:
        item["edge_budget"] = NetworkGenerator.compute_number_edges(
            item["n"], float(spec["edge_percentage"]))
        return item
    budget = int(budget)
    possible = item["n"] * (item["n"] - 1) / 2.0
    available = possible - graph.num_edges
    if budget <= 0 or budget > available:
        raise ValueError(
            "fixed_edge_budget=%d invalid for %s (n=%d, available=%d)" %
            (budget, graph_id, item["n"], int(available)))
    item["edge_budget"] = budget
    item["edge_percentage"] = 100.0 * (budget - 0.5) / possible
    return item


def generate_graphs(spec, conditions, generator_name):
    if spec["which"] == "real_world":
        generator = GENERATORS[generator_name](
            store_graphs=False,
            original_dataset_dir=Path(spec["real_world_data_root"]))
        seeds = list(range(generator.get_num_graphs()))
    else:
        generator = GENERATORS[generator_name](store_graphs=False)
        seeds = list(conditions.test_seeds)
        while len(seeds) < spec["num_graphs"]:
            seeds.append(seeds[-1] + 1)
        seeds = seeds[:spec["num_graphs"]]
    graphs = generator.generate_many(conditions.gen_params, seeds)
    graph_ids = [
        generator.get_graph_name(seed) if spec["which"] == "real_world" else ""
        for seed in seeds
    ]
    return graphs, seeds, graph_ids


def evaluate_batch(spec, models, baselines, generator, graphs, seeds,
                   graph_ids, cache):
    rows = []
    static = {}
    for name in [item for item in baselines if item != "greedy"]:
        results = []
        repeats = spec["random_repeats"] if name == "random" else 1
        for repeat in range(repeats):
            results += evaluate_agent(
                BASELINES[name], spec, generator, graphs, seeds, graph_ids,
                0.5, repeat * 42, name, "baseline", cache,
                ("baseline", name, generator, repeat))
        static[name] = results

    specialists = [model for model in models if model["kind"] == "specialist"]
    for weight in spec["weights"]:
        for results in static.values():
            rows += rescore(results, spec, generator, weight)
        if "greedy" in baselines:
            rows += evaluate_agent(
                GreedyAgent, spec, generator, graphs, seeds, graph_ids,
                weight, 0, "greedy", "baseline", cache,
                ("greedy", generator, float(weight)))
        for model in specialists:
            checkpoint_generator = model.get("checkpoint_generator", generator)
            for seed in model_seeds(model):
                experiment_id, prefix, unused = checkpoint(
                    spec, model, generator, seed, weight)
                rows += evaluate_agent(
                    RNetDQNAgent, spec, generator, graphs, seeds, graph_ids,
                    weight, seed, model["label"], model["kind"], cache,
                    ("specialist", model["label"], generator,
                     checkpoint_generator, float(weight), int(seed),
                     experiment_id, prefix), model)

    conditioned = [model for model in models
                   if model["kind"] == "preference_conditioned"]
    for model in conditioned:
        for seed in model_seeds(model):
            rows += evaluate_conditioned(
                spec, model, generator, graphs, seeds, graph_ids, seed, cache)
    return rows


def evaluate_experiment(spec, models, baselines):
    rows = []
    conditions = conditions_for(spec, 0.5)
    for generator in spec["generators"]:
        graphs, seeds, graph_ids = generate_graphs(spec, conditions, generator)
        print("[%s] generator=%s graphs=%d" %
              (spec["name"], generator, len(seeds)))
        batch_size = (1 if spec["which"] == "real_world" else
                      int(spec.get("evaluation_batch_size", len(graphs))))
        cache = {}
        for start in range(0, len(graphs), batch_size):
            stop = min(start + batch_size, len(graphs))
            if (batch_size == 1 and
                    (start == 0 or stop % 25 == 0 or stop == len(graphs))):
                print("  graph progress=%d/%d" % (stop, len(graphs)))
            batch_spec = (real_world_spec(spec, graphs[start], graph_ids[start])
                          if spec["which"] == "real_world" else spec)
            rows += evaluate_batch(
                batch_spec, models, baselines, generator,
                graphs[start:stop], seeds[start:stop], graph_ids[start:stop],
                cache)
        for agent in cache.values():
            agent.finalize()
    return rows
