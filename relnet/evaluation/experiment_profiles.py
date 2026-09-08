import os


AGENT_SEEDS = [0, 42, 84, 126, 168]
WEIGHTS = [value / 10.0 for value in range(11)]
DENSE_WEIGHTS = [value / 100.0 for value in range(101)]
BASELINES = [
    "random", "greedy", "lowest_degree_product",
    "fiedler_vector", "effective_resistance",
]
SETTINGS = [
    (10, "L2", 2.5), (10, "L3", 5.0), (10, "L5", 10.0),
    (20, "L2", 1.0), (20, "L5", 2.5), (20, "L10", 5.0),
]
PROFILES = ("final", "real_world_density", "real_world_fixed")


def specialist(n, budget, label="Separately trained RNet-DQN", generator=None):
    model = {
        "kind": "specialist",
        "label": label,
        "experiment_id_template":
            "relnet_cpu_v4_n%d_%s_w{weight_label}_s1exact" % (n, budget),
        "agent_seeds": list(AGENT_SEEDS),
    }
    if generator:
        model["checkpoint_generator"] = generator
    return model


def conditioned(n, budget, label="Preference-conditioned RNet-DQN",
                generator=None, density=None):
    if density is None:
        parent = "/experiment_data/relnet_preference_s2v"
        experiment = "relnet_cpu_pref_s2v_full400k_n%d_%s_s1exact" % (n, budget)
    else:
        parent = "/experiment_data/relnet_preference_density"
        experiment = "relnet_cpu_pref_density_v1_n20_m%03d_L5_s5" % density
    model = {
        "kind": "preference_conditioned",
        "label": label,
        "parent_dir": parent,
        "experiment_id": experiment,
        "agent_seeds": list(AGENT_SEEDS),
        "training_weights": list(WEIGHTS),
    }
    if generator:
        model["checkpoint_generator"] = generator
    return model


def synthetic(name, n, percentage, family, models, graphs=100,
              repeats=50, baselines=BASELINES, weights=WEIGHTS):
    return {
        "name": name,
        "which": "synth",
        "n": n,
        "edge_percentage": percentage,
        "generators": ["barabasi_albert", "random_network"],
        "models": models,
        "family": family,
        "output_dir": "/experiment_data/final_evaluations/" + family,
        "num_graphs": graphs,
        "random_repeats": repeats,
        "evaluation_batch_size": 100,
        "baselines": list(baselines),
        "weights": list(weights),
    }


def final_profile():
    experiments = []
    for n, budget, percentage in SETTINGS:
        experiments.append(synthetic(
            "n%d_%s" % (n, budget), n, percentage, "standard",
            [specialist(n, budget), conditioned(n, budget)]))
    for n, budget, percentage in SETTINGS:
        experiments.append(synthetic(
            "cloud_n%d_%s" % (n, budget), n, percentage, "clouds",
            [specialist(n, budget)], graphs=500, repeats=1, baselines=[]))
    for n, budget, percentage in SETTINGS:
        experiments.append(synthetic(
            "dense_n%d_%s" % (n, budget), n, percentage, "dense_101",
            [conditioned(n, budget)], repeats=1, baselines=[],
            weights=DENSE_WEIGHTS))
    for n, budget, percentage in SETTINGS[:3]:
        experiment = synthetic(
            "exact_n%d_%s" % (n, budget), n, percentage, "exact",
            [specialist(n, budget)], graphs=5)
        experiment["evaluation_batch_size"] = 1
        experiments.append(experiment)
    for tau, budget, percentage in [
            ("tau1", "L2", 1.0), ("tau2p5", "L5", 2.5),
            ("tau5", "L10", 5.0)]:
        model = conditioned(20, budget)
        model.pop("training_weights")
        experiments.append({
            "name": "size_transfer_n{n}_" + tau,
            "target_sizes": [20, 30, 40],
            "which": "synth",
            "edge_percentage": percentage,
            "normalization_edge_percentage": percentage,
            "normalization_n": 20,
            "generators": ["barabasi_albert", "random_network"],
            "models": [model],
            "family": "size_generalization",
            "output_dir": "/experiment_data/final_evaluations/size_generalization",
            "num_graphs": 100,
            "random_repeats": 50,
            "evaluation_batch_size": 100,
            "baselines": ["random", "lowest_degree_product",
                          "fiedler_vector", "effective_resistance"],
            "weights": list(WEIGHTS),
        })
    for edges in [19, 29, 38, 48, 57, 67]:
        model = conditioned(20, "L5", density=edges)
        density = edges / 190.0
        experiments.append({
            "name": "density_m%03d_L5" % edges,
            "which": "synth",
            "n": 20,
            "edge_percentage": 2.5,
            "initial_edges": edges,
            "initial_density": density,
            "generators": ["random_network"],
            "generator_params": {"m": edges, "m_percentage_er": 100.0 * density},
            "normalization_manifest": (
                model["parent_dir"] + "/" + model["experiment_id"] +
                "/complete_preference_random_network_seed0.json"),
            "models": [model],
            "family": "starting_density",
            "output_dir": "/experiment_data/final_evaluations/starting_density",
            "num_graphs": 100,
            "random_repeats": 1,
            "evaluation_batch_size": 100,
            "baselines": [],
            "weights": list(DENSE_WEIGHTS),
        })
    return {"parent_dir": "/experiment_data", "experiments": experiments}


def real_world(name, models, generator, budget, family="real_world",
               output="/experiment_data/final_evaluations_fixed_real_world/real_world",
               baselines=("greedy",)):
    percentage = {2: 1.0, 5: 2.5, 10: 5.0}[budget]
    return {
        "name": name,
        "which": "real_world",
        "n": 20,
        "edge_percentage": percentage,
        "normalization_edge_percentage": percentage,
        "normalization_generator": generator,
        "normalization_n": 20,
        "generators": ["euroroad", "scigrid"],
        "models": models,
        "family": family,
        "output_dir": output,
        "real_world_data_root": "/experiment_data/real_world_graphs_n10_50/processed_data",
        "num_graphs": 100,
        "random_repeats": 1,
        "evaluation_batch_size": 1,
        "baselines": list(baselines),
        "weights": list(WEIGHTS),
        "fixed_edge_budget": budget,
    }


def real_world_fixed_profile():
    experiments = []
    for short, generator in [("er", "random_network"),
                             ("ba", "barabasi_albert")]:
        for budget in [2, 5, 10]:
            conditioned_model = conditioned(
                20, "L%d" % budget,
                "%s-trained conditioned RNet-DQN" % short.upper(), generator)
            conditioned_model.pop("training_weights")
            models = [
                specialist(20, "L%d" % budget,
                           "%s-trained specialist RNet-DQN" % short.upper(),
                           generator),
                conditioned_model,
            ]
            experiments.append(real_world(
                "real_world_from_%s_L%d" % (short, budget),
                models, generator, budget))
    return {"parent_dir": "/experiment_data", "experiments": experiments}


def real_world_density_profile():
    standard = conditioned(20, "L5", "Standard conditioned", "random_network")
    standard.pop("training_weights")
    models = [standard]
    names = ["rw_standard_conditioned_L5"]
    manifests = [None]
    for edges in [19, 29, 38, 48, 57, 67]:
        model = conditioned(
            20, "L5", "Density m=%d conditioned" % edges,
            "random_network", edges)
        models.append(model)
        names.append("rw_density_m%03d_conditioned_L5" % edges)
        manifests.append(model["parent_dir"] + "/" + model["experiment_id"] +
                         "/complete_preference_random_network_seed0.json")
    experiments = []
    for name, model, manifest in zip(names, models, manifests):
        experiment = real_world(
            name, [model], "random_network", 5, "real_world_density",
            "/experiment_data/real_world_density_probe/evaluations", ())
        if manifest:
            experiment["normalization_manifest"] = manifest
        experiments.append(experiment)
    return {"parent_dir": "/experiment_data", "experiments": experiments}


def load_profile(name):
    profiles = {
        "final": final_profile,
        "real_world_density": real_world_density_profile,
        "real_world_fixed": real_world_fixed_profile,
    }
    try:
        return profiles[name]()
    except KeyError:
        raise ValueError("unknown evaluation profile: " + str(name))


def model_seeds(model):
    return [int(seed) for seed in model.get(
        "agent_seeds", [model.get("agent_seed", 0)])]


def checkpoint(spec, model, generator, seed, weight=None):
    checkpoint_generator = model.get("checkpoint_generator", generator)
    algorithm = ("rnet_dqn_pref" if model["kind"] == "preference_conditioned"
                 else "rnet_dqn")
    if model["kind"] == "specialist":
        experiment = model["experiment_id_template"].format(
            weight_label="%02d" % int(round(float(weight) * 10)))
    else:
        experiment = model["experiment_id"]
    prefix = "%s-combined_linear-%s-%d-0" % (
        algorithm, checkpoint_generator, int(seed))
    path = os.path.join(
        model.get("parent_dir", spec["parent_dir"]), experiment,
        "models", "checkpoints", prefix, algorithm + "_agent.model")
    return experiment, prefix, path
