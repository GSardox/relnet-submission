import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", default="/experiment_data")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--edge_budget", type=int, required=True)
    args = parser.parse_args()
    failures = []
    completed = 0
    for weight_index in range(11):
        experiment = "relnet_cpu_v4_n%d_L%d_w%02d_s1exact" % (
            args.n, args.edge_budget, weight_index)
        for generator in ["random_network", "barabasi_albert"]:
            for model_seed in [0, 42, 84, 126, 168]:
                root = os.path.join(args.parent_dir, experiment)
                model = os.path.join(
                    root, "models", "checkpoints",
                    "rnet_dqn-combined_linear-%s-%d-0" % (generator, model_seed),
                    "rnet_dqn_agent.model")
                suffix = "" if model_seed == 0 else "_seed%d" % model_seed
                manifest = os.path.join(root, "complete_%s%s.json" %
                                        (generator, suffix))
                if not os.path.isfile(model) or not os.path.getsize(model):
                    failures.append("missing model " + model)
                    continue
                if not os.path.isfile(manifest) or not os.path.getsize(manifest):
                    failures.append("missing manifest " + manifest)
                    continue
                with open(manifest) as handle:
                    details = json.load(handle)
                if details.get("status") != "complete":
                    failures.append("invalid manifest " + manifest)
                    continue
                completed += 1
    print("completed=" + str(completed))
    print("expected=110")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
