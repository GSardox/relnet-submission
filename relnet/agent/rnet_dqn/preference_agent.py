from pathlib import Path

import numpy as np

from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.agent.rnet_dqn.q_net import PreferenceNStepQNet
from relnet.objective_functions.objective_functions import \
    CriticalFractionTargeted, GlobalEfficiency
from relnet.utils.config_utils import get_device_placement


class PreferenceConditionedRNetDQNAgent(RNetDQNAgent):
    algorithm_name = "rnet_dqn_pref"

    def setup(self, options, hyperparams):
        self.training_weights = [float(weight) for weight in options['training_weights']]
        if len(self.training_weights) < 2:
            raise ValueError("preference-conditioned training requires at least two weights")
        if len(set(self.training_weights)) != len(self.training_weights):
            raise ValueError("training weights must be unique")
        if any(weight < 0. or weight > 1. for weight in self.training_weights):
            raise ValueError("training weights must lie in [0, 1]")

        self.preference_schedule = []
        self.current_weight = self.training_weights[0]
        self.preference_hist_out = None
        self.last_validation_rewards = {}
        super().setup(options, hyperparams)

    def setup_nets(self):
        self.net = PreferenceNStepQNet(self.hyperparams, num_steps=2)
        self.old_net = PreferenceNStepQNet(self.hyperparams, num_steps=2)
        if get_device_placement() == 'GPU':
            self.net = self.net.cuda()
            self.old_net = self.old_net.cuda()
        if self.restore_model:
            self.restore_model_from_checkpoint()

    def compute_components(self, graph_list):
        objective = self.environment.objective_function
        kwargs = self.environment.original_objective_function_kwargs

        efficiency = np.array([
            GlobalEfficiency.compute(graph, **kwargs) for graph in graph_list
        ])
        robustness = np.array([
            CriticalFractionTargeted.compute(graph, **kwargs) for graph in graph_list
        ])

        norm_efficiency = (
            efficiency - objective.efficiency_initial
        ) / objective.efficiency_reference_gain
        norm_robustness = (
            robustness - objective.robustness_initial
        ) / objective.robustness_reference_gain
        return norm_efficiency, norm_robustness

    @staticmethod
    def combine_components(components, weight, indices=None):
        efficiency, robustness = components
        if indices is not None:
            efficiency = efficiency[indices]
            robustness = robustness[indices]
        return weight * efficiency + (1. - weight) * robustness

    def setup_graphs(self, train_g_list, validation_g_list):
        self.train_g_list = train_g_list
        self.validation_g_list = validation_g_list
        self.train_components = self.compute_components(train_g_list)
        self.validation_components = self.compute_components(validation_g_list)

    def next_training_weight(self):
        if not self.preference_schedule:
            self.preference_schedule = list(self.training_weights)
            self.local_random.shuffle(self.preference_schedule)
        return self.preference_schedule.pop()

    @staticmethod
    def condition_states(states, weight):
        return [
            (graph, picked_node, banned_actions, float(weight))
            for graph, picked_node, banned_actions in states
        ]

    def current_conditioned_state(self):
        return self.condition_states(
            list(self.environment.get_state_ref()),
            self.current_weight,
        )

    def do_greedy_actions(self, time_t):
        actions, _, _ = self.net(
            time_t % 2,
            self.current_conditioned_state(),
            None,
            greedy_acts=True,
        )
        return list(actions.cpu().numpy())

    def before_simulation(self):
        self.current_weight = self.next_training_weight()
        self.environment.objective_function.weight = self.current_weight

    def training_initial_values(self, selected_idx):
        return self.combine_components(
            self.train_components,
            self.current_weight,
            selected_idx,
        )

    def prepare_replay_states(self, states):
        return self.condition_states(states, self.current_weight)

    def setup_histories_file(self):
        super().setup_histories_file()
        detailed_path = Path(self.eval_histories_path) / (
            self.model_identifier_prefix + "_preferences.csv"
        )
        if detailed_path.exists():
            detailed_path.unlink()
        self.preference_hist_out = open(str(detailed_path), "a")
        self.preference_hist_out.write("timestep,weight,performance\n")

    def eval_at_weight(self, graph_list, weight, components=None, validation=False):
        self.current_weight = float(weight)
        self.environment.objective_function.weight = self.current_weight
        if components is None:
            components = self.compute_components(graph_list)
        initial_values = self.combine_components(components, self.current_weight)
        return self.eval(
            graph_list,
            initial_obj_values=initial_values,
            validation=validation,
        )

    def evaluate_preferences(self, graph_list, weights=None, validation=False):
        weights = self.training_weights if weights is None else weights
        components = self.compute_components(graph_list)
        return {
            float(weight): float(self.eval_at_weight(
                graph_list,
                float(weight),
                components,
                validation=validation,
            ))
            for weight in weights
        }

    def log_validation_loss(self, step, make_action_kwargs=None):
        rewards = {}
        for weight in self.training_weights:
            performance = self.eval_at_weight(
                self.validation_g_list,
                weight,
                self.validation_components,
                validation=True,
            )
            rewards[weight] = float(performance)
            self.preference_hist_out.write(
                "%d,%.6f,%.6f\n" % (step, weight, performance)
            )

        mean_performance = float(np.mean(list(rewards.values())))
        self.last_validation_rewards = rewards
        self.preference_hist_out.flush()

        if self.hist_out is not None:
            self.hist_out.write("%d,%.6f\n" % (step, mean_performance))
            self.hist_out.flush()

        return self.environment.objective_function.upper_limit - mean_performance

    def finalize(self):
        if self.hist_out is not None and not self.hist_out.closed:
            self.hist_out.close()
        if self.preference_hist_out is not None and not self.preference_hist_out.closed:
            self.preference_hist_out.close()
