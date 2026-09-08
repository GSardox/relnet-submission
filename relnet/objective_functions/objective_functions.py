from relnet.objective_functions.objective_functions_ext import *
from relnet.state.graph_state import get_graph_hash


def extract_kwargs(kwargs):
    num_mc_sims = 20
    random_seed = 42
    if 'num_mc_sims' in kwargs:
        num_mc_sims = kwargs['num_mc_sims']
    if 'random_seed' in kwargs:
        random_seed = kwargs['random_seed']
    return num_mc_sims, random_seed


class CriticalFractionRandom(object):
    name = "random_removal"
    upper_limit = 1.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        num_mc_sims, random_seed = extract_kwargs(kwargs)
        N, M, edges = s2v_graph.num_nodes, s2v_graph.num_edges, s2v_graph.edge_pairs
        graph_hash = get_graph_hash(s2v_graph)
        frac = critical_fraction_random(N, M, edges, num_mc_sims, graph_hash, random_seed)
        return frac


class CriticalFractionTargeted(object):
    name = "targeted_removal"
    upper_limit = 1.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        num_mc_sims, random_seed = extract_kwargs(kwargs)
        N, M, edges = s2v_graph.num_nodes, s2v_graph.num_edges, s2v_graph.edge_pairs
        graph_hash = get_graph_hash(s2v_graph)
        frac = critical_fraction_targeted(N, M, edges, num_mc_sims, graph_hash, random_seed)
        return frac


class GlobalEfficiency(object):
    name = "global_efficiency"
    upper_limit = 1.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path

        N = s2v_graph.num_nodes
        if N <= 1:
            return 0.0
        ep = np.asarray(s2v_graph.edge_pairs).reshape(-1, 2)
        if ep.shape[0] == 0:
            return 0.0

        rows = np.concatenate([ep[:, 0], ep[:, 1]])
        cols = np.concatenate([ep[:, 1], ep[:, 0]])
        adj = csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(N, N))

        dist = shortest_path(adj, method='D', unweighted=True)
        with np.errstate(divide='ignore'):
            inv = 1.0 / dist
        inv[~np.isfinite(inv)] = 0.0
        np.fill_diagonal(inv, 0.0)
        return float(inv.sum() / (N * (N - 1)))


class LinearCombinedObjective(object):
    name = "combined_linear"
    upper_limit = 1.
    robustness_objective = CriticalFractionTargeted

    def __init__(self, weight=0.5, efficiency_reference_gain=1., robustness_reference_gain=1.,
                 efficiency_initial=0., robustness_initial=0.):
        self.weight = weight
        self.efficiency_reference_gain = efficiency_reference_gain
        self.robustness_reference_gain = robustness_reference_gain
        self.efficiency_initial = efficiency_initial
        self.robustness_initial = robustness_initial

    def compute(self, s2v_graph, **kwargs):
        rob = self.robustness_objective.compute(s2v_graph, **kwargs)
        eff = GlobalEfficiency.compute(s2v_graph, **kwargs)

        norm_eff = (eff - self.efficiency_initial) / self.efficiency_reference_gain
        norm_rob = (rob - self.robustness_initial) / self.robustness_reference_gain

        return self.weight * norm_eff + (1.0 - self.weight) * norm_rob
