import sys

import numpy as np

import helper
sys.path.insert(0, '..')
from commons.factor import Factor


def stochastic_gradient_descent(grad_func, theta0, max_iter):
    """
    Args:
        grad_fund: A function f: (theta, i) -> (cost, grad),
            returns cost and gradient for instance i wrt current theta.
        theta0: initial theta
        max_iter: run the loop for these many iterations.
        
    Returns:
        theta_opt: Theta after max_iter iterations of SGD.
    """
    theta_opt = theta0.copy()

    # Solution Start
    
    # Solution End
    return theta_opt


def lr_search_lambda_sgd(x_train, y_train, x_validation, y_validation, lambdas):
    """
    Args:
        x_train: training numpy array of shape (N, 129)
        y_train: training numpy array of shape (N, )
        x_validation: validation numpy array of shape (M, 129)
        y_validation: validation numpy array of shape (M, )
        lambdas: a numpy array of candidate regularization parameters.
        
    Returns: a numpy array containing accuracy for each lambda.
        See helper.lr_train, helper.lr_predict, and helper.lr_accuarcy
    """
    acc = np.zeros(len(lambdas))

    # Solution Start
    
    # Solution End

    return acc

def compute_initial_potentials(skeleton):
    """
    Args:
        skeleton: a dictionary with following keys.
            'nodes': a list of sets. Each set is a set of constituent variables. e.g. {1,2,3}
            'edges': a list of edges. A single element would look like ({1,2,3}, {2,3,4})
                which means there is an edge between node {1,2,3} and node {2,3,4}. If (a, b) is
                in the list, (b, a) will not be in the list.
            'factor_list': a list of initialized Factors.

    Returns:
        a dict with ['clique_list', 'edges'] keys.
        'clique_list': a list of factors associated with each clique
        'adj_list': adjacency list with integer nodes. adj_list[0] = {1,2}
            implies that there is an edges  clique_list[0]-lique_list[1]
            and clique_list[0]-clique_list[2]
    """
    n = len(skeleton['nodes'])
    clique_list = [Factor([], []) for i in range(n)]
    adj_list = {i: set() for i in range(n)}

    nodes = skeleton['nodes']
    edges = skeleton['edges']

    for (u, v) in edges:
        u_idx = nodes.index(u)
        v_idx = nodes.index(v)
        adj_list[u_idx].add(v_idx)
        adj_list[v_idx].add(u_idx)

    for factor in skeleton['factor_list']:
        for candidate in np.random.permutation(n):  # A heuristic, not sure if it is better
            if nodes[candidate].issuperset(set(factor.vars)):
                clique_list[candidate] = clique_list[candidate] * factor
                break
                # This approach is not efficient as it computes intermediate factors
                # even though we could know all constituent factors and multiply them
                # in one go.

    return {'clique_list': clique_list, 'adj_list': adj_list}


def get_next_clique(clique_tree):
    adj = clique_tree['adj_list']
    msgs = clique_tree['messages']

    for u in adj:
        n_neighbours = len(adj[u])
        for v in adj[u]:
            if u not in msgs[v] and sum(1 for w in msgs[u] if v != w) == n_neighbours - 1:
                return u, v
    return None


def clique_tree_calibrate(clique_tree, do_logZ=False):
    """
    Args:
        clique_tree: A dict with ['clique_list', 'edges'] keys.
            'clique_list': a list of factors associated with each clique
            'adj_list': adjacency list with integer nodes. adj_list[0] = {1,2}
                implies that there is an edges  clique_list[0]-clique_list[1]
                and clique_list[0]-clique_list[2]
        do_logZ: If True, also returns logZ
    """
    clique_tree['messages'] = {i: {} for i in range(len(clique_tree['clique_list']))}
    msgs = clique_tree['messages']
    if do_logZ:
        msgs_unnorm = {i: {} for i in range(len(clique_tree['clique_list']))}
    adj = clique_tree['adj_list']
    ## clique_tree['messages'][u] = {v: msg_from_v_to_u}

    cliques = clique_tree['clique_list']

    while True:
        ready_edge = get_next_clique(clique_tree)
        if ready_edge is None:
            break
        u, v = ready_edge
        msg_norm = cliques[u]
        msg_unnormalized = cliques[u]
        for w in adj[u]:
            if w == v:
                continue
            msg_norm = msg_norm * msgs[u][w]
            if do_logZ:
                msg_unnormalized = msg_unnormalized * msgs_unnorm[u][w]

        diff_set = set(msg_norm.vars) - (set(msg_norm.vars) & set(cliques[v].vars))

        msg_norm = msg_norm.marginalise(diff_set)
        normalizer = sum(msg_norm.val.values())
        for assignment in msg_norm:
            msg_norm[assignment] /= normalizer
        msgs[v][u] = msg_norm

        if do_logZ:
            msgs_unnorm[v][u] = msg_unnormalized.marginalise(diff_set)

    
    if do_logZ:
        logZ = 0
        # Solution Start
    
        # Solution End

    # return clique_tree
    calibrated_potentials = []
    for i in range(len(cliques)):
        factor = cliques[i]
        for msg in msgs[i].values():
            factor = factor * msg
        calibrated_potentials.append(factor)

    if do_logZ:
        return calibrated_potentials, logZ
    else:
        return calibrated_potentials
    