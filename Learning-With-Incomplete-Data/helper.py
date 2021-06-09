import sys
import itertools

import numpy as np
from scipy.special import logsumexp
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

sys.path.append('..')
from commons.factor import Factor

from vis_helper import *


def wavg(x, w, axis=None):
    if axis is None:
        return np.average(x, weights=w)
    return np.average(x, weights=w, axis=axis)


def fit_g(x, w):
    """
    Args:
        x: numpy array with shape (N, )
        w: weights of shape (N, )
        
    Returns:
        mu, sigma
    """
    
    mu = wavg(x, w)
    sigma = np.sqrt(wavg(x*x, w) - mu**2)
    
    return mu, sigma




def fit_lg(x, u, w):
    M, N = len(x), u.shape[1]
    beta = np.zeros(N+1)
    sigma = 1

    A = np.zeros((N+1, N+1))
    A[0, :-1] = np.average(u, weights=w, axis=0)
    A[0, -1] = 1

    A[1:, -1] = A[0, :-1]
    for i in range(N):
        A[i+1, :-1] = np.average(u * u[:, i][:, None], weights=w, axis=0)

    B = np.zeros(N+1)
    B[0] = np.average(x, weights=w)
    B[1:] = np.average(u * x[:, None], weights=w, axis=0)

    beta = np.linalg.solve(A, B)
    
    var = wavg(x*x, w) - wavg(x, w)**2

    for i in range(N):
        for j in range(N):
            cov = wavg(u[:, i]*u[:, j], w) - wavg(u[:, i], w)*wavg(u[:, j], w)
            var -= beta[i] * beta[j] * cov
    if var < 0:
        var = 0.
    sigma = np.sqrt(var)
    
    if sigma == 0:
        sigma = .01;
    else:
        sigma = sigma + .01

    return beta, sigma
 

# Clique Tree related

def create_clique_tree_hmm(factors):
    max_var = max(max(f.vars) for f in factors)
    card = len(factors[0].domains[0])
    tree = {
        'clique_list': [Factor([i, i+1], [card, card], init=1) for i in range(max_var)], 
        'adj_list': {i: set() for i in range(max_var)}
    }
    
    for i in range(max_var):
        if i > 0:
            tree['adj_list'][i].add(i-1)
            tree['adj_list'][i-1].add(i)
        if i < max_var-1:
            tree['adj_list'][i].add(i+1)
            tree['adj_list'][i+1].add(i)
    
    for f in factors:
        if len(f.vars) == 1:
            if f.vars[0] == 0:
                clique_idx = 0
            else:
                clique_idx = f.vars[0]-1
        else:
            clique_idx = min(f.vars)
            
        tree['clique_list'][clique_idx] = tree['clique_list'][clique_idx] + f
    return tree
        
        
def get_next_clique(clique_tree, msgs):
    adj = clique_tree['adj_list']

    for u in adj:
        n_neighbours = len(adj[u])
        for v in adj[u]:
            if u not in msgs[v] and sum(1 for w in msgs[u] if v != w) == n_neighbours - 1:
                return u, v
    return None


def log_marginalise(factor, vars_to_marginalise):
    new_factor = factor.dummy_marginalise(vars_to_marginalise)
    new_vars_idx = [i for i, v in enumerate(factor.vars) if v not in vars_to_marginalise]

    tmp_dict = {}
    for assignment in itertools.product(*new_factor.domains):
        tmp_dict[assignment] = []

    for assignment in itertools.product(*factor.domains):
        new_assignment = tuple(assignment[i] for i in new_vars_idx)
        tmp_dict[new_assignment].append(factor.val[assignment])
        
    for assignment, values in tmp_dict.items():
        new_factor[assignment] = logsumexp(values)
    
    return new_factor


def clique_tree_calibrate(clique_tree, is_max=0):
    # Note: msgs[u] = {v: msg_from_v_to_u}
    msgs = {i: {} for i in range(len(clique_tree['clique_list']))}
    adj = clique_tree['adj_list']
    cliques = clique_tree['clique_list']


    while True:
        ready_edge = get_next_clique(clique_tree, msgs)
        if ready_edge is None:
            break
        u, v = ready_edge
        msg = cliques[u]
        for w in adj[u]:
            if w == v:
                continue
            msg = msg + msgs[u][w]
            
        diff_set = set(msg.vars) - (set(msg.vars) & set(cliques[v].vars))

        msg = log_marginalise(msg, diff_set)
#         z = logsumexp(list(msg.val.values()))
#         for assignment in msg:
#             msg[assignment] -= z
        msgs[v][u] = msg

    calibrated_potentials = []
    for i in range(len(cliques)):
        factor = cliques[i]
        for msg in msgs[i].values():
            factor = factor + msg
        calibrated_potentials.append(factor)

    return {'clique_list': calibrated_potentials, 'adj_list': adj}


def compute_exact_marginals_hmm(factors):
    clique_tree = create_clique_tree_hmm(factors)
    calibrated_tree = clique_tree_calibrate(clique_tree)
    calibrated_cliques = calibrated_tree['clique_list']
    
    variables = set()
    for f in factors:
        variables.update(f.vars)
    variables = sorted(list(variables))
    
    marginals = [None]*len(variables)
    for var in variables:
        if var == 0:
            clique = calibrated_cliques[0]
            clique = log_marginalise(clique, {1, })
        else:
            clique = calibrated_cliques[var-1]
            clique = log_marginalise(clique, {var-1, })
        
        z = logsumexp(list(clique.val.values()))
        for k in clique.val:
            clique.val[k] -= z
        
        marginals[var] = clique
        
    return marginals, calibrated_cliques
        
            
        