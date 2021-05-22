import sys
import itertools

import numpy as np

sys.path.insert(0, '..')
from commons.factor import Factor
import solution as sol


def create_clique_tree(factors, evidence=[]):
    V, domains = set(), dict()
    for f in factors:
        V.update(f.vars)
        for v, d in zip(f.vars, f.domains):
            if v in domains:
                assert domains[v] == d, "Domain mismatch between factors"
            else:
                domains[v] = d

    adj_list = {v: {v, } for v in V}
    for f in factors:
        for u, v in itertools.permutations(f.vars, 2):
            adj_list[u].add(v)

    cliques_considered = 0
    F = factors
    skeleton = {'nodes': [], 'factor_idxs': [], 'edges': [], 'factor_list': factors}
    while cliques_considered < len(V):
        next_var = min(adj_list, key=lambda x: len(adj_list[x]))
        F = eliminate_var(F, adj_list, next_var, skeleton)
        cliques_considered += 1
        if not adj_list:
            break

    prune_tree(skeleton)
    return sol.compute_initial_potentials(skeleton)


def eliminate_var(F, adj_list, next_var, skeleton):
    use_factors, non_use_factors = [], []
    scope = set()
    for i, f in enumerate(F):
        if next_var in f.vars:
            use_factors.append(i)
            scope.update(f.vars)
        else:
            non_use_factors.append(i)
    scope = sorted(scope)

    for i, j in itertools.permutations(scope, 2):
        if i not in adj_list:
            adj_list[i] = {j, }
        else:
            adj_list[i].add(j)

    # next steps removes the next_var from adj_list
    for k in adj_list:
        if next_var in adj_list[k]:
            adj_list[k].remove(next_var)
    del adj_list[next_var]

    newF, newmap = [], {}
    for i in non_use_factors:
        newmap[i] = len(newF)
        newF.append(F[i])

    new_factor = Factor([], [])
    for i in use_factors:
        new_factor = new_factor @ F[i]  # Since this just a simulation, we don't really need to compute values. So @
    new_factor = new_factor.dummy_marginalise({next_var, })
    newF.append(new_factor)

    for i in range(len(skeleton['nodes'])):
        if skeleton['factor_idxs'][i] in use_factors:
            skeleton['edges'].append((skeleton['nodes'][i], set(scope)))
            skeleton['factor_idxs'][i] = None
        elif skeleton['factor_idxs'][i] is not None:
            skeleton['factor_idxs'][i] = newmap[skeleton['factor_idxs'][i]]
    skeleton['nodes'].append(set(scope))
    skeleton['factor_idxs'].append(len(newF) - 1)

    return newF


def prune_tree(skeleton):
    found = True
    while found:
        found = False

        for u, v in skeleton['edges']:
            if u.issuperset(v):
                found = True
                parent = u
                child = v
                break
            elif v.issuperset(u):
                found = True
                parent = v
                child = u
                break

        if not found:
            break

        new_edges = []
        for u, v in skeleton['edges']:
            if (u, v) == (child, parent) or (v, u) == (child, parent):
                continue
            elif u == child:
                new_edges.append((parent, v))
            elif v == child:
                new_edges.append((u, parent))
            else:
                new_edges.append((u, v))
        skeleton['edges'] = new_edges
        skeleton['nodes'] = [node for node in skeleton['nodes'] if node != child]


def sigmoid(x):
    return 1/(1+np.exp(-x))


def lr_predict(X, theta):
    return ((X@theta) > 0).astype(np.uint8)


def lr_accuracy(ground_truth, prediction):
    return (ground_truth == prediction).mean()


def grad_fn_generator(X, y, lambda_):
    N = X.shape[0]

    def lr_grad(theta, i):
        i = ((i+1) % N)
        x = X[i]
        h = sigmoid(x@theta)
        cost = -y[i]*np.log(h) - (1-y[i])*np.log(1 - h) + 0.5*lambda_*(theta[1:]**2).sum()
        grad = x * (h-y[i])
        grad[1:] += lambda_*theta[1:]
        return cost, grad

    return lr_grad


def lr_train(X, y, lambda_):
    grad_fn = grad_fn_generator(X, y, lambda_)
    theta_opt = sol.stochastic_gradient_descent(grad_fn, np.zeros(X.shape[1]), 5000)
    return theta_opt


def from_mat_to_tree(data):
    factors = [Factor.from_matlab(clique) for clique in data['cliqueList']]
    edges = data['edges']
    tree = {'clique_list': factors, 'adj_list': {}}

    for i in range(len(edges)):
        tree['adj_list'][i] = set()
        for j in range(len(edges)):
            if edges[i, j] == 1:
                tree['adj_list'][i].add(j)
                tree['adj_list'][i].add(j)
    return tree