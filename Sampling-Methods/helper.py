import sys

import numpy as np

sys.path.insert(0, '../')
from commons.factor import Factor


def extract_marginals_from_samples(G, collected_samples):
    M = [Factor([i], [card], init=0.) for i, card in enumerate((G['card']))]

    for sample in collected_samples:
        for i, v in enumerate(sample):
            M[i][v] += 1/len(collected_samples)
    return M


def variable_2_factor(V, F):
    var2f = {i: set() for i in range(V)}
    for i, f in enumerate(F):
        for j in f.vars:
            var2f[j].add(i)
    return var2f


def construct_toy_network(on_diag_weight, off_diag_weight):
    n = 4
    k = 2
    V = n * n

    G = {
        'names': ['pixel%d' % i for i in range(V)],
        'card': [2]*V
    }

    adj_list = {i: set() for i in range(V)}
    for i in range(V):
        for j in range(i+1, V):
            idx_i = np.array(np.unravel_index(i, [n, n], order='F'))
            idx_j = np.array(np.unravel_index(j, [n, n], order='F'))
            if abs(idx_i - idx_j).sum() == 1:
                adj_list[i].add(j)
                adj_list[j].add(i)
    G['adj_list'] = adj_list

    factors = []
    for i in range(V):
        f = Factor([i, ], [2, ])
        if i < V//2:
            f[0] = .4
            f[1] = .6
        else:
            f[0] = .6
            f[1] = .4
        factors.append(f)

    for u, vs in adj_list.items():
        for v in vs:
            if u >= v:
                continue

            f = Factor([u, v], [2, 2])
            f[0, 0] = f[1, 1] = on_diag_weight
            f[0, 1] = f[1, 0] = off_diag_weight
            factors.append(f)

    G['var2factors'] = variable_2_factor(V, factors)
    return G, factors


