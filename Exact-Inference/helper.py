import sys
import itertools
import pprint

sys.path.insert(0, '..')
from commons.factor import Factor
import solution as sol

def create_clique_tree(factors, evidence=None):
    V, domains = set(), dict()
    for factor in factors:
        V.update(factor.vars)
        for v, d in zip(factor.vars, factor.domains):
            if v in domains:
                assert domains[v] == d, "Domain mismatch between factors"
            else:
                domains[v] = d

    adj_list = {v: {v, } for v in V}
    for factor in factors:
        for u, v in itertools.permutations(factor.vars, 2):
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
    tree = sol.compute_initial_potentials(skeleton)
    if evidence:
        for i, f in enumerate(tree['clique_list']):
            tree['clique_list'][i] = f.evidence(evidence)

    return tree


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


def adj_matrix_to_adj_list(matrix):
    edges = {}
    for i in range(len(matrix)):
        nbs = set()
        for j in range(len(matrix)):
            if matrix[i, j] == 1:
                nbs.add(j)
        edges[i] = nbs
    return edges


if __name__ == '__main__':
    fs = []
    fs.append(Factor(['C'], [2]))
    fs.append(Factor(['C', 'D'], [2, 2]))
    fs.append(Factor(['I'], [2]))
    fs.append(Factor(['G', 'I', 'D'], [2, 2, 2]))
    fs.append(Factor(['I', 'S'], [2, 2]))
    fs.append(Factor(['J', 'S', 'L'], [2, 2, 2]))
    fs.append(Factor(['G', 'L'], [2, 2]))
    fs.append(Factor(['G', 'H'], [2, 2]))
    for f in fs:
        for a in itertools.product(*f.domains):
            f[a] = 42
    pprint.pprint(create_clique_tree(fs, []))
