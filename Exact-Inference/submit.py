import itertools
import sys

from scipy.io import loadmat
import numpy as np

import helper
import solution as sol

sys.path.insert(0, '..')

import commons
from commons.factor import Factor


class Grader(commons.SubmissionBase):
    def __init__(self):
        part_names = [None, 
                      'u4XdY', 'SFny2', 'vJtk1', 'oxFvg', 
                      'mdAFl', '4nqJB', 'lXwFM', 'rKODa', 
                      'IxUeH', 'cKBqa', 'PxFaH', 'i3nTw', 
                      'H3HVG', '47MjN', 'x0vXX', 'XGmvB']
        super().__init__('Exact Inference', 'ITvOkANgEea1SAr5vIqVXQ', part_names)

    def __iter__(self):
        for part_id in range(1, len(self.part_names)):
            if part_id % 2 == 0:
                mat = loadmat('./data/PA4Test.mat', simplify_cells=True)
            else:
                mat = loadmat('./data/PA4Sample.mat', simplify_cells=True)

            try:
                if part_id == 1 or part_id == 2:
                    inp = mat['InitPotential']['INPUT']
                    nodes = [set(x - 1) for x in inp['nodes']]

                    edges = []
                    edge_mat = inp['edges']
                    for i in range(len(edge_mat)):
                        for j in range(i + 1, len(edge_mat)):
                            if edge_mat[i, j] == 1:
                                edges.append((nodes[i], nodes[j]))

                    factors = [Factor.from_matlab(f) for f in inp['factorList']]
                    skeleton = {'nodes': nodes, 'edges': edges, 'factor_list': factors}
                    clique_tree = sol.compute_initial_potentials(skeleton)
                    res = serialize_compact_tree(clique_tree, part_id)
                elif part_id == 3 or part_id == 4:
                    arg1, arg2 = mat['GetNextC']['INPUT1'], mat['GetNextC']['INPUT2']
                    clique_tree = {'clique_list': [Factor.from_matlab(f) for f in arg1['cliqueList']],
                                   'adj_list': helper.adj_matrix_to_adj_list(arg1['edges'])}
                    N = arg2.shape[0]
                    msgs = {i: {} for i in range(N)}
                    for i in range(N):
                        for j in range(N):
                            if isinstance(arg2[i, j].var, int) or len(arg2[i, j].var) > 0:
                                msgs[j][i] = Factor.from_mat_struct(arg2[i, j])

                    res = sol.get_next_clique(clique_tree, msgs)
                    if res is None:
                        i, j = 0, 0
                    else:
                        i, j = res
                    res = "%d %d" % (i+1, j+1)
                elif part_id == 5 or part_id == 6:
                    inp = mat['SumProdCalibrate']['INPUT']
                    clique_tree = {'clique_list': [Factor.from_matlab(f) for f in inp['cliqueList']],
                                   'adj_list': helper.adj_matrix_to_adj_list(inp['edges'])}
                    calibrated_tree = sol.clique_tree_calibrate(clique_tree)
                    res = serialize_compact_tree(calibrated_tree)
                elif part_id == 7 or part_id == 8:
                    inp = mat['ExactMarginal']['INPUT']
                    inp = [Factor.from_matlab(f) for f in inp]
                    out = sol.compute_exact_marginals_bp(inp, evidence=None, is_max=0)
                    for f in out:
                        f.vars = [v+1 for v in f.vars]
                    res = serialize_factors_fg_grading(out)
                elif part_id == 9 or part_id == 10:
                    arg1 = Factor.from_matlab(mat['FactorMax']['INPUT1'], start_from_zero=False)
                    arg2 = mat['FactorMax']['INPUT2']
                    out = sol.factor_max_marginalization(arg1, [arg2, ])
                    res = serialize_factors_fg_grading([out])
                elif part_id == 11:
                    res = ''  # Coursera seems to have a bug for this case
                elif part_id == 12:
                    arg1 = mat['MaxSumCalibrate']['INPUT']
                    clique_tree = {'clique_list': [Factor.from_matlab(clique) for clique in arg1['cliqueList']],
                                   'adj_list': helper.adj_matrix_to_adj_list(arg1['edges'])}
                    calibrated_tree = sol.clique_tree_calibrate(clique_tree, is_max=1)
                    res = serialize_compact_tree(calibrated_tree)
                elif part_id == 13 or part_id == 14:
                    arg1 = [Factor.from_matlab(f) for f in mat['MaxMarginals']['INPUT']]
                    out = sol.compute_exact_marginals_bp(arg1, evidence=None, is_max=1)
                    for f in out:
                        f.vars = [v + 1 for v in f.vars]
                    res = serialize_factors_fg_grading(out)
                elif part_id == 15 or part_id == 16:
                    arg1 = [Factor.from_matlab(f) for f in mat['MaxDecoded']['INPUT']]
                    res = " ".join(map(str, sol.max_decoding(arg1)+1))
                yield self.part_names[part_id], res
            except KeyError:
                raise


def serialize_factors_fg_grading(factors, skip=1) -> str:
    lines = ["%d\n" % len(factors)]

    for f in factors:
        lines.append("%d" % (len(f.vars, )))
        lines.append("  ".join(map(str, f.vars)))
        lines.append("  ".join(str(len(d)) for d in f.domains))
        placeholder_idx = len(lines)
        lines.append(None)  # will be replace by nonzero count once we know

        # libDAI expects first variable to change fastest
        # but itertools.product changes the last element fastest
        # hence reversed list
        domains = reversed(f.domains)
        num_lines = 0
        new_lines = []
        for i, assignment in enumerate(itertools.product(*domains)):
            num_lines += 1
            val = f[tuple(reversed(assignment))]
            if abs(val) <= 1e-40 or abs(val - 1) <= 1e-40 or np.isinf(val) or np.isnan(val):
                continue
            new_lines.append("%0.8g" % (val, ))
        new_lines = new_lines[::skip]
        lines[placeholder_idx] = "%d" % (num_lines, )
        lines.extend(new_lines)
        lines.append("")

    return "\n".join(lines)


def serialize_compact_tree(tree, skip=1) -> str:
    adj_list = tree['adj_list']
    N = len(adj_list)
    lines = ['%d' % N]

    for i in range(N):
        nbs = adj_list[i]
        lines.append("  ".join("1" if i in nbs else "0" for i in range(N)))

    # convert 0 based index into 1 based index!! Uuhhh!!
    for factor in tree['clique_list']:
        factor.vars = [v+1 for v in factor.vars]

    factor_graph = serialize_factors_fg_grading(tree['clique_list'], skip)
    lines.append(factor_graph)

    # convert 1 based index into 0 based index!! Uuhhh!!
    for factor in tree['clique_list']:
        factor.vars = [v - 1 for v in factor.vars]

    return '\n'.join(lines)


if __name__ == '__main__':
    grader = Grader()
    grader.grade()
