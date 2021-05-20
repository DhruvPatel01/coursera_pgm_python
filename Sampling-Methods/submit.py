import itertools
import sys

from scipy.io import loadmat
import numpy as np

import helper
import solution
import drandom

sys.path.insert(0, '..')

import commons
from commons.factor import Factor


def _get_toy1(m2=False):
    mat = loadmat('./data/submit_input.mat', simplify_cells=True)['INPUT']
    if m2:
        G = mat['toyNet_m2']
    else:
        G = mat['toyNet']
    G['adj_list'] = commons.adj_matrix_to_adj_list(G['edges'])
    G['var2factors'] = np.array([x - 1 for x in G['var2factors']], dtype='object')
    G['q_list'][:, 0] -= 1
    G['q_list'][:, 1] -= 1
    F = np.array([Factor.from_matlab(f) for f in mat['toyFac']])
    return mat, G, F, mat['A0'] - 1

def _get_toy2(m2=False):
    mat = loadmat('./data/submit_input.mat', simplify_cells=True)['INPUT']
    if m2:
        G = mat['toyNet2_m2']
    else:
        G = mat['toyNet2']
    G['adj_list'] = commons.adj_matrix_to_adj_list(G['edges'])
    G['var2factors'] = np.array([x - 1 for x in G['var2factors']], dtype='object')
    G['q_list'][:, 0] -= 1
    G['q_list'][:, 1] -= 1
    F = np.array([Factor.from_matlab(f) for f in mat['toyFac2']])
    return mat, G, F, mat['A0'] - 1

class Grader(commons.SubmissionBase):
    def __init__(self):
        part_names = [None, 
                      'oLlCG', 'u0bdr', 'ZvlVb', 'kAwIs', 
                      'BnmgG', 'x4EL4', '7vRgM', 'o2kO8', 
                      'E3nMZ', 'NYOmz', 'IhNcD', 'IWmCs', 
                      'R9Ium', 'RiDeK']
        super().__init__('Sampling Methods', 'RT4_2ANgEeacSgpo_ExyYQ', part_names)

    def __iter__(self):
        for part_id in range(1, len(self.part_names)):
            try:
                if part_id == 1:
                    drandom.seed(1)
                    mat, G, F, A = _get_toy1()
                    V = [0]
                    res = solution.block_log_distribution(V, G, F, A.copy())
                elif part_id == 2:
                    drandom.seed(2)
                    mat, G, F, A = _get_toy1()
                    res1 = solution.block_log_distribution([9], G, F, A.copy())
                    res2 = solution.block_log_distribution([14], G, F, A.copy())
                    res = np.r_[res1, res2]
                elif part_id == 3:
                    mat, G, F, A = _get_toy1()
                    drandom.seed(1)
                    out = [A.copy()]
                    for i in range(10):
                        out.append(solution.gibbs_trans(out[-1].copy(), G, F))
                    res = serialize_matrix(out)
                elif part_id == 4:
                    mat, G, F, A = _get_toy2()
                    drandom.seed(2)
                    out = [A.copy()]
                    for i in range(20):
                        out.append(solution.gibbs_trans(out[-1].copy(), G, F))
                    res = serialize_matrix(out)
                    with open('/tmp/pyPart4.txt','w') as f:
                        f.write(res)
                elif part_id == 5:
                    drandom.seed(1)
                    mat, G, F, A = _get_toy1()
                    M, all_samples = solution.mcmc_inference(G, F, None, "Gibbs", 0, 500, 1, A.copy())
                    for f in M:
                        f.vars = [v+1 for v in f.vars]
                    res = serialize_factors_fg_grading(M)
                    
                    M, all_samples = solution.mcmc_inference(G, F, None, "MHGibbs", 0, 500, 1, A.copy())
                    for f in M:
                        f.vars = [v+1 for v in f.vars]
                    res = res.strip()+'\n'+serialize_factors_fg_grading(M)
                    with open('/tmp/pyPart5.txt','w') as f:
                        f.write(res)
                elif part_id == 6:
                    drandom.seed(1)
                    mat, G, F, A = _get_toy2()
                    M, all_samples = solution.mcmc_inference(G, F, None, "Gibbs", 0, 500, 1, A.copy())
                    for f in M:
                        f.vars = [v+1 for v in f.vars]
                    res = serialize_factors_fg_grading(M)
                    
                    M, all_samples = solution.mcmc_inference(G, F, None, "MHGibbs", 0, 500, 1, A.copy())
                    for f in M:
                        f.vars = [v+1 for v in f.vars]
                    res = res.strip()+'\n'+serialize_factors_fg_grading(M)
                    with open('/tmp/pyPart5.txt','w') as f:
                        f.write(res)
                elif part_id == 7:
                    mat, G, F, A = _get_toy1()
                    drandom.seed(1)
                    out = [A.copy()]
                    for i in range(10):
                        out.append(solution.mh_uniform_trans(out[-1].copy(), G, F))
                    res = serialize_matrix(out)
                elif part_id == 8:
                    mat, G, F, A = _get_toy2()
                    drandom.seed(2)
                    out = [A.copy()]
                    for i in range(20):
                        out.append(solution.mh_uniform_trans(out[-1].copy(), G, F))
                    res = serialize_matrix(out)
                elif part_id == 9:
                    mat, G, F, A = _get_toy1()
                    drandom.seed(1)
                    out = [A.copy()]
                    for i in range(10):
                        out.append(solution.mhsw_trans(out[-1].copy(), G, F, 1))
                    res = serialize_matrix(out)
                elif part_id == 10:
                    mat, G, F, A = _get_toy2()
                    drandom.seed(2)
                    out = [A.copy()]
                    for i in range(20):
                        out.append(solution.mhsw_trans(out[-1].copy(), G, F, 1))
                    res = serialize_matrix(out)
                elif part_id == 11:
                    mat, G, F, A = _get_toy1(m2=True)
                    drandom.seed(1)
                    out = [A.copy()]
                    for i in range(20):
                        out.append(solution.mhsw_trans(out[-1].copy(), G, F, 2))
                    res = serialize_matrix(out)
                elif part_id == 12:
                    mat, G, F, A = _get_toy2(m2=True)
                    drandom.seed(2)
                    out = [A.copy()]
                    for i in range(20):
                        out.append(solution.mhsw_trans(out[-1].copy(), G, F, 2))
                    res = serialize_matrix(out)
                elif part_id == 13:
                    drandom.seed(1)
                    mat, G, F, A = _get_toy1()
                    M, all_samples = solution.mcmc_inference(G, F, None, "MHUniform", 0, 500, 1, A.copy())
                    for f in M: f.vars = [v+1 for v in f.vars]
                    res = serialize_factors_fg_grading(M)
                    
                    M, all_samples = solution.mcmc_inference(G, F, None, "MHSwendsenWang1", 0, 500, 1, A.copy())
                    for f in M: f.vars = [v+1 for v in f.vars]
                    res = res + '\n' + serialize_factors_fg_grading(M)
                    
                    M, all_samples = solution.mcmc_inference(G, F, None, "MHSwendsenWang2", 0, 500, 1, A.copy())
                    for f in M: f.vars = [v+1 for v in f.vars]
                    res = res + '\n' + serialize_factors_fg_grading(M)
                elif part_id == 14:
                    drandom.seed(2)
                    mat, G, F, A = _get_toy2()
                    M, all_samples = solution.mcmc_inference(G, F, None, "MHUniform", 0, 500, 1, A.copy())
                    for f in M: f.vars = [v+1 for v in f.vars]
                    res = serialize_factors_fg_grading(M)
                    
                    M, all_samples = solution.mcmc_inference(G, F, None, "MHSwendsenWang1", 0, 500, 1, A.copy())
                    for f in M: f.vars = [v+1 for v in f.vars]
                    res = res+ '\n' + serialize_factors_fg_grading(M)
                    
                    M, all_samples = solution.mcmc_inference(G, F, None, "MHSwendsenWang2", 0, 500, 1, A.copy())
                    for f in M: f.vars = [v+1 for v in f.vars]
                    res = res + '\n' + serialize_factors_fg_grading(M)
                else:
                    raise KeyError

                yield self.part_names[part_id], res
            except KeyError:
                yield self.part_names[part_id], 0
                

def serialize_matrix(matrix, add_one=True) -> str:
    res = ''
    for l in matrix:
        if add_one:
            l += 1
        res = res + '\n'
        for x in l:
            res += '%d ' % x
    return res
            
def serialize_factors_fg_grading(factors) -> str:
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
        for i, assignment in enumerate(itertools.product(*domains)):
            num_lines += 1
            val = f[tuple(reversed(assignment))]
            # if abs(val) <= 1e-40 or abs(val - 1) <= 1e-40 or np.isinf(val) or np.isnan(val):
            #     continue
            lines.append("%d %0.8g" % (i, val, ))
        lines[placeholder_idx] = "%d" % (num_lines, )
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
