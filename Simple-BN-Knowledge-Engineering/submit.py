import itertools
import sys

from scipy.io import loadmat
import numpy as np

import helper

sys.path.insert(0, '..')

import commons
from commons.factor import Factor
import sol


class Grader(commons.SubmissionBase):
    def __init__(self):
        part_names = [None,
                      'COq9M','1AayO','bsUeZ','Y0uy4',
                      '5nfbi','VNVAv','c8RCS','rFh8a',
                      'ThtGy','lM96X','gde9N']
        super().__init__('Simple BN Knowledge Engineering', 'jk3STQNfEeadkApJXdJa6Q', part_names)

    def __iter__(self):
        d = loadmat('./data/submit_input.mat', simplify_cells=True)
        
        for part_id in range(1, len(self.part_names)):
            try:
                if part_id == 1:
                    net = convert_network('./Credit_net.net')
                    res = serialize_factors(net)
                elif part_id == 2:
                    A, B = (helper.from_matlab(f) for f in d['PART2']['SAMPLEINPUT'])
                    C = sol.factor_product(A, B)
                    res = serialize_factors([C])
                elif part_id == 3:
                    A, B = (helper.from_matlab(f) for f in d['PART2']['INPUT1'])
                    C, D = (helper.from_matlab(f) for f in d['PART2']['INPUT2'])
                    F = [sol.factor_product(A, B), sol.factor_product(C, D)]
                    res = serialize_factors(F)
                elif part_id == 4:
                    A = helper.from_matlab(d['PART3']['SAMPLEINPUT'][0]) 
                    V = set([1])
                    C = sol.factor_marginalization(A, V)
                    res = serialize_factors([C])
                elif part_id == 5:
                    A, V1 = helper.from_matlab(d['PART3']['INPUT1'][0]), [d['PART3']['INPUT1'][1]-1]
                    B, V2 = helper.from_matlab(d['PART3']['INPUT2'][0]), [d['PART3']['INPUT2'][1]-1]
                    F = [sol.factor_marginalization(A, V1), sol.factor_marginalization(B, V2)]
                    res = serialize_factors(F)
                elif part_id == 6:
                    Fs = d['PART4']['SAMPLEINPUT'][0]
                    Fs = [helper.from_mat_struct(s) for s in Fs]
                    E = dict(d['PART4']['SAMPLEINPUT'][1] - 1 )
                    O = sol.observe_evidence(Fs, E)
                    res = serialize_factors(O)
                elif part_id == 7:
                    Fs = d['PART4']['INPUT1'][0]
                    Fs = [helper.from_mat_struct(s) for s in Fs]
                    E = dict(d['PART4']['INPUT1'][1] - 1 )
                    O = sol.observe_evidence(Fs, E)
                    res = serialize_factors(O)
                elif part_id == 8:
                    Fs = [helper.from_matlab(f) for f in d['PART5']['SAMPLEINPUT']]
                    J = sol.compute_joint_distribution(Fs)
                    res = serialize_factors([J])
                elif part_id == 9:
                    Fs = [helper.from_matlab(f) for f in d['PART5']['INPUT1']]
                    J = sol.compute_joint_distribution(Fs)
                    res = serialize_factors([J])
                elif part_id == 10:
                    V = set(d['PART6']['SAMPLEINPUT'][0]-1)
                    Fs = [helper.from_mat_struct(f) for f in d['PART6']['SAMPLEINPUT'][1]]
                    E = dict(d['PART6']['SAMPLEINPUT'][2][None]-1)
                    F = sol.compute_marginal(V, Fs, E)
                    res = serialize_factors([F])
                elif part_id == 11:
                    res = []
                    for i in ['INPUT1', 'INPUT2', 'INPUT3', 'INPUT4']:
                        V = d['PART6'][i][0]-1
                        if isinstance(V, int):
                            V = [V]
                        V = set(V)
                        Fs = [helper.from_mat_struct(f) for f in d['PART6'][i][1]]
                        E = d['PART6'][i][2]-1
                        if E.ndim == 2:
                            E = dict(E)
                        else:
                            E = {}
                        F = sol.compute_marginal(V, Fs, E)
                        res.append(F)
                    res = serialize_factors(res)
                else:
                    raise KeyError
                
                yield self.part_names[part_id], res
            except KeyError:
                yield self.part_names[part_id], 0
                
                
def serialize_factors(factors, skip=1) -> str:
    lines = ["%d\n" % len(factors)]

    for f in factors:
        var = [v+1 for v in f.vars]
        lines.append("%d" % (len(var), ))
        lines.append("  ".join(map(str, var)))
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
            new_lines.append("%d %0.8g" % (i, val, ))
        new_lines = new_lines[::skip]
        lines[placeholder_idx] = "%d" % (num_lines, )
        lines.extend(new_lines)
        lines.append("")

    return "\n".join(lines)


nodes = ['DebtIncomeRatio', 'Assets', 'CreditWorthiness', 'Income', 'PaymentHistory', 'FutureIncome', 'Reliability', 'Age']
var_idx = {k: i for i, k in enumerate(nodes)}

def convert_network(fname='./Credit_net.net'):
    var_states = {}
    factors = []
    tbl = str.maketrans("", "", "();")

    with open(fname) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            if 'node ' in line:
                node = line.split()[1]
                while True:
                    line = fin.readline().strip()
                    if 'states' in line:
                        states = line.split(' = ')[1]
                        lidx = states.index('(')
                        ridx = states.index(')')
                        states = [s[1:-1] for s in states[lidx+1:ridx].split()]
                    elif line == "}":
                        break
                var_states[node] = states
            elif 'potential ' in line:
                lidx = line.index('(')
                ridx = line.index(')')
                lst = line[lidx+1:ridx].split()
                bar_idx = lst.index("|")
                var_names = lst[:bar_idx]
                par_names = lst[bar_idx+1:]

                while True:
                    line = fin.readline().strip()
                    if 'data' in line:
                        lidx = line.index('(')
                        s = line[lidx:]
                        while ';' not in line:
                            line = fin.readline()
                            s += line
                        s = s.translate(tbl).split()
                        s = list(map(float, s))
                    elif line == '}':
                        break

                var = list(reversed(par_names + var_names))
                card = [len(var_states[v]) for v in var]
                F = Factor(var, card)
                rev_domains = reversed(F.domains)
                for assn, v in zip(itertools.product(*rev_domains), s):
                    F[tuple(reversed(assn))] = v
                factors.append(F)

    for factor in factors:
        try:
            factor.vars = [var_idx[v] for v in factor.vars]
        except KeyError:
            print("Unwanted variable found. Did you change variable names? ")
            raise
        
    return factors


if __name__ == '__main__':
    grader = Grader()
    grader.grade()