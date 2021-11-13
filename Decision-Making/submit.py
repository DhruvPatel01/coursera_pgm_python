import sys
import itertools

from scipy.io import loadmat
import numpy as np

# import convert_mats
import helper
import sol

sys.path.insert(0, '..')
import commons
from commons.factor import Factor


def load_I(which='FullI'):
    mat = loadmat(f'./data/{which}.mat', simplify_cells=True)[which]
    I = {}
    
    for key in ['random', 'decision', 'utility']:
        Fs = mat[f'{key.title()}Factors']
        if not isinstance(Fs, list):
            Fs = [Fs]
        I[f'{key}_factors'] = [Factor.from_matlab(f) for f in Fs]
    return I


class Grader(commons.SubmissionBase):
    def __init__(self):
        part_names = [None, 'ABz0W', 'B5ynW', 
                      'LYITS', 'fJqCV', 'I0Bvr', 
                      '37kCm', 'D2EEw', '2AHBQ', 
                      'aFNuE', 'Th2m9']
        super().__init__('Decision Making', 'ZPEHfwNgEealXw52htHS4Q', part_names)

    def __iter__(self):
        for part_id in range(1, len(self.part_names)):
            try:
                if part_id == 1:
                    I = load_I('FullI')
                    F = sol.calculate_expected_utility_factor(I)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 2:
                    I = load_I('FullI')
                    I['random_factors'] = helper.observe_evidence(I['random_factors'], {2: 1}, True)
                    F = sol.calculate_expected_utility_factor(I)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 3:
                    I = load_I('FullI')
                    meu, optdr = sol.optimize_meu(I)
                    res = serialize_meu_optimization_fg(meu, optdr)
                elif part_id == 4:
                    I = load_I('FullI')
                    I['random_factors'] = helper.observe_evidence(I['random_factors'], {2: 1}, True)
                    meu, optdr = sol.optimize_meu(I)
                    res = serialize_meu_optimization_fg(meu, optdr)
                elif part_id == 5:
                    I = load_I('MultipleUtilityI')
                    meu, optdr = sol.optimize_with_joint_utility(I)
                    res = serialize_meu_optimization_fg(meu, optdr)
                elif part_id == 6:
                    I = load_I('MultipleUtilityI')
                    I['random_factors'] = helper.observe_evidence(I['random_factors'], {2: 0}, True)
                    meu, optdr = sol.optimize_with_joint_utility(I)
                    res = serialize_meu_optimization_fg(meu, optdr)
                elif part_id == 7:
                    I = load_I('MultipleUtilityI')
                    meu, optdr = sol.optimize_linear_expectations(I)
                    res = serialize_meu_optimization_fg(meu, optdr)
                elif part_id == 8:
                    I = load_I('MultipleUtilityI')
                    I['random_factors'] = helper.observe_evidence(I['random_factors'], {2: 0}, True)
                    meu, optdr = sol.optimize_linear_expectations(I)
                    res = serialize_meu_optimization_fg(meu, optdr)
                elif part_id == 9:
                    I = load_I('FullI')
                    res = "%.4f" % sol.simple_calc_expected_utility(I)
                elif part_id == 10:
                    I = load_I('FullI')
                    I['random_factors'] = helper.observe_evidence(I['random_factors'], {2: 1}, True)
                    res = "%.4f" % sol.simple_calc_expected_utility(I)
                else:
                    raise KeyError
                yield self.part_names[part_id], res
            except KeyError:
                yield self.part_names[part_id], 0
                
                
def serialize_factors_fg_grading(factors) -> str:
    lines = ["%d\n" % len(factors)]

    for f in factors:
        var = [v+1 for v in f.vars]
        lines.append("%d" % (len(var), ))
        lines.append("  ".join(map(str, var)))
        lines.append("  ".join(str(len(d)) for d in f.domains))
        lines.append(str(len(f.val)))

        # libDAI expects first variable to change fastest
        # but itertools.product changes the last element fastest
        # hence reversed list
        domains = reversed(f.domains)
        num_lines = 0
        new_lines = []
        for i, assignment in enumerate(itertools.product(*domains)):
            num_lines += 1
            val = f[tuple(reversed(assignment))]
            new_lines.append("%d %0.8g" % (i, val))
        lines.extend(new_lines)
        
    return "\n".join(lines)


def sort_factor(F):
    domains_d = dict(zip(F.vars, F.domains))
    var = sorted(F.vars)
    domains = [domains_d[v] for v in var]
    newF = Factor(var, domains)
    for k in F:
        assignment = dict(zip(F.vars, k))
        newF[assignment] = F[k]
    return newF


def serialize_meu_optimization_fg(meu, opt_dr):
    opt_dr = sort_factor(opt_dr)
    res = serialize_factors_fg_grading([opt_dr])
    return '%s\n\n%.4f\n' % (res, meu)


if __name__ == '__main__':
    grader = Grader()
    grader.grade()