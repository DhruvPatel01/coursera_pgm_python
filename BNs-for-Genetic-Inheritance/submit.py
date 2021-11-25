import itertools
import sys

from scipy.io import loadmat
import numpy as np
import helper
import sol

sys.path.insert(0, '..')

import commons
from commons.factor import Factor


class Grader(commons.SubmissionBase):
    def __init__(self):
        part_names = [None,
                      '6qJ5F', 'YW7hB', '4SCHR', 'dMfG8',
                      'sVijF', '7ccHc', 'YTHBY', 'mFn3B', 
                      'wCUvg', 'wyMoY', 'flSZ2', 'GfmuH', 
                      'JmkmK', 'J0R6g', 'dpFFP', 'nBdS8']
        super().__init__('BNs for Genetic Inheritance', 'haijvkP8EeaOwRI5GO98Xw', part_names)

    def __iter__(self):
        for part_id in range(1, len(self.part_names)):
            try:
                if part_id == 1:
                    F = sol.phenotype_given_genotype_mendelian_factor(1, 0, 2)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 2:
                    F = sol.phenotype_given_genotype_mendelian_factor(0, 0, 2)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 3:
                    F = sol.phenotype_given_genotype([.8, .6, .1], 0, 2)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 4:
                    F = sol.phenotype_given_genotype([.2, .5, .9], 0, 2)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 5:
                    F = sol.genotype_given_allele_freqs_factor([.1, .9], 0)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 6:
                    F = sol.genotype_given_allele_freqs_factor([.98, .02], 0)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 7:
                    F = sol.genotype_given_parents_genotypes_factor(2, 2, 0, 1)
                    F = sort_factor_later_vars(F)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 8:
                    F = sol.genotype_given_parents_genotypes_factor(3, 2, 0, 1)
                    F = sort_factor_later_vars(F)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 9:
                    allele_freqs = [.1, .9]
                    alpha_list = [.8, .6, .1]
                    pedigree = {
                        'parents': [None, (0, 2), None, (0, 2), (1, 5), None, (1, 5), (3, 8), None],
                        'names': ['Ira','James','Robin','Eva','Jason','Rene','Benjamin','Sandra','Aaron']
                    }

                    cgn = sol.construct_genetic_network(pedigree, allele_freqs, alpha_list)
                    cgn = [sort_factor_later_vars(F) for F in cgn]
                    cgn = sort_struct(cgn)
                    res = serialize_factors_fg_grading(cgn)
                elif part_id == 10:
                    allele_freqs = [.1, .9]
                    alpha_list = [.8, .6, .1]
                    pedigree = {
                    'parents': [None, None, (1, 0), None, (1, 0), None, None, (2, 3), (4, 6), (4, 5)],
                    'names': ['Alan','Vivian','Alice','Larry','Beth','Henry','Leon','Frank','Amy', 'Martin']
                    }

                    cgn = sol.construct_genetic_network(pedigree, allele_freqs, alpha_list)
                    cgn = [sort_factor_later_vars(F) for F in cgn]
                    cgn = sort_struct(cgn)
                    res = serialize_factors_fg_grading(cgn)
                elif part_id == 11:
                    alpha_list = [0.8, 0.6, 0.1, 0.5, 0.05, 0.01]
                    F = sol.phenotype_given_copies_factor(alpha_list, 3, 0, 1, 2)
                    F = sort_factor_later_vars(F)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 12:
                    alpha_list = [0.001, 0.009, 0.3, 0.2, 0.75, 0.95]
                    F = sol.phenotype_given_copies_factor(alpha_list, 3, 0, 1, 2)
                    F = sort_factor_later_vars(F)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 13:
                    pedigree = {
                        'parents': [None, (0, 2), None, (0, 2), (1, 5), None, (1, 5), (3, 8), None],
                        'names': ['Ira','James','Robin','Eva','Jason','Rene','Benjamin','Sandra','Aaron']
                    }
                    alpha_list = [0.8, 0.6, 0.1, 0.5, 0.05, 0.01]
                    allele_freqs = [.1, .7, .2]

                    cgn = sol.construct_decoupled_genetic_network(pedigree, allele_freqs, alpha_list)
                    cgn = [sort_factor_later_vars(F) for F in cgn]
                    cgn = sort_struct(cgn)
                    res = serialize_factors_fg_grading(cgn)
                elif part_id == 14:
                    pedigree = {
                        'parents': [None, None, (1, 0), None, (1, 0), None, None, (2, 3), (4, 6), (4, 5)],
                        'names': ['Alan','Vivian','Alice','Larry','Beth','Henry','Leon','Frank','Amy', 'Martin']
                    }
                    alpha_list = [0.8, 0.6, 0.1, 0.5, 0.05, 0.01]
                    allele_freqs = [.1, .7, .2]

                    cgn = sol.construct_decoupled_genetic_network(pedigree, allele_freqs, alpha_list)
                    cgn = [sort_factor_later_vars(F) for F in cgn]
                    cgn = sort_struct(cgn)
                    res = serialize_factors_fg_grading(cgn)
                elif part_id == 15:
                    allele_weights = [[3, -3], [0.9, -0.8]]
                    phenotype_var = 2;
                    gene_copy_var_parent1_list = [0, 1]
                    gene_copy_var_parent2_list = [3, 4]
                    F = sol.construct_sigmoid_phenotype_factor(allele_weights, gene_copy_var_parent1_list, 
                                                               gene_copy_var_parent2_list, phenotype_var)
                    F = sort_factor_later_vars(F)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 16:
                    allele_weights = [[0.01, -.2], [1, -.5]]
                    phenotype_var = 2;
                    gene_copy_var_parent1_list = [0, 1]
                    gene_copy_var_parent2_list = [3, 4]
                    F = sol.construct_sigmoid_phenotype_factor(allele_weights, gene_copy_var_parent1_list, 
                                                               gene_copy_var_parent2_list, phenotype_var)
                    F = sort_factor_later_vars(F)
                    res = serialize_factors_fg_grading([F])
                else:
                    raise KeyError
                
                yield self.part_names[part_id], res
            except KeyError:
                yield self.part_names[part_id], 0
                
                
def serialize_factors_fg_grading(factors, skip=1) -> str:
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

def sort_factor(F):
    domains_d = dict(zip(F.vars, F.domains))
    var = sorted(F.vars)
    domains = [domains_d[v] for v in var]
    newF = Factor(var, domains)
    for k in F:
        assignment = dict(zip(F.vars, k))
        newF[assignment] = F[k]
    return newF

def sort_factor_later_vars(F):
    if not F.vars:
        return F
    domains_d = dict(zip(F.vars, F.domains))
    var = [F.vars[0]] + sorted(F.vars[1:])
    domains = [domains_d[v] for v in var]
    newF = Factor(var, domains)
    for k in F:
        assignment = dict(zip(F.vars, k))
        newF[assignment] = F[k]
    return newF

def sort_struct(S):
    def key(F):
        s = []
        s.extend([x+1 for x in F.vars])
        s.extend([len(x) for x in F.domains])
        domains = reversed(F.domains)
        s.extend([F[tuple(reversed(assignment))] for assignment in itertools.product(*domains)])
        
        fmt = itertools.cycle(['%d', '%d', '%f'])
        s = ''.join(f%x for f, x in zip(fmt, s))
        return s
    return sorted(S, key=key)


if __name__ == '__main__':
    grader = Grader()
    grader.grade()