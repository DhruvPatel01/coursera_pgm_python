import itertools
import sys

from scipy.io import loadmat
import numpy as np
import convert_mats

import helper
import sol

sys.path.insert(0, '..')

import commons
from commons.factor import Factor


class Grader(commons.SubmissionBase):
    def __init__(self):
        part_names = [None, 'Ga9CX', 'Y6ud3', 'YX6FP', 
                      'sVpuc', 'ZzAEz', 'jF5vU', 'IQZRx', 
                      'bWL2q', 'TfTAH', '44rjP', 'eGTyV', 'VfH4h']
        super().__init__('Markov Networks for OCR', '1RFc-gNfEeapUhL5oS3IIQ', part_names)

    def __iter__(self):
        for part_id in range(1, len(self.part_names)):
            models = convert_mats.pa3_models()
            if part_id % 2 == 0:
                data = convert_mats.pa3_sample_cases(is_test=True)
            else:
                data = convert_mats.pa3_sample_cases(is_test=False)

            try:
                if part_id == 1 or part_id == 2:
                    inp = data['part1_sample_image_input']
                    F = sol.compute_single_factors(inp, models['image_model'])
                    F = [sort_factor(f) for f in F]
                    F = sorted(F, key=lambda f: tuple(f.vars))
                    res = serialize_factors_fg_grading(F)
                elif part_id == 3 or part_id == 4:
                    inp = data['part2_sample_image_input']
                    F = sol.compute_pairwise_factors(inp, 
                                                     models['pairwise_model'],
                                                     models['image_model']['K'])
                    F = [sort_factor(f) for f in F]
                    F = sorted(F, key=lambda f: tuple(f.vars))
                    res = serialize_factors_fg_grading(F)
                elif part_id == 5 or part_id == 6:
                    print("NOTE: compute_triplet_factors(15 marks) will not be submitted, as there is")
                    print("a known bug at Coursera for this assignment.")
#                     inp = data['part3_sample_image_input']
#                     F = sol.compute_triplet_factors(inp, models['triplet_list'], models['image_model']['K'])
#                     F = [sort_factor(f) for f in F]
#                     res = serialize_factors_fg_grading(F, 2)
                    res = '0'
                elif part_id == 7:
                    inp = data['part4_sample_image_input']
                    F = sol.compute_similarity_factor(inp, models['image_model']['K'], 0, 1)
                    F = sort_factor(F)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 8:
                    inp = data['part4_sample_image_input']
                    F = sol.compute_similarity_factor(inp, models['image_model']['K'], 2, 3)
                    F = sort_factor(F)
                    res = serialize_factors_fg_grading([F])
                elif part_id == 9 or part_id == 10:
                    inp = data['part5_sample_image_input']
                    F = sol.compute_all_similarity_factors(inp, models['image_model']['K'])
                    F = [sort_factor(f) for f in F]
                    F = sorted(F, key=lambda f: tuple(f.vars))
                    res = serialize_factors_fg_grading(F)
                elif part_id == 11 or part_id == 12:
                    inp = data['part6_sample_factors_input']
                    F = sol.choose_top_similarity_factors(inp, 2)
                    F = [sort_factor(f) for f in F]
                    F = sorted(F, key=lambda f: tuple(f.vars))
                    res = serialize_factors_fg_grading(F)
                yield self.part_names[part_id], res
            except KeyError:
                raise
                
                
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
            if val != 1:
                new_lines.append("%0.8g" % (val, ))
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


if __name__ == '__main__':
    grader = Grader()
    grader.grade()