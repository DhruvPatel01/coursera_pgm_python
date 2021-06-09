import sys
import os

from scipy.io import loadmat
import numpy as np

import helper
import sol

sys.path.insert(0, '..')

import commons

def load_mat(part_id):
    if part_id%2 == 1:
        mat = loadmat('./data/PA9SampleCases.mat', simplify_cells=True)['exampleINPUT']
    else:
        mat = loadmat('./data/submit_input.mat', simplify_cells=True)['INPUT']
    return mat
        
class Grader(commons.SubmissionBase):
    def __init__(self):
        part_names = [None, 
                      'BwB8H', 'CWFts',
                      '0tRrx', 'EgIZ9',
                      'Ev05z', 'zPvH5',
                      'BGZlk']
        super().__init__('Learning with Incomplete Data', 'HRaFgotIEeaoKxJCmMZ6SQ', part_names)

    def __iter__(self):
        for part_id in range(1, len(self.part_names)):
            try:
                if part_id in (1, 2):
                    mat = load_mat(part_id)
                    G = mat['t1a2']
                    G[1:, 1] -= 1
                    P, log_likelihood, class_prob = sol.em_cluster(mat['t1a1'], G, mat['t1a3'], mat['t1a4'])
                    tmp = np.r_[P['c'], 
                                np.concatenate([clg['sigma_x'] for clg in P['clg']]),
                                np.concatenate([clg['sigma_y'] for clg in P['clg']]),
                                np.concatenate([clg['sigma_angle'] for clg in P['clg']]),
                                log_likelihood, np.ravel(class_prob, order='F')]
                    res = commons.sprintf("%.4f", tmp)
                elif part_id in (3, 4):
                    mat = load_mat(part_id)
                    action_data = mat['t2a1']
                    for a in action_data:
                        a['marg_ind'] -= 1
                        a['pair_ind'] -= 1
                    G = mat['t2a3']
                    G[1:, 1] -= 1
                    P, log_likelihood, class_prob, pair_prob = sol.em_hmm(action_data, mat['t2a2'], G, mat['t2a4'], mat['t2a5'], mat['t2a6'])
                    tmp = np.r_[P['c'], 
                                np.concatenate([clg['sigma_x'] for clg in P['clg']]),
                                np.concatenate([clg['sigma_y'] for clg in P['clg']]),
                                np.concatenate([clg['sigma_angle'] for clg in P['clg']]),
                                log_likelihood, 
                                np.ravel(class_prob, order='F'), 
                                np.ravel(pair_prob, order='F')]
                    res = commons.sprintf("%.4f", tmp)
                elif part_id in (5, 6):
                    mat = load_mat(part_id)
                    for actionData in mat['t3a1']:
                        for action in actionData['actionData']:
                            action['marg_ind'] -= 1
                            action['pair_ind'] -= 1

                    for action in mat['t3a2']['actionData']:
                        action['marg_ind'] -= 1
                        action['pair_ind'] -= 1
                    mat['t3a2']['labels'] -= 1
                    
                    G = mat['t3a3']
                    G[1:, 1] -= 1
                    acc, pred = sol.recognize_actions(mat['t3a1'], mat['t3a2'], G, mat['t3a4'])
                    tmp = np.r_[acc, pred+1]
                    res = commons.sprintf("%.4f", tmp)
                elif part_id == 7:
                    if not os.path.isfile('./Predictions.npy'):
                        print("Warning: Prediction.py is not generated. Not grading it.")
                        res = 0
                    else:
                        mat = np.load('./Predictions.npy')
                        if max(mat) < 3:
                            mat += 1
                        tmp = np.r_[len(mat), mat]
                        res = '\n'.join(map(str, tmp))
                else:
                    raise KeyError
                yield self.part_names[part_id], res
            except KeyError:
                yield self.part_names[part_id], 0


if __name__ == '__main__':
    grader = Grader()
    grader.grade()