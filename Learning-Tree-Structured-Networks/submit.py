import sys

from scipy.io import loadmat
import numpy as np

import helper
import solution as sol

sys.path.insert(0, '..')

import commons


class Grader(commons.SubmissionBase):
    def __init__(self):
        part_names = [None, 
                      'ZWTe6', 'MHKM2', 'ersuQ', 'bzqOa', 
                      '7vCzT', 'N3abb', 'YMrzB', 'loxgM', 
                      'izsvG', 'xWOCw', 'pBr7I', 'a7bhx', 
                      '22gxv', 'Bd2ZE']
        super().__init__('Learning Tree-structured Networks', 'ZtTFjgNhEeahzAr11F1cUw', part_names)

    def __iter__(self):
        for part_id in range(1, len(self.part_names)):
            try:
                if part_id == 1:
                    mat = loadmat('./data/PA8SampleCases.mat', simplify_cells=True)
                    mu, sigma = sol.fit_gaussian_parameters(mat['exampleINPUT']['t1a1'])
                    res = np.array([mu, sigma])
                elif part_id == 2:
                    mat = loadmat('./data/submit_input.mat', simplify_cells=True)
                    mu, sigma = sol.fit_gaussian_parameters(mat['INPUT']['t1a1'])
                    res = np.array([mu, sigma])
                elif part_id == 3:
                    mat = loadmat('./data/PA8SampleCases.mat', simplify_cells=True)
                    beta, sigma = sol.fit_linear_gaussian_parameters(mat['exampleINPUT']['t2a1'],
                                                                     mat['exampleINPUT']['t2a2'])
                    res = np.r_[beta, sigma]
                elif part_id == 4:
                    mat = loadmat('./data/submit_input.mat', simplify_cells=True)
                    beta, sigma = sol.fit_linear_gaussian_parameters(mat['INPUT']['t2a1'],
                                                                     mat['INPUT']['t2a2'])
                    res = np.r_[beta, sigma]
                elif part_id == 5:
                    mat = loadmat('./data/PA8SampleCases.mat', simplify_cells=True)
                    G = mat['exampleINPUT']['t3a2']
                    G[:, 1] -= 1
                    res = sol.compute_log_likelihood(mat['exampleINPUT']['t3a1'], G, mat['exampleINPUT']['t3a3'])
                elif part_id == 6:
                    mat = loadmat('./data/submit_input.mat', simplify_cells=True)
                    G = mat['INPUT']['t3a2']
                    G[:, 1] -= 1
                    res = sol.compute_log_likelihood(mat['INPUT']['t3a1'], G, mat['INPUT']['t3a3'])
                elif part_id == 7:
                    mat = loadmat('./data/PA8SampleCases.mat', simplify_cells=True)
                    G = mat['exampleINPUT']['t4a2']
                    G[:, 1] -= 1
                    P, L = sol.learn_cpd_given_graph(mat['exampleINPUT']['t4a1'], G, mat['exampleINPUT']['t4a3'])
                    res = np.r_[P['c'],
                                np.concatenate([clg['sigma_x'] for clg in P['clg']]),
                                np.concatenate([clg['sigma_y'] for clg in P['clg']]),
                                np.concatenate([clg['sigma_angle'] for clg in P['clg']]), L]

                elif part_id == 8:
                    mat = loadmat('./data/submit_input.mat', simplify_cells=True)
                    G = mat['INPUT']['t4a2']
                    G[:, 1] -= 1
                    P, L = sol.learn_cpd_given_graph(mat['INPUT']['t4a1'], G, mat['INPUT']['t4a3'])
                    res = np.r_[P['c'],
                                np.concatenate([clg['sigma_x'] for clg in P['clg']]),
                                np.concatenate([clg['sigma_y'] for clg in P['clg']]),
                                np.concatenate([clg['sigma_angle'] for clg in P['clg']]), L]
                elif part_id == 9:
                    mat = loadmat('./data/PA8SampleCases.mat', simplify_cells=True)
                    G = mat['exampleINPUT']['t5a4']
                    G[:, 1] -= 1
                    res = sol.classify_dataset(mat['exampleINPUT']['t5a1'], mat['exampleINPUT']['t5a2'],
                                               mat['exampleINPUT']['t5a3'], G)
                elif part_id == 10:
                    mat = loadmat('./data/submit_input.mat', simplify_cells=True)
                    G = mat['INPUT']['t5a4']
                    G[:, 1] -= 1 
                    res = sol.classify_dataset(mat['INPUT']['t5a1'], mat['INPUT']['t5a2'],
                                               mat['INPUT']['t5a3'], G)
                elif part_id == 11:
                    mat = loadmat('./data/PA8SampleCases.mat', simplify_cells=True)
                    A, W = sol.learn_graph_structure(mat['exampleINPUT']['t6a1'])
                    res = ' '.join(map(str, A.ravel(order='F')))
                elif part_id == 12:
                    mat = loadmat('./data/submit_input.mat', simplify_cells=True)
                    A, W = sol.learn_graph_structure(mat['INPUT']['t6a1'])
                    res = ' '.join(map(str, A.ravel(order='F')))
                elif part_id == 13 or part_id == 14:
                    if part_id == 13:
                        mat = loadmat('./data/PA8SampleCases.mat', simplify_cells=True)['exampleINPUT']
                    else:
                        mat = loadmat('./data/submit_input.mat', simplify_cells=True)['INPUT']
                    P, G, L = sol.learn_graph_and_cpds(mat['t7a1'], mat['t7a2'])
                    G[G[:, 0, 0]!=0, 1, 0] += 1
                    G[G[:, 0, 1]!=0, 1, 1] += 1
                    tmp = [P['c']]
                    tmp.extend([P['clg'][i]['sigma_x'] for i in range(10)])
                    tmp.extend([P['clg'][i]['sigma_y'] for i in range(10)])
                    tmp.extend([P['clg'][i]['sigma_angle'] for i in range(10)])
                    tmp.append(G[:, 0, 0]); tmp.append(G[:, 1, 0]);
                    tmp.append(G[:, 0, 1]); tmp.append(G[:, 1, 1]);
                    tmp.append(np.array([L]))
                    tmp = np.concatenate(tmp)
                    res = commons.sprintf("%.4f", tmp)
                else:
                    raise KeyError
                yield self.part_names[part_id], res
            except KeyError:
                yield self.part_names[part_id], 0


if __name__ == '__main__':
    grader = Grader()
    grader.grade()