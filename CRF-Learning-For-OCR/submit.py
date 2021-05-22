import sys

from scipy.io import loadmat

import helper
import solution as sol

sys.path.insert(0, '..')

import commons
from instance_neg_log_likelyhood import instance_neg_log_likelyhood


class Grader(commons.SubmissionBase):
    def __init__(self):
        part_names = [None, 
                      'QD9Fo', 'apl5y', 'n3klU', 'I45dY', 
                      'jrVA2', 'HIPL6', '0ctrF', 'cM7Gc', 
                      'zl35t', 'ZtLBP']
        super().__init__('CRF Learning for OCR', 'hdcdUQNgEealXw52htHS4Q', part_names)

    def __iter__(self):
        for part_id in range(1, len(self.part_names)):
            try:
                if part_id == 1:
                    X = loadmat("./data/Train1X.mat")['Train1X']
                    y = loadmat("./data/Train1Y.mat")['Train1Y']
                    res = helper.lr_train(X, y, 1)
                elif part_id == 2:
                    X = loadmat("./data/Train2X.mat")['Train2X']
                    y = loadmat("./data/Train2Y.mat")['Train2Y']
                    res = helper.lr_train(X, y, 1)
                elif part_id == 3:
                    x_train = loadmat('./data/Train1X.mat')['Train1X']
                    y_train = loadmat('./data/Train1Y.mat')['Train1Y'].squeeze()
                    x_validation = loadmat('./data/Validation1X.mat')['Validation1X']
                    y_validation = loadmat('./data/Validation1Y.mat')['Validation1Y'].squeeze()
                    lambdas = [2, 8]
                    res = sol.lr_search_lambda_sgd(x_train, y_train, x_validation, y_validation, lambdas)
                elif part_id == 4:
                    x_train = loadmat('./data/Train2X.mat')['Train2X']
                    y_train = loadmat('./data/Train2Y.mat')['Train2Y'].squeeze()
                    x_validation = loadmat('./data/Validation2X.mat')['Validation2X']
                    y_validation = loadmat('./data/Validation2Y.mat')['Validation2Y'].squeeze()
                    lambdas = [2, 8]
                    res = sol.lr_search_lambda_sgd(x_train, y_train, x_validation, y_validation, lambdas)
                elif part_id == 5:
                    mat = loadmat("./data/Part2Sample.mat", simplify_cells=True)
                    tree = helper.from_mat_to_tree(mat['sampleUncalibratedTree'])
                    _, logz = sol.clique_tree_calibrate(tree, True)
                    res = logz
                elif part_id == 6:
                    mat = loadmat("./data/Part2LogZTest.mat", simplify_cells=True)
                    tree = helper.from_mat_to_tree(mat['logZTestCliqueTree'])
                    _, logz = sol.clique_tree_calibrate(tree, True)
                    res = logz
                elif part_id == 7 or part_id == 9:
                    mat = loadmat("./data/Part2Sample.mat", simplify_cells=True)
                    sample_params = mat['sampleModelParams']
                    model_param = {'num_hidden_states': sample_params['numHiddenStates'],
                                   'num_observed_states': sample_params['numObservedStates'],
                                   'lambda': sample_params['lambda']}
                    nll, grad = instance_neg_log_likelyhood(mat['sampleX']-1, mat['sampleY']-1, mat['sampleTheta'],
                                                            model_param)
                    res = nll if part_id == 7 else grad
                elif part_id == 8 or part_id == 10:
                    mat = loadmat("./data/Part2Test.mat", simplify_cells=True)
                    model_param = mat['testModelParams']
                    model_param = {'num_hidden_states': model_param['numHiddenStates'],
                                   'num_observed_states': model_param['numObservedStates'],
                                   'lambda': model_param['lambda']}
                    nll, grad = instance_neg_log_likelyhood(mat['testX'] - 1, mat['testY'] - 1, mat['testTheta'],
                                                            model_param)
                    res = nll if part_id == 8 else grad
                else:
                    raise KeyError
                yield self.part_names[part_id], res
            except KeyError:
                yield self.part_names[part_id], 0


if __name__ == '__main__':
    grader = Grader()
    grader.grade()