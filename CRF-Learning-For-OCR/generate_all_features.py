from collections import namedtuple

import numpy as np

Feature = namedtuple("Feature", 'var assignment param_idx')


def compute_conditioned_singleton_features(mat, model_params, param_idx_base=0):
    height, width = mat.shape
    n_hidden_states = model_params['num_hidden_states']
    n_observed_states = model_params['num_observed_states']

    features = []
    for hidden_state in range(n_hidden_states):
        for feature_num in range(width):
            for v in range(height):
                param_idx = np.ravel_multi_index([mat[v, feature_num], feature_num, hidden_state],
                                                 dims=[n_observed_states, width, n_hidden_states],
                                                 order='F')
                features.append(Feature(var=(v, ), assignment=(hidden_state, ), param_idx=param_idx_base+param_idx))

    return features


def compute_unconditioned_singleton_features(length, model_params, param_idx_base=0):
    features = []
    for state in range(model_params['num_hidden_states']):
        for v in range(length):
            features.append(Feature(var=(v, ), assignment=(state, ), param_idx=param_idx_base+state))
    return features


def compute_unconditioned_pair_features(length, model_params, param_idx_base=0):
    features = []
    if length < 2:
        return features
    K = model_params['num_hidden_states']
    for state1 in range(K):
        for state2 in range(K):
            param_idx = param_idx_base + np.ravel_multi_index([state2, state1], [K, K], order='F')
            for v in range(length-1):
                features.append(Feature(var=(v, v+1), assignment=(state1, state2), param_idx=param_idx))
    return features


def generate_all_features(mat, model_params):
    param_idx_base = 0

    all_features = []

    features = compute_conditioned_singleton_features(mat, model_params, param_idx_base)
    if features:
        all_features.extend(features)
        # we can not look into max(f.param_idx for f in features) as some combination might not have been observed
        # so I'm computing this using below formula.
        # In case there are more than one conditioned features, one will have to adjust it accordingly.
        # Code by Daphne Koller handles it differently, but I didn't understand it, so I used below formula.
        param_idx_base = mat.shape[1] * model_params['num_hidden_states'] * model_params['num_observed_states']

    for fn in [compute_unconditioned_singleton_features, compute_unconditioned_pair_features]:
        features = fn(mat.shape[0], model_params, param_idx_base)
        if features:
            param_idx_base = max(f.param_idx for f in features) + 1
            all_features.extend(features)

    return {'num_params': param_idx_base, 'features': all_features}




