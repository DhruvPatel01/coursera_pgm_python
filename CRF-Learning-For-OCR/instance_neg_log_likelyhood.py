import sys

import numpy as np

import helper
import solution as sol

sys.path.insert(0, '..')
from commons.factor import Factor
import generate_all_features


def instance_neg_log_likelyhood(X, y, theta, model_params):
    """
    Args:
        X: Data, (numCharacters, numImageFeatures matrix) shaped numpy array.
            X[:,0] is all ones, i.e., it encodes the intercept/bias term.
        y: Data labels. (numCharacters, ) shaped numpy array
        theta: CRF weights/parameters. (numParams, ) shaped np array.
               These are shared among the various singleton / pairwise features.
        modelParams:  A dict with three keys:
                num_hidden_states: in our case, set to 26 (26 possible characters)
                .num_observed_states: in our case, set to 2  (each pixel is either on or off)
                .lambda: the regularization parameter lambda
                
    Returns:
        nll: Negative log-likelihood of the data.    (scalar)
        grad: Gradient of nll with respect to theta   (numParams, ) shaped np array
        
    Copyright (C) Daphne Koller, Stanford Univerity, 2012
    """
    
    feature_set = generate_all_features.generate_all_features(X, model_params)
    """
    feature_set is a dict with two keys:
    num_params - the number of parameters in the CRF (this is not numImageFeatures
                 nor numFeatures, because of parameter sharing)
    features   - a list comprising the features in the CRF.

    Each feature is a binary indicator variable, represented by a named tuple with three fields.
    .var           - a tuple containing the variables in the scope of this feature
    .assignment    - the assignment(tuple) that this indicator variable corresponds to
    .param_idx     - the index in theta that this feature corresponds to

    For example, if we have:
    feature = Feature(var=(0, 1), assignment=(4, 5), param_idx=8)

    then feature is an indicator function over X_0 and X_1, 
    which takes on a value of 1 if X_0 = 5 and X_1 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    Its contribution to the log-likelihood would be theta[8] if it's 1, and 0 otherwise.

    If you're interested in the implementation details of CRFs, 
    feel free to read through generate_all_features.py and the functions it calls!
    For the purposes of this assignment, though, you don't
    have to understand how this code works. (It's complicated.)
    """
    

    length = len(y)
    K = model_params['num_hidden_states']
    nll = 0.
    grad = np.zeros_like(theta)
    
    # Use the feature_set to calculate nll and grad.
    # This is the main part of the assignment, and it is very tricky - be careful!
    # You might want to code up your own numerical gradient checker to make sure
    # your answers are correct.

    # Hint: you can use `helper.clique_tree_calibrate` to calculate logZ effectively. 
    # We have halfway-modified clique_tree_calibrate; 
    # complete our implementation if you want to use it to compute logZ.

    # Solution Start
        
    # Solution End

    return nll, grad
