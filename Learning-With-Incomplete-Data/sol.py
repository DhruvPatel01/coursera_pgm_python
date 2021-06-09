import sys

import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

import helper
sys.path.append('..')
from commons.factor import Factor

def em_cluster(pose_data, G, initial_class_prob, max_iter):
    """
    Args:
        pose_data: (N, 10, 3) array, where N is number of poses;
            pose_data[i,:,:] yields the (10, 3) matrix for pose i.
        G: graph parameterization as explained in PA8
        initial_class_prob: (N, K), initial allocation of the N poses to the K
          classes. initial_class_prob[i, j] is the probability that example i belongs
          to class j
        max_iter: max number of iterations to run EM
    
    Returns:
        (P, log_likelihood, class_prob):
            P: dict holding the learned parameters as described in previous Python PA
            log_likelihood: (#(iterations run), 1) vector of loglikelihoods stored for
                each iteration
            class_prob:  (N, K) conditional class probability of the N examples to the
                K classes in the final iteration. class_prob[i, j] is the probability that
                example i belongs to class j
            
    
    Copyright (C) Daphne Koller, Stanford Univerity, 2012
    """
    
    N = pose_data.shape[0]
    K = initial_class_prob.shape[1]
    
    class_prob = initial_class_prob.copy()
    log_likelihood = np.zeros(max_iter)
    P = {'c': np.zeros(K)}
    
    # Following four lines are added to make grader work.
    # Remove once you have your implementation
    clg = []
    for i in range(10):
        clg.append({'sigma_x': [], 'sigma_y': [], 'sigma_angle': []})
    P['clg'] = clg
    
    for iter in range(max_iter):
        pass
        # M-STEP to estimate parameters for Gaussians
        #
        # Fill in P['c'] with the estimates for prior class probabilities
        # Fill in P['clg'] for each body part and each class
        # Make sure to choose the right parameterization based on G[i, 0]
        #
        # Hint: This part should be similar to your work from PA8
        
        ################
        # Your Code Here
        ################
            
        # E-STEP to re-estimate class_prob using the new parameters
        # 
        # Update class_prob with the new conditional class probabilities.
        # Recall that class_prob[i, j] is the probability that example i belongs to
        # class j.
        # 
        # You should compute everything in log space, and only convert to
        # probability space at the end.
        # 
        # Tip: Consider scipy.stats.norm for log pdf computation.
        # 
        # Hint: You should use the scipy.special.logsumexp(already imported)
        # function here to do probability normalization in log space 
        # to avoid numerical issues.
        class_prob = np.zeros_like(class_prob)
        
        ################
        # Your Code Here
        ################
       
        # Compute log likelihood of dataset for this iteration
        print("EM Iteration %d: log likelihood %f" % (iter, log_likelihood[iter]))
        
        if iter > 0 and log_likelihood[iter] < log_likelihood[iter-1]:
            break

    log_likelihood = log_likelihood[:iter+1]
    return P, log_likelihood, class_prob


def em_hmm(action_data, pose_data, G, initial_class_prob, initial_pair_prob, max_iter):
    """
    Args:
      action_data: list holding the actions as described in the PA
      pose_data: (N,10,3) numpy array, where N is number of poses in all actions
      G: graph parameterization as explained in PA description
      initial_class_prob: (N, K) numpy array, initial allocation of the N poses to the K
        states. initial_class_prob[i,j] is the probability that example i belongs
        to state j.
        This is described in more detail in the PA.
      initial_pair_prob: (V, K^2) numpy array, where V is the total number of pose
        transitions in all HMM action models, and K is the number of states.
        This is described in more detail in the PA.
      max_iter: max number of iterations to run EM

    Returns:
      P: dict holding the learned parameters as described in the PA
      log_likelihood: #(iterations run) x 1 vector of loglikelihoods stored for
        each iteration
      class_prob: (N, K) numpy array of the conditional class probability of the N examples to the
        K states in the final iteration. class_prob[i,j] is the probability that
        example i belongs to state j. This is described in more detail in the PA.
      pair_prob: (V, K^2) numpy array, where V is the total number of pose transitions
        in all HMM action models, and K is the number of states. This is
        described in more detail in the PA.

    Copyright (C) Daphne Koller, Stanford Univerity, 2012
    """
    
    N = pose_data.shape[0]
    K = initial_class_prob.shape[1]
    L = len(action_data)
    V = initial_pair_prob.shape[0]
    
    
    
    first_pose_idxs = np.array([a['marg_ind'][0] for a in action_data])
    
    class_prob = initial_class_prob
    pair_prob = initial_pair_prob
    log_likelihood = np.zeros(max_iter)
    
    P = {'c': np.zeros(K)}
    
    # Following four lines are added to make grader work.
    # Remove once you have your implementation
    clg = []
    for i in range(10):
        clg.append({'sigma_x': [], 'sigma_y': [], 'sigma_angle': []})
    P['clg'] = clg
    
    for iter in range(max_iter):
        P['c'] = np.zeros(K)
        # M-STEP to estimate parameters for Gaussians
        # Fill in P['c'], the initial state prior probability 
        # (NOT the class probability as in PA8 and em_cluster)
        # Fill in P['clg'] for each body part and each class
        # Make sure to choose the right parameterization based on G[i, 1]
        # Hint: This part should be similar to your work from PA8 and em_cluster
        
        ################
        # Your Code Here
        ################
         
        # M-STEP to estimate parameters for transition matrix
        # Fill in P['transMatrix'], the transition matrix for states
        # P['transMatrix'][i,j] is the probability of transitioning from state i to state j
        
        # Add Dirichlet prior based on size of poseData to avoid 0 probabilities
        P['transMatrix'] = np.zeros((K,K)) + pair_prob.shape[0] * .05
    
        ################
        # Your Code Here
        ################
        
        # E-STEP preparation: compute the emission model factors
        # (emission probabilities) in log space for each 
        # of the poses in all actions = log( P(Pose | State) )
        # Hint: This part should be similar to (but NOT the same as) your code in em_cluster
        
        log_emission_prob = np.zeros((N,K))
        
        ################
        # Your Code Here
        ################
        
        # E-STEP to compute expected sufficient statistics
        # class_prob contains the conditional class probabilities 
        # for each pose in all actions
        # pair_prob contains the expected sufficient statistics 
        # for the transition CPDs (pairwise transition probabilities)
        # Also compute log likelihood of dataset for this iteration
        # You should do inference and compute everything in log space, 
        # only converting to probability space at the end
        # Hint: You should use the logsumexp function here to do 
        # probability normalization in log space to avoid numerical issues

        class_prob = np.zeros((N,K))
        pair_prob = np.zeros((V,K*K))
        log_likelihood[iter] = 0
        
        ################
        # Your Code Here
        ################
        
        print('EM iteration %d: log likelihood: %f' % (iter, log_likelihood[iter]))
        if iter > 0 and log_likelihood[iter] < log_likelihood[iter-1]:
            break
        
    log_likelihood = log_likelihood[:iter+1]    
    return P, log_likelihood, class_prob, pair_prob


def recognize_actions(dataset_train, dataset_test, G, max_iter, return_Ps=False):
    """
    Args:
        dataset_train: dataset for training models, see PA for details
        dataset_test: dataset for testing models, see PA for details
        G: graph parameterization as explained in PA decription
        max_iter: max number of iterations to run for EM
        return_Ps: (Not in the original assignment), If True, also return Ps.
        
    Returns:
        accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
        predicted_labels: (N,) shaped numpy array with the predicted labels for
            each of the instances in dataset_test, with N being the number of unknown test instances
            
    Copyright (C) Daphne Koller, Stanford Univerity, 2012
    """
    accuracy = 0.0
    pred = np.zeros(3, dtype=int) # replace this line with appropriate line
    
    # Train a model for each action
    # Note that all actions share the same graph parameterization and number of max iterations
    
    ################
    # Your Code Here
    ################    
    
    
    # Classify each of the instances in dataset_test
    # Compute and return the predicted labels and accuracy
    # Accuracy is defined as (#correctly classified examples / #total examples)
    # Note that all actions share the same graph parameterization

    ################
    # Your Code Here
    ################

    
    if return_Ps:
        return accuracy, pred, Ps
    else:
        return accuracy, pred