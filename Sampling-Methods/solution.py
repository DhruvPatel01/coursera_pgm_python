import sys
from functools import reduce

import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import connected_components
import tqdm

import drandom as rand
import helper

sys.path.insert(0, '../')
from commons import factor
import commons


def block_log_distribution(V, G, F, A):
    """
    log_bs = block_log_distribution(V, G, F, A) returns the log of a
    block-sampling array (which contains the log-unnormalized-probabilities of
    selecting each label for the block), given variables V to block-sample in
    network G with factors F and current assignment A.  Note that the variables
    in V must all have the same dimensionality.

    Args:
        V: a list of variable names
        G: the graph dict with the following keys:
         names - a list where names[i] = name of variable i in the graph
         card - an array where card[i] is the cardinality of variable i
         adj_list - an adjacency list of the graph. adj_list[i] = set of neighbours
          of i variable
         var2factors - an array of arrays where var2factors[i] gives
            an array where the entries are the indices of the factors
            including variable i
        F: An array of Factor objects. Read Factor documentation for details.
        A: an array with 1 entry for each variable in G s.t. A[i] is the current
           assignment to variable i in G.

    Each entry in log_bs is the log-probability that that value is selected.
    log_bs is the P(V | X_{-v} = A_{-v}, all X_i in V have the same value), where
    X_{-v} is the set of variables not in V and A_{-v} is the corresponding
    assignment to these variables consistent with A.  In the case that |V| = 1,
    this reduces to Gibbs Sampling.  NOTE that exp(log_bs) is not normalized to
    sum to one at the end of this function (nor do you need to worry about that
    in this function).

    Copyright (C) Daphne Koller, Stanford University, 2012
    """
    A = A.copy()  # just to be safe!

    if len(set(G['card'][V])) != 1:
        raise ValueError("Cardinality of the selected block is ill defined.")

    d = G['card'][V[0]]
    log_bs = np.zeros(d)

    """
    YOUR CODE HERE
    Compute log_bs by multiplying (adding in log-space) in the correct values from
    each factor that includes some variable in V.  

    NOTE: As this is called in the innermost loop of both Gibbs and 
    Metropolis-Hastings, you should make this fast.  You may want to make use of
    G['var2factors'].

    Also you should have only ONE for-loop for main computation. You can have
    one additional for loop at the end to convert Factor into required format.
    """
    # Solution Start

    # Solution End

    log_bs -= log_bs.min()
    return log_bs


def gibbs_trans(A, G, F):
    """
    Args:
        A: current assignment, change this inplace
        G: graph structure
        F: An array of factors

    Returns:
        A: next assignment
    """

    for i in range(len(G['names'])):
        """
        YOUR CODE HERE
        
        For each variable in the network sample a new value for it given everything
        else consistent with A.  Then update A with this new value for the
        variable.  NOTE: Your code should call BlockLogDistribution().
        
        IMPORTANT: you should call the function rand.randsample() exactly once
        here, and it should be the only random function you call.
        
        Also, note that rand.randsample() requires arguments in raw probability space
        be sure that the arguments you pass to it meet that criteria
        """
        # Solution Start

        # Solution End

    return A


def mh_gibbs(A, G, F):
    A_prop = gibbs_trans(A, G, F)
    p_acceptance = 1.0

    if rand.rand() < p_acceptance:
        return A_prop
    else:
        return A


def mh_uniform_trans(A, G, F):
    A_prop = (rand.rand(1, len(A)) * G['card']).astype(int)[0]
    p_acceptance = 0.0
    
    # Solution Start

    # Solution End
    
    if rand.rand() < p_acceptance:
        return A_prop
    else:
        return A
    
    
def mhsw_trans(A, G, F, variant=1):    
    q_list = G['q_list']
    q_keep_index = A[G['q_list'][:, 0].astype(int)] == A[G['q_list'][:, 1].astype(int)]
    q_list = q_list[q_keep_index]
    selected_edges_q_list_indx = q_list[:, 2] > rand.rand(1, len(q_list))[0]
    selected_edges = q_list[selected_edges_q_list_indx, :2].astype(int)
    csgraph = dok_matrix((len(G['card']), len(G['card'])), dtype=np.int64)
    for u, v in selected_edges:
        csgraph[u, v] = 1
        csgraph[v, u] = 1
    csgraph = csgraph.tocsr()
    n_comp, labels = connected_components(csgraph, directed=False)
    x = rand.rand()
    selected_cc = int(x*n_comp)
    selected_vars = np.nonzero(labels == selected_cc)[0]
    
    old_value = A[selected_vars[0]]
    d = G['card'][selected_vars[0]]
    log_R = np.zeros(d)
    if variant == 1:
        pass
        # YOUR CODE HERE
        # Specify the log of the distribution (log_R) from 
        # which a new label for Y is selected for variant 1
        
        
        ####################################################
    elif variant == 2:
        pass
        # YOUR CODE HERE
        # Specify the log of the distribution (log_R) from 
        # which a new label for Y is selected for variant 2
        #
        # We suggest you read through the preceding code
        # before implementing this, one of the generated
        # data structures may be useful in implementing this section
        
        ####################################################
    new_value = rand.randsample(d, 1, np.exp(log_R))
    A_prop = A.copy()
    A_prop[selected_vars] = new_value
    
    q_list_uv = G['q_list'][:, :2].astype(int)
    q_list_w = G['q_list'][:, 2]
    log_QY_ratio = 0.0
    for k in range(len(q_list_uv)):
        u, v = q_list_uv[k]
        q_ij = q_list_w[k]
        if len(np.intersect1d([u, v], selected_vars)) == 1:
            if A[u] == old_value and A[v] == old_value:
                log_QY_ratio -= np.log(1 - q_ij)
            if A_prop[u] == new_value and A_prop[v] == new_value:
                log_QY_ratio += np.log(1 - q_ij)
    
    p_acceptance = 1.
    # YOUR CODE HERE
    # Compute acceptance probability
    #
    # Read through the preceding code to understand
    # how to find the previous and proposed assignments
    # of variables, as well as some ratios used in computing
    # the acceptance probabilitiy.
    
    
    #################################

    if rand.rand() < p_acceptance:
        return A_prop
    else:
        return A


def mcmc_inference(G, F, E, trans_name, mix_time, num_samples, sampling_interval, A0):
    """
    Performs inference given a Markov Net or Bayes Net, G, a list
    of factors F, evidence E, and a list of parameters specifying the type of MCMC to be conducted.

    Copyright (C) Daphne Koller, Stanford University, 2012

    Args:
        G: graph dict
        F: list of factors
        E: Evidence
        trans_name: Name of the transition e.g. "Gibbs"
        mix_time:  is the number of iterations to wait until samples are collected.
            The user should determine mix_time by observing behavior
             using the visualization framework provided.
        num_samples: is the number of additional samples (after the initial sample
            following mixing) to collect
        sampling_interval: is the number of iterations in the chain to wait
            between collecting samples (after mix_time has been reached).
            This should ALWAYS be set to 1, unless memory usage is a concern
            in which case you may want to ignore some samples.
        A0: is the initial state of the Markov Chain.  Note that it is a joint assignment to the
            variables in G, where element is the value of the variable corresponding to the index.

    Returns:
        M: Samples after taking mix_time, sampling_interval into account
        all_samples: All the samples generated
    """
    if E:
        evidence_dict = {i: e for i, e in enumerate(E) if 0 <= e < 255}
        F = [f.evidence(evidence_dict) for f in F]
        
    def get_sw_variant(var):
        def trans(A, G, F):
            return mhsw_trans(A, G, F, var)
        return trans

    trans_fn = {
        'Gibbs': gibbs_trans,
        'MHGibbs': mh_gibbs,
        'MHUniform': mh_uniform_trans,
        'MHSwendsenWang1': get_sw_variant(1),
        'MHSwendsenWang2': get_sw_variant(2),
    }[trans_name]
    
    if 'SwendsenWang' in trans_name:
        E2F = {}
        for f in F:
            for i, u in enumerate(f.vars):
                for v in f.vars[i+1:]:
                    E2F[u, v] = f
                    E2F[v, u] = f
        vs, us = np.nonzero(G['edges'])
        q_list = []
        for u, v in zip(us, vs):
            if u <= v:
                continue
            edge_factor = E2F[u, v]
            
            q_ij = 0.0
            if trans_name == 'MHSwendsenWang1':
                pass
                # YOUR CODE HERE 
                # Specify the q_{i,j}'s for Swendsen-Wang for variant 1
            elif trans_name == 'MHSwendsenWang2':
                pass
                # YOUR CODE HERE 
                # Specify the q_{i,j}'s for Swendsen-Wang for variant 1
                
            assert 0 <= q_ij <= 1.0
            q_list.append([u, v, q_ij])
        G['q_list'] = np.array(q_list)
                    

    A = A0
    max_iter = mix_time + num_samples * sampling_interval
    all_samples = np.zeros((max_iter + 1, len(A)), dtype=np.int64)
    all_samples[0, :] = A0

    for i in tqdm.trange(max_iter, desc="Running the markov chain."):
        """
        YOUR CODE HERE
        Transition A to the next state in the Markov Chain and store the new sample in all_samples
        """
        # Solution Start
        
        # Solution End
        all_samples[i+1, :] = A

    # Don't change following lines
    collected_samples = all_samples[mix_time::sampling_interval, :]
    M = helper.extract_marginals_from_samples(G, collected_samples)

    return M, all_samples
