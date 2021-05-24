
import numpy as np
from scipy.stats import norm

import helper


def fit_gaussian_parameters(x):
    """
    Args:
        x: (N, ) shaped numpy array
        
    Returns:
        (mu, std)
    """
    mu = 0
    sigma = 1

    # Solution Start
    
    # Solution End

    return mu, np.sqrt(sigma)


def fit_linear_gaussian_parameters(x, u):
    """Estimate parameters of the linear Gaussian model:

    X|U ~ N(Beta(0)*U(0) + ... + Beta(n-1)*U(n-1) + Beta(n), sigma^2);


    X: (M, ), the child variable, M examples
    U: (M x N), N parent variables, M examples

    Copyright (C) Daphne Koller, Stanford Univerity, 2012
    """
    M = len(x)
    N = u.shape[1]

    beta = np.zeros(N+1)
    sigma = 1

    # collect expectations and solve the linear system
    # A = [ E[U(1)],      E[U(2)],      ... , E[U(n)],      1     ;
    #       E[U(1)*U(1)], E[U(2)*U(1)], ... , E[U(n)*U(1)], E[U(1)];
    #       ...         , ...         , ... , ...         , ...   ;
    #       E[U(1)*U(n)], E[U(2)*U(n)], ... , E[U(n)*U(n)], E[U(n)] ]

    # construct A

    # Solution Start

    # Solution End

    
    
    # B = [ E[X]; E[X*U(1)]; ... ; E[X*U(n)] ]
    # construct B
    # Solution Start
    
    # Solution End

    # solve A*Beta = B
    # Solution Start
    
    # Solution End

    # then compute sigma according to eq. (11) in PA description
    # Solution Start
    
    # Solution End

    return beta, sigma


def compute_log_likelihood(P, G, dataset):
    """
    returns the (natural) log-likelihood of data given the model and graph structure

    Args:
        P: dict of parameters (explained in PA description)
        G: graph structure and parameterization (explained in PA description)

           NOTICE that G could be either (10, 2) (same graph shared by all classes)
           or (10, 2, 2) (each class has its own graph). Your code should compute
           the log-likelihood using the right graph.

        dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)

    Returns:
        log_likelihood: log-likelihood of the data (scalar)

    Copyright (C) Daphne Koller, Stanford Univerity, 2012
    """
    log_likelyhood = 0
    N = dataset.shape[0]

    # You should compute the log likelihood of data as in eq. (12) and (13)
    # in the PA description
    # Hint: Use scipy.stats.norm.logpdf instead of log(normpdf) to prevent underflow.
    #       You may use log(sum(exp(logProb))) to do addition in the original
    #       space, sum(Prob).
    # Solution Start

    
    # Solution End
    return log_likelyhood


def learn_cpd_given_graph(dataset, G, labels):
    """
    Args:
        dataset: (N, 10, 3), N poses represented by 10 parts in (y, x, alpha)
        G: graph parameterization as explained in PA description.
        labels: (N, 2) true class labels for the examples. labels[i,j]=1 if the
            the ith example belongs to class j and 0 elsewhere

    Returns:
        P: dict (explained in PA description, and in README)
        loglikelihood: log-likelihood of the data (scalar)

    Copyright (C) Daphne Koller, Stanford Univerity, 2012
    """

    N = dataset.shape[0]
    K = labels.shape[1]
    log_likelyhood = 0
    P = {'c': np.zeros(K)}

    # estimate parameters
    # fill in P['c'], MLE for class probabilities
    # fill in P['clg'] for each body part and each class
    # choose the right parameterization based on G[i,0]
    # compute the likelihood - you may want to use compute_log_likelyhood
    # you just implemented.

    # Solution Start

    # Solution End

    # Following dummy line is added so that submit.py works even without implementing this
    # function. Kindly comment/remove it once solution is implemented.
    P['clg'] = [{'sigma_x': np.array([]), 'sigma_y': np.array([]), 'sigma_angle': np.array([])}]

    return P, log_likelyhood


def classify_dataset(dataset, labels, P, G):
    """returns the accuracy of the model P and graph G on the dataset

    Args:
        dataset: N x 10 x 3, N test instances represented by 10 parts
        labels:  N x 2 true class labels for the instances.
                 labels(i,j)=1 if the ith instance belongs to class j
        P: struct array model parameters (explained in PA description, and in README)
        G: graph structure and parameterization (explained in PA description)

    Returns:
        accuracy: fraction of correctly classified instances (scalar)

    Copyright (C) Daphne Koller, Stanford Univerity, 2012
    """
    accuracy = 0.
    N = dataset.shape[0]
    K = labels.shape[1]

    # Solution Start

    # Solution End

    return accuracy


def learn_graph_structure(dataset):
    """
    Args:
        dataset: a (N, 10, 3) numpy array.
    
    Returns:
        A: maximum spanning tree computed from the weight matrix W
        W: 10 x 10 weight matrix, where W(i,j) is the mutual information between
           node i and j. 
    """
    
    N = dataset.shape[0]
    W = np.zeros((10, 10))
    
    # Solution Start

    # Solution End
    
    return helper.maximum_spanning_tree(W), W


def learn_graph_and_cpds(dataset, labels):
    """
    Args:
        dataset: An (N, 10, 3) dim numpy array
        labels: (N, 2) array for class
        
    Returns:
        P: Learned parameters
        G: Learned graph structure
        likelyhood: likelyhood
    """
    N = len(dataset)
    K = labels.shape[1]
    
    G = np.zeros((10, 2, K), dtype=np.int64)
    G[1:, :, :] = 1
    
    # estimate graph structure for each class
    for k in range(K):
        pass
        # fill in G[:,:,k]
        # use helper.convert_A2G to convert a maximum spanning tree to a graph G
        # Solution Start

        # Solution End
        
    log_likelyhood = 0
    P = {'c': np.zeros(K)}
    
    P['clg'] = []
    for i in range(10):
        create_mu = False
        create_theta = False
        for k in range(K):
            if G[i, 0, k] == 0:
                create_mu = True
            else:
                create_theta = True
        d = {'sigma_y': np.zeros(K),
             'sigma_x': np.zeros(K),
             'sigma_angle': np.zeros(K)}
        if create_mu:
            d['mu_y'], d['mu_x'] = np.zeros(K), np.zeros(K)
            d['mu_angle'] = np.zeros(K)
        else:
            d['mu_x'] = d['mu_y'] = d['mu_angle'] = np.array([])
        
        if create_theta:
            d['theta'] = np.zeros((K, 12))
        else:
            d['theta'] = np.array([])
        P['clg'].append(d)
    
    # Solution Start

    # Solution End        
    return P, G, log_likelyhood