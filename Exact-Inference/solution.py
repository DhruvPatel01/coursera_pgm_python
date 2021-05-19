import itertools
import sys

import numpy as np

sys.path.insert(0, '..')
from commons.factor import Factor
import helper


def compute_initial_potentials(skeleton):
    """
    Args:
        skeleton: a dictionary with following keys.
            'nodes': a list of sets. Each set is a set of constituent variables. e.g. {1,2,3}
            'edges': a list of edges. A single element would look like ({1,2,3}, {2,3,4})
                which means there is an edge between node {1,2,3} and node {2,3,4}. If (a, b) is
                in the list, (b, a) will not be in the list.
            'factor_list': a list of initialized Factors.

    Returns:
        a dict with ['clique_list', 'edges'] keys.
        'clique_list': a list of factors associated with each clique
        'adj_list': adjacency list with integer nodes. adj_list[0] = {1,2}
            implies that there is an edges  clique_list[0]-clique_list[1]
            and clique_list[0]-clique_list[2]
    """
    n = len(skeleton['nodes'])
    var_domain = {}
    for factor in skeleton['factor_list']:
        for var, domain in zip(factor.vars, factor.domains):
            var_domain[var] = domain

    clique_list = []
    for clique in skeleton['nodes']:
        clique = sorted(clique)
        domains = [var_domain[v] for v in clique]
        clique_list.append(Factor(clique, domains, init=1))
    adj_list = {i: set() for i in range(n)}

    # Solution Start

    # Solution End

    return {'clique_list': clique_list, 'adj_list': adj_list}


def get_next_clique(clique_tree, msgs):
    """
    Args:
        clique_tree: a structure returned by `compute_initial_potentials`
        msgs: A dictionary of dictionary.
            If u has sent message to v, that msg will be msgs[v][u].

    Returns:
        a tuple (i, j) if i is ready to send the message to j. 
        If all the messages has been passed, return None.
        
        If more than one message is ready to be transmitted, return 
        the pair (i,j) that is numerically smallest. If you use an outer
        for loop over i and an inner for loop over j, breaking when you find a 
        ready pair of cliques, you will get the right answer.
    """
    adj = clique_tree['adj_list']
    
    # Solution Start

    # Solution End
    
    return None


def clique_tree_calibrate(clique_tree, is_max=0):
    # msgs[u] = {v: msg_from_v_to_u}
    msgs = {i: {} for i in range(len(clique_tree['clique_list']))}
    adj = clique_tree['adj_list']

    # Solution Start
    
    # Following is a dummy line to make the grader happy when this is unimplemented.
    # Delete it or create new list `calibrated_potentials`
    calibrated_potentials = [f for f in clique_tree['clique_list']]

    # Solution End

    return {'clique_list': calibrated_potentials, 'adj_list': adj}


def compute_exact_marginals_bp(factors, evidence=None, is_max=0):
    """
    this function takes a list of factors, evidence, and a flag is_max, 
    runs exact inference and returns the final marginals for the 
    variables in the network. If is_max is 1, then it runs exact MAP inference,
    otherwise exact inference (sum-prod).
    It returns a list of size equal to the number of variables in the 
    network where M[i] represents the factor for ith variable.
 
    Args:
        factors: list[Factor]
        evidence: dict[variable] -> observation
        is_max: use max product algorithm

    Returns:
        list of factors. Each factor should have only one variable.
    """

    marginals = []
    if evidence is None:
        evidence = {}

    # Solution Start
    
    # Solution End
    return marginals


def factor_max_marginalization(factor, variables=None):
    if not variables or not factor.vars:
        return factor

    new_vars = sorted(set(factor.vars) - set(variables))
    if not new_vars:
        raise ValueError("Resultant factor has empty scope.")
    new_map = [factor.vars.index(v) for v in new_vars]

    new_factor = Factor(new_vars, [factor.domains[i] for i in new_map], init=float('-inf'))

    # Solution Start
   
    # Solution End
    
    return new_factor


def max_decoding(marginals):
    """
    Finds the best assignment for each variable from the marginals passed in.
    Returns A such that A[i] returns the index of the best instantiation for variable i.

    For instance: Let's say we have two variables 0 and 1. 
    Marginals for 0 = [0.1, 0.3, 0.6]
    Marginals for 1 = [0.92, 0.08]
    max_decoding(marginals) == [2, 0]

    M is a list of factors, where each factor is only over one variable.
    """

    A = np.zeros(len(marginals), dtype=np.int32)

    # Solution Start
    
    # Solution End
    return A
