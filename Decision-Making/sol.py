import sys
import itertools

import numpy as np

import helper
sys.path.append('..')
from commons.factor import Factor


def simple_calc_expected_utility(I):
    """
    Given a fully instantiated influence diagram with a single utility node and decision node,
    calculate and return the expected utility.  Note - assumes that the decision rule for the 
    decision node is fully assigned.
    
    Args: 
        I: See README.md for details.
        
    Returns: A scaler, expected utility of I.
    """
    
    
    """
    Hints:
    
    You can use F.Z to get the sum of all values for the factor F.
    """
    EU = 0.0
    
    # Solution Start


    
    # Solution End
    
    return EU


def calculate_expected_utility_factor(I):
    """
    Args: 
        I: An influence diagram I with a single decision node and a single utility node.
            See README.md for details.
            
    Returns:
        A factor over the scope of the decision rule D from I that
           gives the conditional utility given each assignment for D.vars
    """
    
    DF = I['decision_factors'][0]
    EUF = Factor(DF.vars, DF.domains, init=0)
    
    # Solution Start


    # Solution End
    
    return EUF
    

def optimize_meu(I):
    """
    Args:
        I: see README.md for the details.
        
    Returns:
        MEU, opt_decision_rule:
            MEU: a scaler. Maximum Expected Utility
            decision_rule: A factor.
    """
    MEU = 0
    D = I['decision_factors'][0]
    opt_decision_rule = Factor(D.vars, D.domains, init=0)
    
    """
    Python related hints:
    - You can use itertools.product(domain1, domain2, domain3 ...) to enumerate though 
    the cartesian product of the domains.
    - You can index a factor using a dictionary of full assignment.
        F[{'a': 0, 'b': 1}] for factor with two variables 'a' and 'b'.
    """
    
    # Solution Start


        
    # Solution End
    
    return MEU, opt_decision_rule


def optimize_with_joint_utility(I):
    """
    Same signature as optimize_meu. Now len(I['utility_factors']) > 1.
    """
    
    """
    Tip: You should try to implement factor sum on your own, if you haven't so far.
        Eventhough, Factor implementation overloads `+` operator.
    """
    
    MEU = 0.0
    opt_decision_rule = Factor([], [])
    
    # Solution Start



    # Solution End
    
    return MEU, opt_decision_rule


def optimize_linear_expectations(I):
    """
    Same signature as optimize_meu. Now len(I['utility_factors']) > 1.
    """
        
    D = I['decision_factors'][0]
    MEU = 0.0
    opt_decision_rule = Factor(D.vars, D.domains, init=0)
    
    # Solution Start
    


    # Solution End
    
    return MEU, opt_decision_rule
    