import sys

sys.path.insert(0, '..')
import helper
from commons.factor import Factor

# Note: Variable names start with 0, i.e. first node in the chain has name `0`
# Unlike, MATLAB assignment, use zero based indexing for both variables and values.
# i.e. charater `a` has index 0, and `z` has index 25.


def compute_single_factors(images, image_model):
    n = len(images['img'])
    factors = []  # fill this array with factors
    # Solution Start
   
    # Solution End

    return factors


def compute_equal_pairwise_factors(images, K):
    n = len(images['img'])
    factors = []  # fill this array with factors, first factor will have score [0, 1] etc.
    
    # Solution Start
  
    # Solution End

    return factors


def compute_pairwise_factors(images, pairwise_model, K):
    n = len(images['img'])
    factors = []
    if n < 2:
        return factors

    # Solution Start
    
        
    # Solution End

    return factors


def compute_triplet_factors(images, triplet_list, K):
    n = len(images['img'])
    factors = []
    if n < 3:
        return factors

    # Solution Start
    
    

    # Solution End

    return factors


def compute_similarity_factor(images, K, i, j):
    f = Factor([i, j], [K, K], init=1.)

    # Solution Start

    
        
    # Solution End

    return f


def compute_all_similarity_factors(images, K):
    n = len(images['img'])
    factors = []

    # Solution Start
    
    

    # Solution End

    return factors


def choose_top_similarity_factors(factors, F):
    if len(factors) <= F:
        return factors

    new_factors = []
    # Solution Start
    
    
    
    # Solution End

    return new_factors

