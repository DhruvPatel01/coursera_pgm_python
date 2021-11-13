import sys

import numpy as np

import helper
sys.path.append('..')
from commons.factor import Factor


def variable_elimination(Fs, Zs):
    """
    Args:
        Fs: a list of Factors
        Zs: a list of variables to marginalize.
        
    Returns:
        A single factor, with Zs marginalized out.
    """
    
    F = Factor([], [])
    for f in Fs:
        F = F * f
        
    return F.marginalise(Zs)


def observe_evidence(Fs, E, normalize=False):
    """
    Args:
        Fs: List of Factors.
        E: Dictionary of evidence in the form {'var': observed_value, ...}.
        normalize: Should this function normalize the CPD after observing?
            Assumes that first variable (i.e. F.vars[0]) is the child and 
            all remaining ones (i.e. F.vars[1:]) are parent.
            
    Returns:
        A list of factors after observing the evidence. 
        If the intersection of F.vars and E is empty, factor is returned
        unchanged.
    """
    new_Fs = []
    for F in Fs:
        new_F = F.evidence(E)
        if normalize and new_F != F:
            new_F.conditional_normalize(F.vars[1:])
        new_Fs.append(new_F)
    return new_Fs