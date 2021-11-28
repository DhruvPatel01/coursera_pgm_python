import itertools

class Factor:
    def __init__(self, var, card, init=0.0):
        self.card = card
        self.domains = [range(d) for d in card]
        self.vars = var
        self.val = {}
        
        for assignment in itertools.product(*self.domains):
            self[assignment] = init
            
    def __getitem__(self, assignment):
        if isinstance(assignment, (int, float)):
            assignment = (assignment, )
        elif isinstance(assignment, dict):
            assignment = tuple(assignment[v] for v in self.vars)
        return self.val[assignment]
    
    def __setitem__(self, assignment, value):
        if isinstance(assignment, (int, float)):
            assignment = (assignment, )
        elif isinstance(assignment, dict):
            assignment = tuple(assignment[v] for v in self.vars)
        self.val[assignment] = value
        
    def __repr__(self):
        return repr(self.val)
    
    
def factor_product(A: Factor, B: Factor):
    """Return a product of factor
    
    DO NOT DELETE ANYTHING BEFORE `# Solution Start`
    """
    
    if not A.vars:
        return B
    if not B.vars:
        return A
    
    C = Factor([], [])
    
    # Solution Start
    
    
    # Solution End
    
    return C


def factor_marginalization(A, V):
    """
    Return a new factor B with variables V marginalized out of A.
    
    Hint: You can index a factor with a dict. 
    This dict can contain variables that are not part of factor itself. 
        they will be ignored.
    """
    
    if not A.vars or not V:
        return A
    
    B_vars = sorted(set(A.vars) - set(V))
    
    A_card = dict(zip(A.vars, A.domains))
    B_card = [len(A_card[k]) for k in B_vars]
    
    B = Factor(B_vars, B_card, init=0.0)
    
    # Solution Start
    
    
    # Solution End
    
    return B


def observe_evidence(Fs, E):
    """
    For each factor F in Fs
        Overwrite entries in F that are not consistent with E to 0.
    
    E is a dictionary with keys = variables and values = observed values.
    """
    
    # Solution Start
    

    # Solution End
    
    return Fs


def compute_joint_distribution(Fs):
    """
    Compute the joint distribution.
    """
    
    F = Factor([], [])
    
    # Solution Start
    
    
    # Solution End
    
    return F



def compute_marginal(V, Fs, E):
    """
    computes the marginal over variables V in the distribution induced by the set of factors Fs, given evidence E
    """
    
    marginal = Factor([], [])
    
    # Solution Start
    
        
    # Solution End
    
    return marginal