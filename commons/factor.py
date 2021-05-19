import itertools

import numpy as np


class Factor:
    """Generic class for discrete factors.
        To create a factor of two variables, x (having two possible values) and y (having 3 possible values) use,
        f = Factor(['x', 'y'], [2, 3])

        To assign or access value of assignment [x=1, y=0], you can either use
            f[1, 0] = .5 (here order of the keys must match `variables` argument in the constructor) or,
            f[{'y': 1, 'x': 0}] = .5

        This class supports following operators.

        Equality:
            f1 == f2 returns true if both factors have same set of variables, and
                same domain for each of the variables, and
                same value(up to six decimal places) for each assignment.

        Factor multiplication:
            ```f1 = Factor(['a', 'b', 'c'], [2, 2, 2])
               f2 = Factor(['b', 'c', 'd'], [2, 2, 3])
               initialize(f1); initialize(f2);
               f3 = f1*f2``` returns a `Factor(['a', 'b', 'c', 'd'], [2, 2, 2, 3])`

        Factor addition:
            similar to factor multiplication, except factors are added to get final factor

        Uninitialized factor multiplication:
            `f3 = f1 @ f2`, returns uninitialized factor product
    """

    def __init__(self, variables, domains, init=None, default=None):
        """
        One needs to assign factor values after initialization as constructor does not initialize assignments.

        Args:
            variables: a list of hashable names. e.g. list[str] or list[int]
            domains: list of list of elements. 
                e.g. if variables[0] is "Coin_Flip", domains[0] could be ["Heads", "Tails"]
                if domains[i] is an integer, it will be converted to [0, 1, ..., domains[i]-1]
            init: float: initialize value of each assignment to init value
            default: float: if not None, when assignment is not assigned (e.g. no init) return this value.
                be careful of using this argument as not all methods can be applied on Factor object if
                this argument is not None.
        """
        if len(set(variables)) != len(variables):
            raise ValueError("Duplicate variable names are not permitted")

        if len(variables) != len(domains):
            raise ValueError("domains size must match variables list")

        self.domains = [None]*len(domains)
        for i, d in enumerate(domains):
            if isinstance(d, (int, np.integer)):
                self.domains[i] = range(d)
            else:
                self.domains[i] = domains[i]

        self.vars = variables
        self.val = {}

        if init is not None:
            for assignment in itertools.product(*self.domains):
                self.val[assignment] = init

        self.default = default

    def __repr__(self):
        s = "Discrete Factor with variables: %r" % (self.vars, )
        return s

    def __eq__(self, other):
        if set(self.vars) != set(other.vars):
            return False

        other_map = [other.vars.index(var) for var in self.vars]
        for i in range(len(self.vars)):
            if self.domains[i] != other.domains[other_map[i]]:
                return False
        
        for other_assignment in itertools.product(*other.domains):
            assignment = tuple(other_assignment[i] for i in other_map)
            if abs(self[assignment] - other[other_assignment]) > 1e-6:
                return False

        return True

    def __setitem__(self, assignment, value):
        if isinstance(assignment, dict):
            assignment = tuple(assignment[k] for k in self.vars)
        elif not isinstance(assignment, tuple):
            if len(self.vars) > 1:
                raise KeyError("Unable to understand the key")
            assignment = (assignment, )
        self.val[assignment] = value

    def __getitem__(self, assignment):
        if isinstance(assignment, dict):
            assignment = tuple(assignment[k] for k in self.vars)
        elif not isinstance(assignment, tuple):
            if len(self.vars) > 1:
                raise KeyError("Unable to understand the key")
            assignment = (assignment, )

        if assignment not in self.val:
            if self.default is None:
                raise KeyError("Factor value for assignment: %r is not set" % (assignment, ))
            else:
                return self.default
        return self.val[assignment]

    def __iter__(self):
        for key in self.val.keys():
            yield key

    def __matmul__(self, other):
        for var in set(self.vars) & set(other.vars):
            if self.domains[self.vars.index(var)] != other.domains[other.vars.index(var)]:
                raise ValueError("Domains of common variable %r do not match in both factors." % (var, ))

        new_vars = sorted(set(self.vars) | set(other.vars))
        new_domains = []
        for var in new_vars:
            if var in self.vars:
                new_domains.append(self.domains[self.vars.index(var)])
            else:
                new_domains.append(other.domains[other.vars.index(var)])

        return Factor(new_vars, new_domains)

    def __mul__(self, other):
        new_factor = self @ other

        left_map = [new_factor.vars.index(var) for var in self.vars]
        right_map = [new_factor.vars.index(var) for var in other.vars]

        for assignment in itertools.product(*new_factor.domains):
            left_val = self[tuple(assignment[t] for t in left_map)] if left_map else 1  # else allows empty left
            right_val = other[tuple(assignment[t] for t in right_map)] if right_map else 1  # else allows empty right
            new_factor[assignment] = left_val * right_val

        return new_factor

    def __add__(self, other):
        new_factor = self @ other

        left_map = [new_factor.vars.index(var) for var in self.vars]
        right_map = [new_factor.vars.index(var) for var in other.vars]

        for assignment in itertools.product(*new_factor.domains):
            left_val = self[tuple(assignment[t] for t in left_map)] if left_map else 0  # else allows empty left
            right_val = other[tuple(assignment[t] for t in right_map)] if right_map else 0  # else allows empty right
            new_factor[assignment] = left_val + right_val

        return new_factor

    def dummy_marginalise(self, vars_to_marginalise):
        """
        Eliminates `vars_to_marginalise` without initializing resulting factor.
        i.e. just computes skeleton of the new factor.
        """
        vars_to_marginalise = set(vars_to_marginalise)
        for var in vars_to_marginalise:
            if var not in self.vars:
                raise ValueError("Variable %r not in factor" % (var,))

        new_vars_idx = [i for i, v in enumerate(self.vars) if v not in vars_to_marginalise]
        new_vars = [self.vars[i] for i in new_vars_idx]
        new_domains = [self.domains[i] for i in new_vars_idx]
        new_factor = Factor(new_vars, new_domains)
        return new_factor

    def marginalise(self, vars_to_marginalise):
        """Returns new marginalised factor where `vars_to_marginalise` are summed up."""
        new_factor = self.dummy_marginalise(vars_to_marginalise)
        new_vars_idx = [i for i, v in enumerate(self.vars) if v not in vars_to_marginalise]

        for assignment in itertools.product(*new_factor.domains):
            new_factor[assignment] = 0

        for assignment in itertools.product(*self.domains):
            new_assignment = tuple(assignment[i] for i in new_vars_idx)
            new_factor[new_assignment] += self.val[assignment]
        return new_factor

    def normalize(self):
        """normalizes the factor. Inplace operation. Returns nothing."""
        z = sum(self.val.values())
        for key in self.val:
            self.val[key] = self.val[key] / z

    def evidence(self, evidence):
        """Returns new factor consistent with the evidence
        Args:
            evidence: A dictionary that maps variable to the observed value.
                e.g. {"coin1": "Heads"}

        Returns:
            new factor with same set of variables and same set of domains but
            factor values that are not consistent with the evidence are zeroed out.
        """
        if self.default is not None:
            raise NotImplementedError("Yet to implement for factor having default assignment")

        relevant_evidence = {}
        for v, e in evidence.items():
            if v in self.vars:
                idx = self.vars.index(v)
                if e not in self.domains[idx]:
                    raise ValueError("%r not in the domain of %r, which is %r" % (e, v, self.domains[idx]))
                relevant_evidence[idx] = e
        evidence = relevant_evidence

        if not evidence:
            return self
        new_factor = Factor(self.vars, self.domains)

        for assignment, value in self.val.items():
            for v, e in evidence.items():
                if assignment[v] != e:
                    new_factor[assignment] = 0
                    break
            else:
                new_factor[assignment] = value

        return new_factor

    def log_transform(self, inplace=False):
        if inplace:
            new_factor = self
        else:
            new_factor = Factor(self.vars, self.domains)

        for assignment in self.val:
            new_factor[assignment] = np.log(self.val[assignment])
        return new_factor

    @staticmethod
    def from_matlab(factor_dict, start_from_zero=True):
        """This method is used for course assignments.

        Factor saved in .mat format can be loaded by `scipy.io.loadmat` matrix and
        can be passed to this function to create Factor object.
        """
        var = factor_dict['var']
        if start_from_zero:
            var = var - 1

        card = factor_dict['card']
        if not isinstance(var, np.ndarray):
            var = [int(var)]
            card = [int(card)]
        else:
            var = var.astype(int).tolist()
            card = card.astype(int).tolist()
        
        f = Factor(var, card)
        for i, val in enumerate(factor_dict['val']):
            assignment = np.unravel_index(i, card, order='F')
            f[assignment] = val

        return f

    @staticmethod
    def from_mat_struct(struct, start_from_zero=True):
        var = struct.var
        if start_from_zero:
            var = var - 1

        card = struct.card
        if not isinstance(var, np.ndarray):
            var = [var]
            card = [card]
        else:
            var = var.tolist()
            card = card.tolist()

        f = Factor(var, card)
        for i, val in enumerate(struct.val):
            assignment = np.unravel_index(i, card, order='F')
            f[assignment] = val

        return f


def log_prob_of_joint_assignment(factors, assignment):
    if isinstance(assignment, (list, tuple, np.ndarray)):
        assignment = {i: a for i, a in enumerate(assignment)}
    assert isinstance(assignment, dict)
    return np.sum([np.log(f[assignment]) for f in factors])
