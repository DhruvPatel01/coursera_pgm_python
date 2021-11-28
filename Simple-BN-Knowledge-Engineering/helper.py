import numpy as np

import sol

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

    f = sol.Factor(var, card)
    for i, val in enumerate(factor_dict['val']):
        assignment = np.unravel_index(i, card, order='F')
        f[assignment] = val

    return f


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

    f = sol.Factor(var, card)
    for i, val in enumerate(struct.val):
        assignment = np.unravel_index(i, card, order='F')
        f[assignment] = val

    return f


