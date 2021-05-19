import numpy as np
import warnings


def index_2_assignment(I, C, zero_based_index=True):
    """Assignments are zero based. 
       e.g. Binary assignment is either 0 or 1
    """
    warnings.warn("Use np.unravel_index instead.", DeprecationWarning)
    if isinstance(C, np.ndarray):
        C = C.tolist()

    if not isinstance(I, np.ndarray):
        I = np.array(I)

    if I.ndim == 1:  # to apply broadcasting
        I = I[:, None]

    if not zero_based_index:
        I -= 1

    cumprod = np.cumprod([1] + C[:-1])
    return (I // cumprod) % C
