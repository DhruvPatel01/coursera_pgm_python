import numpy as np

seeds = {}

disable = False

for s in [1, 2, 26288942]:
    with open(f'./data/seed{s}.txt') as fin:
        data = fin.read()
        seeds[s] = list(map(int, data.split()))

current_store = seeds[1]
current_index = 0


def seed(arg):
    assert arg in seeds, "Invalid seed"
    global current_store, current_index
    current_store = seeds[arg]
    current_index = 0


def randi(max_val, n_rows=None, n_cols=None):
    global current_index
    if n_rows is None:
        current_index += 1
        return current_store[current_index-1] % max_val

    if n_cols is None:
        n_cols = n_rows

    rand_matrix = np.zeros((n_rows, n_cols), dtype=np.int64)
    for i in range(n_rows):
        for j in range(n_cols):
            current_index += 1
            rand_matrix[i, j] = current_store[current_index-1] % max_val
    return rand_matrix


def rand(n_rows=None, n_cols=None):
    """
    Args:
        (None, None) returns random scalar in between 0 and 1.
        (n_rows, None) returns random matrix of (n_rows, n_rows) size
        (n_rows, n_cols) returns random matrix of (n_rows, n_cols) size
    """
    if n_rows is None:
        if disable:
            return np.random.rand()
        return randi(1e6)/1e6

    if n_cols is None:
        n_cols = n_rows
        
    if disable:
        return np.random.rand(n_rows, n_cols)
    else:
        return randi(1e6, n_rows, n_cols)/1e6


def randsample(vals: int, num_samples: int, p=None):
    """
    Returns n random integers(with replacement) from [0, vals).

    Note: Original Octave code seems to have a bug
    when replacement flag is False. However since
    none of the tasks requires replacement to be False, I've
    not implemented bug free version of without replacement.
    If you need it please implement by yourself and open a
    PR request so I can have that too!

    Args:
        vals: 1 plus the highest value required
        num_samples: how many samples are required?
        p: An array of length V, gives weights. If None all weights
         are same

    Returns:
        Array of n longs
    """
    assert vals > 0, "Only +ve vals is allowed"
    if p is None:
        w = np.linspace(0, 1, vals+1)
    else:
        p = np.array(p)
        w = np.r_[0, (p/p.sum()).cumsum()]

    w = w[None, :]
    probs = rand(num_samples, 1)
    idx = (w[:, :-1] <= probs) & (w[:, 1:] >= probs)
    _, sample = idx.nonzero()
    return sample