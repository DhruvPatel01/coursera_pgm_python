"""
Following code is converted from Octave/Matlab to Python
so that this Python assignment can be submitted successfully
on Coursera.

Note: I don't think this code works!

Ported by: Dhruv Patel (https://github.com/DhruvPatel01)
"""
import math

import numpy as np


# x_i is the global state. Please do not make it integer ever.
x_i = 1.0
p1 = 160481183
p2 = 179424673


def seed(arg=1.):
    global x_i
    x_i = arg * 1.  # seed needs to be float tobe compatible with matlab


def mod(x, y):  # custom mod is required, as % operator does not behave similar to matlab mod
    return x - math.floor(x/y) * y


def randi(max_val, n_rows=None, n_cols=None):
    global x_i

    if n_rows is None:
        x_i = mod(x_i * (p1 + 1) + p1, p2)
        return int(x_i % max_val)

    if n_cols is None:
        n_cols = n_rows

    rand_matrix = np.zeros((n_rows, n_cols), dtype=np.int64)
    for i in range(n_rows):
        for j in range(n_cols):
            x_i = mod(x_i * (p1 + 1) + p1, p2)
            rand_matrix[i, j] = int(x_i % max_val)
    return rand_matrix


def rand(n_rows=None, n_cols=None):
    if n_rows is None:
        return randi(1e6)/1e6

    if n_cols is None:
        n_cols = n_rows

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




