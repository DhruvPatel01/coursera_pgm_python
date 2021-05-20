import numpy as np

def smooth(Y, window=5):
    Y = Y.squeeze()
    if window%2 == 0:
        window = window+1
    mid = (window-1)//2
    
    smoother = np.zeros((len(Y), len(Y)))
    for i in range(len(Y)):
        dev = min(mid, min(i, len(Y)-1-i))
        smoother[i, i-dev:i+dev+1] = 1
    col = smoother.sum(axis=1)
    return (smoother @ Y)/col
