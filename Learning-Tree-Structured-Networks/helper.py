import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl


def show_pose(pose):
    """
    Original Authors: Huayan Wang, Andrew Duchi
    """
    pose[:, 0] += 100
    pose[:, 1] += 150

    part_length = np.array([60, 20, 32, 33, 32, 33, 46, 49, 46, 49])
    part_width = np.array([18, 10, 7, 5, 7, 5, 10, 7, 10, 7])

    img = np.zeros((300, 300), dtype=np.uint8)

    for i in range(10):
        startpt = pose[i, :2].round().astype(int)
        axis = np.r_[np.sin(pose[i, 2] - np.pi / 2), np.cos(pose[i, 2] - np.pi / 2)]
        xaxis = np.r_[np.cos(pose[i, 2] - np.pi / 2), -np.sin(pose[i, 2] - np.pi / 2)]
        endpt = (startpt + part_length[i] * axis).round().astype(int)

        corner1 = (startpt + xaxis * part_width[i]).round().astype(int)
        corner2 = (startpt - xaxis * part_width[i]).round().astype(int)
        corner3 = (endpt + xaxis * part_width[i]).round().astype(int)
        corner4 = (endpt - xaxis * part_width[i]).round().astype(int)

        img = cv2.line(img, tuple(corner1[::-1]), tuple(corner2[::-1]), 255, 2)
        img = cv2.line(img, tuple(corner3[::-1]), tuple(corner4[::-1]), 255, 2)
        img = cv2.line(img, tuple(corner1[::-1]), tuple(corner3[::-1]), 255, 2)
        img = cv2.line(img, tuple(corner2[::-1]), tuple(corner4[::-1]), 255, 2)

        img = cv2.rectangle(img, tuple(startpt[::-1] - 4), tuple(startpt[::-1] + 4), 255, -1)
    return img


def sample_pose(P, G, k):
    """
    Args:
        P:
        G:
        k: None for unknown class, else label $\in$ 0, 1, 2, ... , k-1
    """
    sample = np.zeros((10, 3))
    if k is None:
        k = np.random.choice(len(P['c']), p=P['c'])
        
    remaining = set(range(10))
    while remaining:
        i = remaining.pop()
        clg = P['clg'][i]
        par = G[i, 1]
        
        if G[i, 0] == 0:
            sample[i, 0] = clg['mu_y'][k] + clg['sigma_y'][k]*np.random.randn()
            sample[i, 1] = clg['mu_x'][k] + clg['sigma_x'][k]*np.random.randn()
            sample[i, 2] = clg['mu_angle'][k] + clg['sigma_angle'][k]*np.random.randn()
        elif G[i, 0] == 1:
            if par in remaining:
                remaining.add(i)
                continue
                
            muy = (clg['theta'][k,0] + 
                   clg['theta'][k,1] * sample[par,0] +
                   clg['theta'][k,2] * sample[par,1] +
                   clg['theta'][k,3] * sample[par,2])
            mux = (clg['theta'][k,4] +
                   clg['theta'][k,5] * sample[par,0] +
                   clg['theta'][k,6] * sample[par,1] +
                   clg['theta'][k,7] * sample[par,2])
            muangle = (clg['theta'][k,8] + 
                   clg['theta'][k,9] * sample[par,0] +
                   clg['theta'][k,10] * sample[par,1] +
                   clg['theta'][k,11] * sample[par,2])
            
            sample[i, 0] = muy + clg['sigma_y'][k]*np.random.randn()
            sample[i, 1] = mux + clg['sigma_x'][k]*np.random.randn()
            sample[i, 2] = muangle + clg['sigma_angle'][k]*np.random.randn()
        elif G[i, 0] == 2:
            if par in remaining:
                remaining.add(i)
                continue
                
            muy = (clg['gamma'][k,0] + 
                   clg['gamma'][k,1] * sample[par,0] +
                   clg['gamma'][k,2] * sample[par,1] +
                   clg['gamma'][k,3] * sample[par,2])
            mux = (clg['gamma'][k,4] +
                   clg['gamma'][k,5] * sample[par,0] +
                   clg['gamma'][k,6] * sample[par,1] +
                   clg['gamma'][k,7] * sample[par,2])
            muangle = (clg['gamma'][k,8] + 
                   clg['gamma'][k,9] * sample[par,0] +
                   clg['gamma'][k,10] * sample[par,1] +
                   clg['gamma'][k,11] * sample[par,2])
            
            sample[i, 0] = muy + clg['sigma_y'][k]*np.random.randn()
            sample[i, 1] = mux + clg['sigma_x'][k]*np.random.randn()
            sample[i, 2] = muangle + clg['sigma_angle'][k]*np.random.randn()
    return sample
            
            
def visualize_models(P, G):
    K = len(P['c'])
    figs = []
    for k in range(K):
        if G.ndim == 2:
            pose = sample_pose(P, G, k)
        else:
            pose = sample_pose(P, G[:, :, k], k)
        pose = show_pose(pose)
        figs.append(pose)
    return figs


def visualize_dataset(dataset):
    images = []
    for pose in dataset:
        images.append(show_pose(pose))
    return images


def create_html5_animation(*images, labels=None, nframes=10000, interval=500):
    nrows = len(images)
    nframes = min(nframes, min(map(len, images)))
    
    width = 3
    height = nrows*3
    
    fig, axs = plt.subplots(nrows, 1, figsize=(width, height))
    ims = []
    for i in range(nrows):
        ims.append(axs[i].imshow(images[i][0], cmap='binary'))
        axs[i].set_axis_off()
        if labels is not None:
            axs[i].set_title(labels[i])
    plt.tight_layout()
    
    def init():
        for i in range(nrows):
            ims[i].set_data(images[i][0])
    
    def animate(j, *args, **kwargs):
        for i in range(nrows):
            ims[i].set_data(images[i][j])
        
    ani = FuncAnimation(fig, animate, frames=nframes, init_func=init, interval=interval)
    ani_html = ani.to_html5_video();
    plt.close()
    return ani_html
 

def maximum_spanning_tree(G):
    N = G.shape[0]
    parent = [None]*N
    value = np.ones(N)*-1
    
    Q = set(range(N))
    
    def next_v():
        best_v = None
        best_val = -10
        for v in Q:
            if value[v] > best_val:
                best_val = value[v]
                best_v = v
        return best_v
    
    while Q:
        u = next_v()
        Q.remove(u)
        
        for v in range(N):
            if v in Q and G[u, v] > value[v]:
                value[v] = G[u, v]
                parent[v] = u
    
    adj = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        if parent[i] is not None:
            adj[parent[i], i] = 1
    
    return adj
    

def gaussian_mutual_information(X, Y):
    Z = np.concatenate([X, Y], axis=-1)
    Sxx = np.cov(X, rowvar=False)
    Syy = np.cov(Y, rowvar=False)
    S = np.cov(Z, rowvar=False)
    
    return 0.5 * np.log(np.linalg.det(Sxx)*np.linalg.det(Syy)
                        / np.linalg.det(S))


def convert_A2G(A):
    G = np.zeros((10, 2), dtype=np.int64)
    A = A + A.T
    assert A.sum() == 18
    
    visited = {0}
    while len(visited) < 10:
        for i in range(1, 10):
            for j in range(10):
                if A[i, j] == 1 and j in visited:
                    G[i, 0] = 1
                    G[i, 1] = j
                    visited.add(i)
                    break
    return G