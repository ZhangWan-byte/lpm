import ot
import numpy as np

def sinkhorn_wasserstein2(X, Y, reg=0.1):
    if X.shape[0] != Y.shape[0]:
        N1 = X.shape[0]
        N2 = Y.shape[0]
        a = np.ones(N1) / N1  # uniform weights for X
        b = np.ones(N2) / N2  # uniform weights for Y
    else:
        N = X.shape[0]
        a = b = np.ones(N) / N  # uniform weights
        
    M = ot.dist(X, Y, metric='euclidean') ** 2
    transport_plan = ot.sinkhorn(a, b, M, reg)
    return np.sqrt(np.sum(transport_plan * M))