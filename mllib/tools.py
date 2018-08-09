import numpy as np

def whiten(patches,eps_zca):
    """
    Performs whitening of the patches
    -------------
    Parameters:
        patches: collection of patches, numpy array nb_patches x dim_patch
        eps_zca: zca whitening constant
    """
    C = np.cov(patches,rowvar=False)
    M = np.mean(patches,axis=0)
    M = M.reshape((len(M),1))
    D,V = np.linalg.eig(C)
    D = D.reshape((len(D),1))
    D_zca = np.sqrt(1 / (D + eps_zca))
    D_zca = D_zca.reshape(len(D_zca))
    D_zca = np.diag(D_zca)
    P = np.dot(V,np.dot(D_zca,V.transpose()))
    patches = np.dot((patches.transpose() - M).transpose(),P)
    return patches,M,P

def standard(X):
    """
    Subtract the mean to each row and divide by the standard deviation
    ---------------
    Parameters:
        X: multidimensional numpy array
    """
    mean = np.mean(X,axis=0)
    var = np.var(X,axis=0,ddof=1)+0.01
    var = np.sqrt(var)
    X_standard = X - mean
    X_standard = np.divide(X_standard,var)
    X_standard = np.concatenate((X_standard,np.ones((X_standard.shape[0],1))), axis=1)
    return X_standard