import numpy as np
import math
import gudhi as gd
from sklearn.neighbors import KDTree
import timeit
from tqdm import tqdm


def DTM(X,query_pts,m):
    '''
    Compute the values of the DTM (with exponent p=2) of the empirical measure of a point cloud X
    Require sklearn.neighbors.KDTree to search nearest neighbors
    
    Input:
    X: a nxd numpy array representing n points in R^d
    query_pts:  a kxd numpy array of query points
    m: parameter of the DTM in [0,1)
    
    Output: 
    DTM_result: a kx1 numpy array contaning the DTM of the 
    query points
    
    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Q = np.array([[0,0],[5,5]])
    DTM_values = DTM(X, Q, 0.3)
    '''
    N_tot = X.shape[0]     
    k = math.floor(m*N_tot)+1   # number of neighbors

    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    NN_Dist, NN = kdt.query(query_pts, k, return_distance=True)  

    DTM_result = np.sqrt(np.sum(NN_Dist*NN_Dist,axis=1) / k)
    
    return(DTM_result)


def biline_rips_st(alpha, beta, pts):

    start_time = timeit.default_timer()
    # Density function on a simplex
    DTM_values = DTM(pts, pts, 0.01)
    def s(simplex):
        return np.max(DTM_values[simplex])
    print(f"Calculation of density {timeit.default_timer() - start_time}")

    start_time = timeit.default_timer()
    # Build the rips complex
    complex = gd.RipsComplex(points=pts)
    st_rips = complex.create_simplex_tree(max_dimension=2)
    print(f"Building rips complex {timeit.default_timer() - start_time}")

    # Creation of new simplex tree
    st = gd.SimplexTree()
    for simplex in st_rips.get_skeleton(2): 

        # Assign filtration value to each simplex (see thesis)
        if len(simplex[0]) == 1:
            i = simplex[0][0]
            dens = DTM_values[i]
            if dens <= beta:
                st.insert([i], filtration=0)
            else:
                print("Change ab")
                st.insert([i], filtration=(dens-beta) + (1/alpha)*(dens-beta))
        else:
            dens = s(simplex[0])
            orig_filt = simplex[1]
            if alpha>= (dens - beta)/orig_filt or dens <= beta:
                st.insert(simplex[0], filtration=alpha*orig_filt + orig_filt)
            else:
                st.insert(simplex[0], filtration=(dens-beta) + (1/alpha)*(dens-beta))
    return st