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

def create_line_pd(pts, alpha=0.5, beta=0):
    """
    Args:
        tensor: a pytorch tensor of size Nx2
        key_fn: some function that gives a value for each row

    Returns:
        - The tensor with respect to that tensor.
    """
    # Creaton of alpha DTM only works for non-grad tensors
    if pts.requires_grad:
        non_grad_pts = pts.detach().numpy()
    else:
        non_grad_pts = pts

    # Creation of simplex tree
    biline_st = DTM_filtrations.biline_rips_st(pts=pts.detach().numpy(), alpha=alpha, beta=beta)


    # Calculating persistence
    biline_st.compute_persistence(2)
    p = biline_st.persistence_pairs()

    # Keep only pairs that contribute to H1, i.e. (edge, triangle), and separate birth (p1b) and death (p1d)
    p1b = torch.tensor([i[0] for i in p if len(i[0]) == 2])
    p1d = torch.tensor([i[1] for i in p if len(i[0]) == 2])

    # Keep only pairs that contribute to H0, i.e. (vertex, edge), and separate birth (p1b0) and death (p1d0)
    # Skipping the infinities by checking second part instead of first part.
    p0b = torch.tensor([i[0] for i in p if len(i[1]) == 2])
    p0d = torch.tensor([i[1] for i in p if len(i[1]) == 2])

    # Compute the distance between the extremities of the birth edge for H1
    if len(p1b) == 0:
        diag1 = torch.tensor([])
    else:
        b = torch.norm(pts[p1b[:,1]] - pts[p1b[:,0]], dim=-1, keepdim=True)
        
        if len(p1d) == 0:
            d = torch.tensor([float('inf')])
        else:
            # For the death triangle, compute the maximum of the pairwise distances
            d_1 = torch.norm(pts[p1d[:,1]] - pts[p1d[:,0]], dim=-1, keepdim=True)
            d_2 = torch.norm(pts[p1d[:,1]] - pts[p1d[:,2]], dim=-1, keepdim=True)
            d_3 = torch.norm(pts[p1d[:,2]] - pts[p1d[:,0]], dim=-1, keepdim=True)
            d = torch.max(d_1, torch.max(d_2, d_3))

        # *Not* the same as the finite part of st.persistence_intervals_in_dimension(1)
        diag1 = torch.cat((b,d), 1)
    
    # Compute the distance between the extremities of the birth edge for H0
    if len(p0b) == 0:
        diag0 = torch.tensor([])
    else:
        # All birth times are 0 for the zero dimensional features
        # b0 = torch.norm(pts[p0b[:,1]] - pts[p0b[:,0]], dim=-1, keepdim=True)
        b0 = torch.tensor([[0] for _ in range(len(p0b))])

        # Calculate the death times 
        d0 = torch.norm(pts[p0d[:,1]] - pts[p0d[:,0]], dim=-1, keepdim=True)

        diag0 = torch.cat((b0,d0), 1)
    

    return [diag0, diag1]