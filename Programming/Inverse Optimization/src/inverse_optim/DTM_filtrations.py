## Original version from https://github.com/GUDHI/TDA-tutorial/blob/master/DTM_filtrations.py
## Optimized the code for speed increase

import torch
import gudhi as gd
import numpy as np
import torch
import math
from sklearn.neighbors import KDTree


def DTM(X,query_pts,m):
    '''
    Compute the values of the DTM (with exponent p=2) of the empirical measure of a point cloud X.
    For Pytorch GPU acceleration, make sure X and query_pts are stored on the GPU
    
    Input:
    X: a nxd torch tensor or numpy array representing n points in R^d
    query_pts:  a kxd torch tensor or numpy array of query points
    m: parameter of the DTM in [0,1)
    
    Output: 
    DTM_result: a kx1 torch tensor or numpy array contaning the DTM of the query points
    
    Example:
    X = torch.tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Q = torch.tensor([[0,0],[5,5]])
    DTM_values = DTM(X, Q, 0.3)
    '''

    if isinstance(X, torch.Tensor):
        if type(X) is not type(query_pts):
            raise "The query_pts and should be of the same type as the reference point set X."
        
        # Computation of number of neighbors
        N_tot = X.size(dim=0)    
        k = math.floor(m*N_tot) + 1   

        # Computation of pairwise distance of all points
        distances = torch.cdist(X, query_pts)

        # Obtains k indices of the small distances
        _, indices = distances.topk(k, dim=1, largest=False)
        NN_Dist = torch.gather(distances, 1, indices)

        # Calculation of DTM values
        DTM_result = torch.sqrt(torch.sum(NN_Dist**2, dim=1) / k)

        return DTM_result
    else:
        if type(X) is not type(query_pts):
            "The query_pts and should be of the same type as the reference point set X."
        
        N_tot = X.shape[0]     
        k = math.floor(m*N_tot) + 1   # number of neighbors

        kdt = KDTree(X, leaf_size=30, metric='euclidean')
        NN_Dist, NN = kdt.query(query_pts, k, return_distance=True)  

        DTM_result = np.sqrt(np.sum(NN_Dist*NN_Dist,axis=1) / k)

        return DTM_result
    
def AlphaDTMFiltration(X, m, p=1, dimension_max=2):
    '''
    /!\ this is a heuristic method, that speeds-up the computation.
    It computes the DTM-filtration seen as a subset of the Delaunay filtration.
    
    Input:
        X (torch.tensor or numpy array): filtration values of the points x
        p (float): parameter of the weighted Rips filtration, in 1, 2 or np.inf
        m (float): parameter of the DTM, in [0,1) 
        dimension_max (int): maximal dimension to expand the complex
        
    Output: 
        st (float): a gudhi.SimplexTree 
    '''
    if isinstance(X, torch.Tensor):

        # Creation of alpha complex and calculation of DTM values
        N_tot = X.size(dim=0)     
        alpha_complex = gd.AlphaComplex(points=X)
        st_alpha = alpha_complex.create_simplex_tree()    
        Y = torch.tensor([alpha_complex.get_point(i) for i in range(N_tot)], dtype=X.dtype).to(X.device)
        DTM_values = DTM(X,Y,m)        

        # Creation of simplex tree
        st = gd.SimplexTree()                     
        edges = [] # NOTE: THERE MIGHT BE A MORE EFFICIENT WAY USING TORCH.ZEROS
        for simplex in st_alpha.get_skeleton(2):       
            
            # Add vertices with filtration value 
            if len(simplex[0])==1:
                i = simplex[0][0]
                st.insert([i], filtration=DTM_values[i])

            # Obtain list of edges (filtration values to be added later)
            if len(simplex[0])==2:                     
                i = simplex[0][0]
                j = simplex[0][1]
                edges.append([i,j])
        
        # Creation of list of edges and the distances between the vertices of the edges
        edges = torch.tensor(edges).to(X.device)
        distances = torch.linalg.norm(Y[edges[:,0]] - Y[edges[:,1]], axis=-1)

        # Creation of tensor with the filtration values of the edges
        if p==np.inf:
            # THIS STILL NEEDS TO BE TESTED TO SEE IF VALUES DO ACTUALLY MATCH UP
            distances = distances/2
            edges_distances = torch.cat((DTM_values[edges], distances.unsqueeze(1)), dim=1)
            values = torch.max(edges_distances, 1).values

        else:
            # Calculate condition
            cond = distances < torch.abs(DTM_values[edges[:, 0]]**p - DTM_values[edges[:, 1]]**p)**(1/p)

            # Apply different operations using boolean indexing
            values = torch.zeros((len(cond),), dtype=X.dtype).to(X.device)
            values[cond] = torch.max(DTM_values[edges[cond]], dim=1).values 
            
            if p==1:
                values[~cond] = (DTM_values[edges[~cond,0]] + DTM_values[edges[~cond,1]] + \
                                distances[~cond])/2
            elif p==2:
                # THIS STILL NEEDS TO BE TESTED TO SEE IF VALUES DO ACTUALLY MATCH UP
                # Computation of the value given some edge [i,j] with distance d
                def value_2(i, j, d):
                    return torch.sqrt(((i + j)**2 + d**2) * ((i - j)**2 + d**2)) / (2 * d)
            
                values[~cond] = value_2(DTM_values[edges[~cond,0]], DTM_values(edges[~cond, 1]), \
                                distances[~cond])
            else:
                raise "Currently only supported for values p=np.inf, p=1 and p=2"

        # Inserting all the edges with filtration values
        edges = edges.transpose(0, 1)
        if edges.is_cuda:
            edges = edges.to("cpu")
        if values.is_cuda:
            values = values.to("cpu")
        st.insert_batch(edges.detach().numpy(), values.detach().numpy())

        # Expanding simplicial complex to get higher dimensional complex
        st.expansion(dimension_max)
        
        return st
    else:

        # Creation of alpha complex and calculation of DTM values
        N_tot = X.shape[0]     
        alpha_complex = gd.AlphaComplex(points=X)
        st_alpha = alpha_complex.create_simplex_tree()   
        Y = np.array([alpha_complex.get_point(i) for i in range(N_tot)])
        DTM_values = DTM(X,Y,m) 
        
        # Creation of simplex tree
        st = gd.SimplexTree()                     
        edges = [] # NOTE: THERE MIGHT BE A MORE EFFICIENT WAY USING TORCH.ZEROS
        for simplex in st_alpha.get_skeleton(2):

            # Add vertices with filtration value 
            if len(simplex[0])==1:
                i = simplex[0][0]
                st.insert([i], filtration=DTM_values[i])

            # Obtain list of edges (filtration values to be added later)
            if len(simplex[0])==2:                     
                i = simplex[0][0]
                j = simplex[0][1]
                edges.append([i,j])
        
        # Creation of list of edges and the distances between the vertices of the edges
        edges = np.array(edges)
        distances = np.linalg.norm(Y[edges[:,0]] - Y[edges[:,1]], axis=-1)
        
        # Creation of array with the filtration values of the edges
        if p==np.inf:
            # THIS STILL NEEDS TO BE TESTED TO SEE IF VALUES DO ACTUALLY MATCH UP
            distances = distances/2
            edges_distances = np.concatenate((DTM_values[edges], distances.reshape(-1, 1)), axis=1)
            values = np.max(edges_distances, axis=1)
        else:
            cond = distances < np.abs(DTM_values[edges[:, 0]]**p - DTM_values[edges[:, 1]]**p)**(1/p)

            values = np.zeros((len(cond),))
            values[cond] = np.max(DTM_values[edges[cond]], axis=1)
            
            if p==1:
                values[~cond] = (DTM_values[edges[~cond,0]] + DTM_values[edges[~cond,1]] + distances[~cond])/2
            elif p==2:
                # THIS STILL NEEDS TO BE TESTED TO SEE IF VALUES DO ACTUALLY MATCH UP
                def value_2(i, j, d):
                    return np.sqrt(((i + j)**2 + d**2) * ((i - j)**2 + d**2)) / (2 * d)
            
                values[~cond] = value_2(DTM_values[edges[~cond,0]], DTM_values[edges[~cond, 1]], distances[~cond])
            else:
                raise "Currently only supported for values p=np.inf, p=1 and p=2"
        
        # Inserting all the edges with filtration values
        edges = edges.T
        st.insert_batch(edges, values)

        # Expanding simplicial complex to get higher dimensional complex
        st.expansion(dimension_max)
        
        return st

def biline_rips_st(alpha, beta, pts):

    # start_time = timeit.default_timer()
    # Density function on a simplex
    DTM_values = DTM(pts, pts, 0.01)
    def s(simplex):
        return np.max(DTM_values[simplex])
    # print(f"Calculation of density {timeit.default_timer() - start_time}")

    # start_time = timeit.default_timer()
    # Build the rips complex
    complex = gd.AlphaComplex(points=pts)
    st_rips = complex.create_simplex_tree()
    # print(f"Building rips complex {timeit.default_timer() - start_time}")

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
                # print("Change ab")
                st.insert([i], filtration=(dens-beta) + (1/alpha)*(dens-beta))
        else:
            dens = s(simplex[0])
            orig_filt = simplex[1]
            if alpha>= (dens - beta)/orig_filt or dens <= beta:
                st.insert(simplex[0], filtration=alpha*orig_filt + orig_filt)
            else:
                st.insert(simplex[0], filtration=(dens-beta) + (1/alpha)*(dens-beta))
    return st
