"""filtrations.py

This module is used to create persistence diagrams for different types of filtrations.
It is important to note that these filtrations are already widely supported using Gudhi.
The biggest difference with the implementations of Gudhi, is that they are usually not implemented
in a way that is differentiable (with respect to Pytorch). 
This module creates the persistence diagrams in a differentiable manner.

Todo:
    * Make a differentiable version of the sublevel filtration
    * Include the differentiable biline filtrations
    * Adapt the version of the create_rips_pd so that it also produces 0-dimensional pd's
    * Change the
    * Change the differentiable Alpha (DTM) filtration so that it is more faithful to the actual Alpha DTM filtration.
        - Currently the filtration values for 0 dimensional simplices are all 0
        - Currently the filtration values for the higher dimensional simplices are all given by the rips distances
    * Check of GRF still works now that grad part is removed?
"""

import torch
import gudhi as gd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from inverse_optim import DTM_filtrations
from sklearn.neighbors import KernelDensity


def create_rips_pd(pts, max_edge_length):
    """
    Args:
        pts       : points of the dataset that we want to change into a dataset that 
                    has the same topological features of some other dataset (shape is Nx2)
        max_edge_length
                  : length of the maximal length of an edge that we are willing to 
                    include in the filtration

    Returns:
        - Persistence differentiable rips diagram of with only 1-dimensional features

    NOTE: Currently only supports pd's that only include 1-dimensional features.
    """
    complex0 = gd.RipsComplex(points=pts, max_edge_length=max_edge_length)
    rips_complex = complex0.create_simplex_tree(max_dimension=2)
    rips_complex.compute_persistence()

    i = rips_complex.flag_persistence_generators()
    if len(i[1]) > 0:
        i1 = torch.tensor(i[1][0])  # pytorch sometimes interprets it as a tuple otherwise
    else:
        i1 = torch.empty((0, 4), dtype=int)

    # Same as the finite part of st.persistence_intervals_in_dimension(1), but differentiable
    diag1 = torch.norm(pts[i1[:, (0, 2)]] - pts[i1[:, (1, 3)]], dim=-1)
    return [torch.tensor([[0,0]]), diag1]

def create_hybrid_dtm_pd(pts, m=0.7, p=1):
    """
    Args:
        pts       : points of the dataset that we want to change into a dataset that 
                    has the same topological features of some other dataset (shape is Nx2)
        m (float) : percentage of nearest neighbors, i.e. some parameter in [0,1)
        p (float) : parameter of the weighted Rips filtration, in 1, 2 or np.inf

    Returns:
        - A list [pd0, pd1] where pd0 and pd1 are borne from (adjusted) alpha dtm filtrations 
    """
    # Creaton of alpha DTM only works for non-grad tensors
    if pts.requires_grad:
        non_grad_pts = pts.detach().numpy()
    else:
        non_grad_pts = pts

    # Creation of simplex tree
    # alpha_dtm_st = DTM_filtrations.AlphaDTMFiltration(X=non_grad_pts, m=m, p=p)
    alpha_dtm_st = DTM_filtrations.AlphaDTMFiltration(X=non_grad_pts, m=m)

    # Calculating persistence
    alpha_dtm_st.compute_persistence(2)
    p = alpha_dtm_st.persistence_pairs()

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

def create_hybrid_pd(pts):
    """
    Args:
        pts       : points of the dataset that we want to change into a dataset that 
                    has the same topological features of some other dataset (shape is Nx2)

    Returns:
        - A list [pd0, pd1] where pd0 and pd1 are borne from (adjusted) alpha filtrations 
    """
    # Creation of simplex tree
    alpha_complex = gd.AlphaComplex(points=pts)
    alpha_st = alpha_complex.create_simplex_tree()

    # Calculating persistence
    alpha_st.compute_persistence(2)
    p = alpha_st.persistence_pairs()

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
        b0 = torch.tensor([[0] for _ in range(len(p0b))])

        # Calculate the death times 
        d0 = torch.norm(pts[p0d[:,1]] - pts[p0d[:,0]], dim=-1, keepdim=True)

        diag0 = torch.cat((b0,d0), 1)
    
    return [diag0, diag1]


# Still under development
def create_sublevel_kernel_pd(pts, nx, ny):
    """
    NOTE: This still needs to be finished. Currently this is not differentiable.
    Given some point set, we create the sublevel filtration of the ker.
    nx is subdivision of x coordinates
    ny is subdivision of y coordinates
    With this you get a box that is divided up in boxes given by these subdivisions.
    """

    # Upper and lowerbounds of box
    np_pts = pts.detach().numpy()
    try:
        # This goes wrong for tensors
        pts_min = np.min(pts)
        pts_max = np.max(pts)
    except:
        pts_min = np.min(np_pts)
        pts_max = np.max(np_pts)

    # Creation of grid
    xval = np.linspace(pts_min, pts_max, nx)
    yval = np.linspace(pts_min, pts_max, ny)

    # Creation of kernel density estimator
    kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.3).fit(np_pts)
    positions = np.array([[u, v] for u in xval for v in yval])
    filt_values = -kde.score_samples(X=positions)

    # Creation of persistence diagram 
    cc = gd.CubicalComplex(dimensions = [nx ,ny], top_dimensional_cells = filt_values)
    cc.compute_persistence()
    goal_pd0 = torch.tensor(cc.persistence_intervals_in_dimension(0))
    goal_pd1 = torch.tensor(cc.persistence_intervals_in_dimension(1))

    # Removing infinities (for now)
    for index, interval in enumerate(goal_pd0):
        if interval[1] == np.inf:
            goal_pd0 = np.delete(goal_pd0, index, axis=0)

    for index, interval in enumerate(goal_pd1):
        if interval[1] == np.inf:
            goal_pd1 = np.delete(goal_pd1, index, axis=0)

    goal_pd = [goal_pd0, goal_pd1]

    return goal_pd
