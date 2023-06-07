"""gen_data.py

This module is used to generate new datasets that must obey some topological constraints given by a persistence diagram.
It can also be used to transform new datasets into ones that must have some other topological properties.

Todo:
    * Figure out how to sliced wasserstein distance exactly works for multiple thetas and improve the code for speed.
    * Experiment with changing the ordening of the loss (first 1 dimenions, and then 0 dimenions) instead of the same time?
    * Add support for changing m and p value of alpha dtm directly from generate_data.
"""

import torch
from torch.optim.lr_scheduler import LambdaLR
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from inverse_optim import pd
import timeit

### HELP FUNCTIONS ####
def tensor_sort(tensor, key_fn):
    """
    Args:
        tensor: a pytorch tensor of size Nx2
        key_fn: some function that gives a value for each row

    Returns:
        - The sorted tensor with respect to that function.
    """
    # Apply the key function to each element in the tensor
    keys = torch.tensor([key_fn(x) for x in tensor])
    
    # Sort the keys and get the indices
    _, indices = torch.sort(keys)
    
    # Use the sorted indices to sort the original tensor
    sorted_tensor = tensor[indices]
    
    return sorted_tensor

def general_sliced_wasserstein_distance(dgms, ccards, thetas):
    """
    Args:
        dgms: list of persistent diagrams
        ccards: cumulative sum of diagram cardinalities (ccards = np.cumsum([0]+[dgm.shape[0] for dgm in dgms]))
        thetas: angles parametrizing the lines

    Returns:
        - Sliced wasserstein distance
    """

    # Convert ccards and thetas to tensor in case it is not a tensor
   
    if type(ccards) != torch.Tensor:
        ccards = torch.tensor(ccards)
    if type(thetas) != torch.Tensor:
        thetas = torch.tensor(thetas)
    
    dgm_cat = torch.cat(dgms,dim=0).to(torch.float32)
    projected_dgms = torch.matmul(dgm_cat, .5*torch.ones(size=(2,2), dtype=torch.float32))
    dgms_temp = [torch.reshape(
        torch.cat([dgm, projected_dgms[:ccards[idg]], projected_dgms[ccards[idg+1]:]], dim=0), \
            [-1,2,1,1]) for idg, dgm in enumerate(dgms)]
    dgms_big = torch.cat(dgms_temp, dim=2)
    cosines, sines = torch.cos(thetas), torch.sin(thetas)
    vecs = torch.cat([torch.reshape(cosines,[1,1,1,-1]), torch.reshape(sines,[1,1,1,-1])], dim=1)
    theta_projs, _ = torch.sort(torch.sum(torch.mul(dgms_big, vecs), dim=1), dim=0)

    t1 = torch.reshape(theta_projs, [ccards[-1], -1, 1, len(thetas)])
    t2 = torch.reshape(theta_projs, [ccards[-1], 1, -1, len(thetas)])

    dists = torch.mean(torch.sum(torch.abs(t1 - t2), dim = 0), dim = 2)
    return dists

def sliced_wasserstein_distance(dgms, theta):
    ccards = torch.tensor(np.cumsum([0] + [dgm.shape[0] for dgm in dgms]))
    dists = general_sliced_wasserstein_distance(dgms, ccards, [theta])
    dist = dists[0,1]
    return dist

### GENERATION OF DATA ###
def wasser_loss(pts, goal_pd, sliced=False, thetas=torch.tensor([k/4 * np.pi for k in range(5)]), \
                filtr="alpha_rips_hybrid", max_edge_length=0.5, nx=16, ny=16):
    """
    Args:
        pts       : points of the dataset that we want to change into a dataset that 
                    has the same topological features of some other dataset
        goal_pd   : persistence diagram of the dataset we want our initial point set 
                    to approximate in the sense that it has the same topological 
                    features.
                    IMPORTANT. If you use sliced method with float, the goal_pd's are assumed to be 1D.
        sliced    : if set to true, it will use the sliced wasserstein distance. If set to a float, 
                    you will get that percentage of the most persistent features after which we do a 
                    euclidean distance
        thetas    : the angles used for the sliced wasserstein distance
        filtr     : name of the filtration method you would like to use
        max_edge_length
                  : length of the maximal length of an edge that we are willing to 
                    include in the filtration 
        nx        : given sublevel filtration, it is the amount of subdivisions of the x coordinates
        ny        : given sublevel filtration, it is the amount of subdivisions of the y coordinates

    Returns:
        - The distance between the PD created using pts and some goal pd

    NOTE: Currently the sliced wasserstein distance only supports distances between 1 dimensional pd's.
    """

    # Creation of persistene diagram
    if filtr == "alpha_rips_hybrid":
        diags = pd.create_hybrid_pd(pts)
    elif filtr == "rips":
        diags = pd.create_rips_pd(pts, max_edge_length=max_edge_length)
    elif filtr == "alpha_dtm":
        diags = pd.create_hybrid_dtm_pd(pts)
    # elif filtr == "line":
    #     diags = pd.create_line_pd(pts)
    # elif filtr == "sub_kernel":
    #     diags = pd.create_sublevel_kernel_pd(pts, nx, ny)
    else:
        raise Exception("Please specify a filtration method that is known.\
                        The options are: alpha_rips_hybrid, rips, alpha_dtm")
    
    # Extract diagrams
    diag0 = diags[0]
    diag1 = diags[1]
    goal_pd0 = goal_pd[0]
    goal_pd1 = goal_pd[1]


    # Total persistence is a special case of Wasserstein distance
    if sliced == False:
        dist0 = wasserstein_distance(diag0, goal_pd0, order=1, enable_autodiff=True, keep_essential_parts=False)  
        dist1 = wasserstein_distance(diag1, goal_pd1, order=1, enable_autodiff=True, keep_essential_parts=False)  
        dist = dist0 + dist1
        # dist =  dist1
        return dist
    elif isinstance(sliced, float):
        if len(goal_pd0.size()) != 1 or len(goal_pd1.size()) != 1:
            raise print("The pd's should be 1 dimensional. You can use the following code:\
                         diag0 = torch.reshape(gen_data.tensor_sort(diag0, key_fn), (-1,))\
                        where key_fn = lambda x: x[0] - x[1]")

        start_timer = timeit.default_timer() # START TIMER
        # Sort in reverse by persistence (first the longest)
        key_fn = lambda x: x[0] - x[1]

        # Sorting everyting by persistence and making it into a vector
        diag0 = torch.reshape(tensor_sort(diag0, key_fn), (-1,))
        if diag1.shape[0] != 0:
            diag1 = torch.reshape(tensor_sort(diag1, key_fn), (-1,))
        else:
            diag1 = torch.tensor([])

        # Calculating the "sliced" percent of features we want to keep
        zero_dim_amount = round(min(diag0.shape[0], goal_pd0.shape[0]) * sliced)
        one_dim_amount = round(min(diag1.shape[0], goal_pd1.shape[0]) * sliced)

        # Throwing out the non-persistent features
        diag0 = diag0[:zero_dim_amount]
        goal_pd0 = goal_pd0[:zero_dim_amount]
        diag1 = diag1[:one_dim_amount]
        goal_pd1 = goal_pd1[:one_dim_amount]

        # Computing the distance
        dist0 = torch.linalg.norm(goal_pd0 - diag0)
        dist1 = torch.linalg.norm(goal_pd1 - diag1)

        dist = dist0 + dist1
        return dist
    else:
        # dist = sliced_wasserstein_distance([diag, goal_pd], thetas[3])
        dists = [sliced_wasserstein_distance([diag1, goal_pd1], theta) for theta in thetas]
        dist = sum(dists)/len(dists)
        return dist

def generate_data(goal_pd, amount, dim, lr, epochs, decay_speed=10, investigate=False, sliced=False, \
                  thetas=torch.tensor([k/4 * np.pi for k in range(5)]), filtr="alpha_rips_hybrid", \
                    max_edge_length=0.5, box_size=1, init_pts="random", nx=16, ny=16):
    """
    Args:
        goal_pd             : persistence diagram that we want to 'approximate' in the sense that we want to 
                              generate a new dataset that approximately has the same persistence diagram
        amount (int)        : amount of data points
        dim (int)           : dimension of data points
        lr (float)          : learning rate 
        epochs (int)        : amount of learning steps
        decay_speed         : parameter that tunes how fast the learning rates decays (note that higer values correspond with 
                              lower decay speed).
        investigate (bool) 
                            : if set to true, it will produce a list of the losses and avg/max/min magnitude of movements per points.
        sliced              : if set to true, it will use the sliced wasserstein distance
        thetas              : the angles used for the sliced wasserstein distance
        filtr               : name of the filtration method you would like to use
        max_edge_length     : length of the maximal length of an edge that we are willing to 
                              include in the filtration
        box_size            : size of the box of the target point set
        init_pts            : initial point set that we want to change into some target point set
        nx                  : given sublevel filtration, it is the amount of subdivisions of the x coordinates
        ny                  : given sublevel filtration, it is the amount of subdivisions of the y coordinates
    
    Returns:
        - The generated dataset
        - A list of the losses at each epoch (if investigate_loss==True)
        - Plots of the generated dataset every 100 epochs including the final epoch (if investigate_loss==False)
    """
    if init_pts == "random":
        pts = (torch.rand((amount, dim)))
        pts = pts * box_size
        pts.requires_grad_()
    else:
        pts = torch.tensor(init_pts)
        if pts.requires_grad:
            raise Exception("The initial point set must have requires_grad_() to to False.\
                             Also make sure for the research part that the init_pts is not a tensor.")
        else:
            pts.requires_grad_()
    
    # Set up optimizer for SGD
    opt = torch.optim.SGD([pts], lr=lr)
    scheduler = LambdaLR(opt,[lambda epoch: decay_speed/(decay_speed+epoch)])

    # Initialize loss list
    if investigate:
        loss_list = []
        
    if investigate == False:
        # Plot initial point cloud
        P = pts.detach().numpy()
        plt.scatter(P[:, 0], P[:, 1])
        plt.show()

    # Perform SGD
    for epoch in tqdm(range(epochs)):
        opt.zero_grad()

        wasser_loss(pts, goal_pd, sliced, thetas, filtr, max_edge_length,nx=nx, ny=ny).backward()
        if investigate:
            loss_list.append(wasser_loss(pts, goal_pd, sliced, thetas, filtr).item())
       
        opt.step()
        scheduler.step()

        if investigate == False:
            # Draw every 100 epochs
            if epoch % 100 == 99:
                P = pts.detach().numpy()
                plt.scatter(P[:, 0], P[:, 1])
                plt.show()
    
    if investigate:
        return pts, loss_list
    return pts