import torch
from torch.optim.lr_scheduler import LambdaLR
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from inverse_optim import DTM_filtrations
from sklearn.neighbors import KernelDensity
import timeit

### HELP FUNCTIONS ####
def tensor_sort(tensor, key_fn):
    # Apply the key function to each element in the tensor
    keys = torch.tensor([key_fn(x) for x in tensor])
    
    # Sort the keys and get the indices
    _, indices = torch.sort(keys)
    
    # Use the sorted indices to sort the original tensor
    sorted_tensor = tensor[indices]
    
    return sorted_tensor


### FILTRATIONS ###

def create_line_pd(pts, alpha=0.5, beta=0):

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

def create_rips_pd(pts, max_edge_length):
    """
    Args:
    pts       : points of the dataset that we want to change into a dataset that 
                has the same topological features of some other dataset
    goal_diag : persistence diagram of the dataset we want our initial point set 
                to approximate in the sense that it has the same topological 
                features
    max_edge_length
              : length of the maximal length of an edge that we are willing to 
                include in the filtration

    Returns:
        persistence diagram

    NOTE: Currently only supports 1 dimensional pd's.
    """
    complex0 = gd.RipsComplex(points=pts, max_edge_length=max_edge_length)
    rips_complex = complex0.create_simplex_tree(max_dimension=2)
    rips_complex.compute_persistence()

    i = rips_complex.flag_persistence_generators()
    if len(i[1]) > 0:
        i1 = torch.tensor(i[1][0])  # pytorch sometimes interprets it as a tuple otherwise
    else:
        i1 = torch.empty((0, 4), dtype=int)
    
    # if len(i[0]) > 0:
    #     i0 = torch.tensor(i[0][0]) 
    # else:
    #     i0 = torch.empty((0, 4), dtype=int)

    # Same as the finite part of st.persistence_intervals_in_dimension(1), but differentiable
    diag1 = torch.norm(pts[i1[:, (0, 2)]] - pts[i1[:, (1, 3)]], dim=-1)

    # # Same as the finite part of st.persistence_intervals_in_dimension(1), but differentiable
    # diag0 = torch.norm(pts[i1[:, (0, 2)]] - pts[i1[:, (1, 3)]], dim=-1)
    return [torch.tensor([[0,0]]), diag1]

def create_hybrid_dtm_pd(pts,m=0.7, p=1):

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
    Given some point set, we create a Hybrid Rips/alpha complex.
    Args:
        pts: A list / numpy array / torch.tensor of coordinates in 2D or 3D.

    Returns:
        - Tensor of birth/death coordinates that represent a persistent diagram (diag)

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

def create_sublevel_kernel_pd(pts, nx, ny):
    """
    Given some point set, we create the sublevel filtration of the ker
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
def wasser_loss(pts, goal_pd, sliced=False, thetas=torch.tensor([k/4 * np.pi for k in range(5)]), filtr="alpha_rips_hybrid", max_edge_length=0.5, nx=16, ny=16):
    """
    Args:
        pts       : points of the dataset that we want to change into a dataset that 
                    has the same topological features of some other dataset
        goal_pd : persistence diagram of the dataset we want our initial point set 
                    to approximate in the sense that it has the same topological 
                    features.
                    IMPORTANT. If you use sliced method with float, the goal_pd's are assumed to be 1D.
        sliced    : if set to true, it will use the sliced wasserstein distance. If set to a float, 
                    you will get that percentage of the most persistent features after which we do a 
                    euclidean distance
        thetas    : the angles used for the sliced wasserstein distance
        max_edge_length
                  : length of the maximal length of an edge that we are willing to 
                    include in the filtration 

    Returns:
        - Loss which consists of the Wasserstein distance of initial PD and goal PD.

    NOTE: Currently the sliced wasserstein distance only supports distances between 1 dimensional pd's.
    """

    # Creation of persistene diagram
    if filtr == "alpha_rips_hybrid":
        diags = create_hybrid_pd(pts)
    elif filtr == "rips":
        diags = create_rips_pd(pts, max_edge_length=max_edge_length)
    elif filtr == "alpha_dtm":
        diags = create_hybrid_dtm_pd(pts)
    elif filtr == "line":
        diags = create_line_pd(pts)
    elif filtr == "sub_kernel":
        diags = create_sublevel_kernel_pd(pts, nx, ny)
    else:
        raise Exception("Please specify a filtration method that is known.\
                        The options are: alpha_rips_hybrid, rips, alpha_dtm, sub_kernel")
    diag0 = diags[0]
    diag1 = diags[1]
    
    goal_pd0 = goal_pd[0]
    goal_pd1 = goal_pd[1]


    # Total persistence is a special case of Wasserstein distance
    if sliced == False:
        # start_timer = timeit.default_timer() # START TIMER
        dist0 = wasserstein_distance(diag0, goal_pd0, order=1, enable_autodiff=True, keep_essential_parts=False)  
        dist1 = wasserstein_distance(diag1, goal_pd1, order=1, enable_autodiff=True, keep_essential_parts=False)  
        # print(f"0: {dist0}")
        # print(f"1: {dist1}")
        dist = dist0 + dist1
        # dist =  dist1
        # print(f"Wasserstein distance: {timeit.default_timer() - start_timer}") # END TIMER
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
        # NOTE: I actually want to make it max, but in those cases I wouldn't know how to calculate the distance
        zero_dim_amount = round(min(diag0.shape[0], goal_pd0.shape[0]) * sliced)
        one_dim_amount = round(min(diag1.shape[0], goal_pd1.shape[0]) * sliced)

        # Throwing out the non-persistent features
        # NOTE: Not doing max can result in a diagram with 0 features on both making the distance 0, which is not what we want.
        diag0 = diag0[:zero_dim_amount]
        goal_pd0 = goal_pd0[:zero_dim_amount]
        diag1 = diag1[:one_dim_amount]
        goal_pd1 = goal_pd1[:one_dim_amount]

        # Computing the distance
        dist0 = torch.linalg.norm(goal_pd0 - diag0)
        dist1 = torch.linalg.norm(goal_pd1 - diag1)

        dist = dist0 + dist1

        print(f"Topk distance: {timeit.default_timer() - start_timer}") # END TIMER
        return dist
    else:
        # dist = sliced_wasserstein_distance([diag, goal_pd], thetas[3])
        dists = [sliced_wasserstein_distance([diag1, goal_pd1], theta) for theta in thetas]
        dist = sum(dists)/len(dists)
        return dist

def generate_data(goal_pd, amount, dim, lr, epochs, decay_speed=10, investigate=False, sliced=False, thetas=torch.tensor([k/4 * np.pi for k in range(5)]), filtr="alpha_rips_hybrid", max_edge_length=0.5, box_size=1, init_pts="random", nx=16, ny=16):
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
        filtr               : what kind of filtration to use
        max_edge_length     : length of the maximal length of an edge that we are willing to 
                              include in the filtration
        box_size            : size of the box of the target point set
        init_pts            : initial point set that we want to change into some target point set
    
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


### ALPHA FILTRATIONS WITH SLICED WASSERSTEIN ###

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