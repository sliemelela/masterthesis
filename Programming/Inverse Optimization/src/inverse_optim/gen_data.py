import torch
from torch.optim.lr_scheduler import LambdaLR
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import tadasets

### ALPHA FILTRATIONS ###

def create_alpha_pd(pts):
    """
    Given some point set, we create a Hybrid Rips/alpha complex.
    Args:
        pts: A list / numpy array / torch.tensor of coordinates in 2D or 3D.

    Returns:
        - Tensor of birth/death coordinates that represent a persistent diagram (diag)

    """

    # Ensuring points are tensor
    if type(pts) != torch.Tensor:
        pts = torch.tensor(pts)
    
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

def myloss_alpha(pts, goal_pd, sliced=False, thetas=torch.tensor([k/4 * np.pi for k in range(5)])):
    """
    Args:
        pts       : points of the dataset that we want to change into a dataset that 
                    has the same topological features of some other dataset
        goal_diag : persistence diagram of the dataset we want our initial point set 
                    to approximate in the sense that it has the same topological 
                    features
        sliced    : if set to true, it will use the sliced wasserstein distance
        thetas    : the angles used for the sliced wasserstein distance

    Returns:
        - Loss which consists of the Wasserstein distance of initial PD and goal PD,
          and which penalizes the points going outside of the circle (loss).
    """

    # Creation of persistene diagram
    diags = create_alpha_pd(pts)
    diag0 = diags[0]
    diag1 = diags[1]

    goal_pd0 = goal_pd[0]
    goal_pd1 = goal_pd[1]
    # Total persistence is a special case of Wasserstein distance
    if sliced == False:
        dist0 = wasserstein_distance(diag0, goal_pd0, order=1, enable_autodiff=True, keep_essential_parts=False)  
        dist1 = wasserstein_distance(diag1, goal_pd1, order=1, enable_autodiff=True, keep_essential_parts=False)  
        dist = dist0 + dist1
        return dist
    else:
        # dist = sliced_wasserstein_distance([diag, goal_pd], thetas[3])
        dists = [sliced_wasserstein_distance([diag1, goal_pd1], theta) for theta in thetas]
        dist = sum(dists)/len(dists)
        return dist

def generate_data_alpha(goal_pd, amount, dim, lr, epochs, decay_speed=10, investigate=False, sliced=False, thetas=torch.tensor([k/4 * np.pi for k in range(5)])):
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
    
    Returns:
        - The generated dataset
        - A list of the losses at each epoch (if investigate_loss==True)
        - Lists of the magnitude of the avg/max/min movement (lr*gradient) movement of points
        - Plots of the generated dataset every 100 epochs including the final epoch 
    """

    pts = (torch.rand((amount, dim))).requires_grad_()

    # Set up optimizer for SGD
    opt = torch.optim.SGD([pts], lr=lr)
    scheduler = LambdaLR(opt,[lambda epoch: decay_speed/(decay_speed+epoch)])

    # Initialize loss list
    if investigate:
        loss_list = []
        max_mag_list = []
        min_mag_list = []
        avg_mag_list = []
    
    if investigate == False:
        # Plot initial point cloud
        P = pts.detach().numpy()
        plt.scatter(P[:, 0], P[:, 1])
        plt.show()

    # Perform SGD
    for epoch in tqdm(range(epochs)):
        opt.zero_grad()
        myloss_alpha(pts, goal_pd, sliced, thetas).backward()

        if investigate:
            loss_list.append(myloss_alpha(pts, goal_pd, sliced, thetas).item())
            
        opt.step()
        scheduler.step()
        
    

        if investigate:
            movement = pts.grad * scheduler.get_last_lr()[0]
            movement_mag = torch.zeros(amount)

            for i in range(amount):
                movement_mag[i] = torch.linalg.norm(movement[i])
            
            max_mag = torch.max(movement_mag)
            max_mag_list.append(max_mag)
            min_mag = torch.min(movement_mag)
            min_mag_list.append(min_mag)
            avg_mag = torch.mean(movement_mag) 
            avg_mag_list.append(avg_mag)

        if investigate == False:
            # Draw every 100 epochs
            if epoch % 100 == 99:
                P = pts.detach().numpy()
                plt.scatter(P[:, 0], P[:, 1])
                plt.show()

    
    if investigate:
        return pts, loss_list, max_mag_list, min_mag_list, avg_mag_list
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




### Periodicity tryout
def per(x,y):
    z = torch.fmod(x,y)
    return z.sum()

def per_generate_data_alpha(goal_pd, amount, dim, lr, epochs, per1, per2, decay_speed=10, investigate=False, sliced=False, thetas=torch.tensor([k/4 * np.pi for k in range(5)])):
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
    
    Returns:
        - The generated dataset
        - A list of the losses at each epoch (if investigate_loss==True)
        - Lists of the magnitude of the avg/max/min movement (lr*gradient) movement of points
        - Plots of the generated dataset every 100 epochs including the final epoch 
    """

    # Creation of initial random dataset
    pts = (torch.rand((amount, dim)))
    pts = pts * per1
    pts.requires_grad_()

    box_size = torch.tensor([per1, per1])
    # Set up optimizer for SGD
    opt = torch.optim.SGD([pts], lr=lr)
    scheduler = LambdaLR(opt,[lambda epoch: decay_speed/(decay_speed+epoch)])

    # Initialize loss list
    if investigate:
        loss_list = []
        max_mag_list = []
        min_mag_list = []
        avg_mag_list = []
    
    # # Plot initial point cloud
    # P = pts.detach().numpy()
    # plt.scatter(P[:, 0], P[:, 1])
    # plt.show()

    # Perform SGD
    for epoch in tqdm(range(epochs)):
        opt.zero_grad()
        myloss_alpha(pts, goal_pd, sliced, thetas).backward()
        # per(pts, per1).backward()

        if investigate:
            loss_list.append(myloss_alpha(pts, goal_pd, sliced, thetas).item())
            
        # Update the coordinates of the points
        with torch.no_grad():
            # # Apply periodic boundary conditions
            # pts_update = pts - opt.param_groups[0]['lr'] * pts.grad
            # pts_update = (pts_update + box_size/2) % box_size - box_size/2

            # # Update the parameters with the modified updates
            # opt.step(pts_update - pts)

            pts -= scheduler.get_last_lr()[0] * pts.grad

            # Apply periodic boundary conditions
            pts = pts % per1
            # pts = torch.fmod(pts, per1)
            # pts = torch.fmod((pts + box_size/2), box_size) - box_size/2

            pts.requires_grad_()

            # Zero out the gradients
            # pts.grad.zero_()
            
        # opt.step()
        scheduler.step()

        # pts = pts % per1
        
        # print(pts.requires_grad)

        # if investigate:
        #     movement = pts.grad * scheduler.get_last_lr()[0]
        #     movement_mag = torch.zeros(amount)

        #     for i in range(amount):
        #         movement_mag[i] = torch.linalg.norm(movement[i])
            
        #     max_mag = torch.max(movement_mag)
        #     max_mag_list.append(max_mag)
        #     min_mag = torch.min(movement_mag)
        #     min_mag_list.append(min_mag)
        #     avg_mag = torch.mean(movement_mag) 
        #     avg_mag_list.append(avg_mag)

        if investigate == False:
            # Draw every 100 epochs
            if epoch % 100 == 99:
                P = pts.detach().numpy()
                plt.scatter(P[:, 0], P[:, 1])
                plt.show()

    
    if investigate:
        return pts, loss_list, max_mag_list, min_mag_list, avg_mag_list
    return pts





# class InverseOptimization:
    
#     def __init__(self, pts, filtr_type):
#         """
#         Args:
#             pts: A list / numpy array / torch.tensor of coordinates in 2D or 3D. 
#             filtr_type: String specifying by which method you want to create the persistence diagrams.
#                 Currently only "alpha" is available.
#         """
#         self.pts = pts
#         self.filtr_type
    

#     def create_pd(self):
#         """
#         Given some point set, we create a Hybrid Rips/alpha complex.
           
#         Creates:
#             - Tensor of birth/death coordinates that represent a persistent diagram (goal_pd)
#         """
#         if self.filtr_type == "alpha":
#             # Creation of simplex tree
#             alpha_complex = gd.AlphaComplex(points=self.pts)
#             alpha_st = alpha_complex.create_simplex_tree()

#             # Calculating persistence
#             alpha_st.compute_persistence(2)
#             p = alpha_st.persistence_pairs()

#             # Keep only pairs that contribute to H1, i.e. (edge, triangle), and separate birth (p1b) and death (p1d)
#             p1b = torch.tensor([i[0] for i in p if len(i[0]) == 2])
#             p1d = torch.tensor([i[1] for i in p if len(i[0]) == 2])

#             # Compute the distance between the extremities of the birth edge
#             if len(p1b) == 0 and len(p1d) == 0:
#                 diag = torch.tensor([])
#             else:
#                 b = torch.norm(self.pts[p1b[:,1]] - self.pts[p1b[:,0]], dim=-1, keepdim=True)
#                 # For the death triangle, compute the maximum of the pairwise distances
#                 d_1 = torch.norm(self.pts[p1d[:,1]] - self.pts[p1d[:,0]], dim=-1, keepdim=True)
#                 d_2 = torch.norm(self.pts[p1d[:,1]] - self.pts[p1d[:,2]], dim=-1, keepdim=True)
#                 d_3 = torch.norm(self.pts[p1d[:,2]] - self.pts[p1d[:,0]], dim=-1, keepdim=True)
#                 d = torch.max(d_1, torch.max(d_2, d_3))

#                 # *Not* the same as the finite part of st.persistence_intervals_in_dimension(1)
#                 diag = torch.cat((b,d), 1)
            
#             self.goal_pd = diag
#         else:
#             raise Exception("At the moment, this method is not supported")
            
    
#     def wass_loss(self, diag, order=1, enable_autodiff=True, keep_essential_parts=False):
#         dist = wasserstein_distance(diag, self.goal_pd, order=order, enable_autodiff=enable_autodiff, keep_essential_parts=keep_essential_parts)  
#         return dist
        


