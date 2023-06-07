"""mult_algo_min.py

This module is used to define functions needed to find the best path in the bipersistence module with respect to some loss.
If you run this module, you will get a widget so you can see how the PD changes by changing the lines.

Todo:
    * Make faster
    * Implement a way to do line pieces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import torch
import gudhi as gd
import numpy as np
import torch
import math
import tadasets
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

# Code to create pd
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
        k = math.floor(m*N_tot)+1   # number of neighbors

        kdt = KDTree(X, leaf_size=30, metric='euclidean')
        NN_Dist, NN = kdt.query(query_pts, k, return_distance=True)  

        DTM_result = np.sqrt(np.sum(NN_Dist*NN_Dist,axis=1) / k)

        return DTM_result

def s(simplex, DTM_values):
    return np.max(DTM_values[simplex])

def orig_filt(simplex):
    if len(simplex) == 1:
        return 0
    else:
        simplex_coords = pts[simplex]
        try:
            # Points with the largest distance are in the convex hull 
            hull = ConvexHull(simplex_coords)

            # Extract the points forming the hull
            hullpoints = simplex_coords[hull.vertices,:]

            # Naive way of finding the best pair in O(H^2) time if H is number of points on hull
            hdist = cdist(hullpoints, hullpoints, metric='euclidean')

            # Get the farthest apart points
            bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

            # Calculate the distance between the furthest pair
            distance = np.linalg.norm(hullpoints[bestpair[0]] - hullpoints[bestpair[1]])

            return distance
        except:
        # Naive way of finding the best pair in O(H^2) time if H is number of points on hull
            alldist = cdist(simplex_coords, simplex_coords, metric='euclidean')

            # Get the farthest apart points
            bestpair = np.unravel_index(alldist.argmax(), alldist.shape)

            # Calculate the distance between the furthest pair
            distance = np.linalg.norm(simplex_coords[bestpair[0]] - simplex_coords[bestpair[1]])
            return distance

def filtration_function(alpha, beta, simplex, DTM_values):

    # The filtration function for 0 dimensional simplices (and infinity)
    if len(simplex) == 0:
        return np.inf

    # Assign filtration value to each simplex (see thesis)
    dens = s(simplex, DTM_values)
    orig_filt_value = orig_filt(simplex)
    if orig_filt_value >= 1/alpha * dens - beta/alpha:
        filtration = min(orig_filt_value, alpha*orig_filt_value)
        return filtration
    else:
        filtration = min(1/alpha * abs(dens - beta), abs(dens - beta))
        return filtration

def biline_rips_diag(alpha, beta, pts):

    # Density value for each vertex
    DTM_values = DTM(pts, pts, 0.01)

    # Build the rips/alpha complex
    complex = gd.AlphaComplex(points=pts)
    st_rips = complex.create_simplex_tree()

    # Creation of new simplex tree
    st = gd.SimplexTree()
    for simplex in st_rips.get_skeleton(2): 
        filtration = filtration_function(alpha, beta, simplex[0], DTM_values)
        st.insert(simplex[0], filtration=filtration)
    return st

def create_line_pd(alpha, beta, pts, m=0.5):

  # Creation of simplex tree
  st = biline_rips_diag(alpha, beta, pts)

  # Calculating persistence
  st.compute_persistence(2)
  p = st.persistence_pairs()

  # Creation of PD
  diag0 = []
  diag1 = []
  for simplex1, simplex2 in p:
    birth = filtration_function(alpha, beta, simplex1, DTM_values)
    death =  filtration_function(alpha, beta, simplex2, DTM_values)

    if len(simplex1) == 1:
      diag0.append((birth, death))
    elif len(simplex1) == 2:
      diag1.append((birth, death))
    else:
      print("yeh bg")
  
  diag0 = np.array(diag0)
  diag1 = np.array(diag1)

  return [diag0, diag1]

##########
# pts = np.array([[1,2], [3,4], [2,5], [2,7], [3,7]])
pts = tadasets.sphere(n=20, r=1, noise=0.4)
DTM_values = DTM(pts, pts, 0.5)

# Define initial parameters
init_alpha = 1
init_beta = 0

# Create the figure and the line that we will manipulate
fig, (ax1, ax2) = plt.subplots(1,2)
diags = create_line_pd(init_alpha, init_beta, pts)

# Limits and diagonal line
xlim = 10
ylim = 10
t = np.linspace(0, xlim, 100)
ax1.plot(t, t, color="k")

# Persistence Diagram
ax1.set_title('Persistence Diagram')
ax1.scatter(diags[0][:, 0],diags[0][:, 1], label="H0")
if len(diags[1]) != 0:
    ax1.scatter(diags[1][:, 0],diags[1][:, 1], label="H1")
ax1.set_xlabel('Birth time')
ax1.set_ylabel('Death time')
ax1.legend()
ax1.set_xlim(0, xlim)
ax1.set_ylim(0, ylim)
ax1.grid(True)
ax1.plot()

# Line in bipersistence diagram
ax2.plot(t, init_alpha*t + init_beta)
ax2.set_title('Bipersistence Module')
ax2.grid(True)
ax2.set_xlim(0, xlim)
ax2.set_ylim(0, ylim)
ax2.set_xlabel("Radius of ball")
ax2.set_ylabel("Density")

# Adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
alpha_list = fig.add_axes([0.25, 0.1, 0.65, 0.03])
alpha_slider = Slider(
    ax=alpha_list,
    label='Slope (Alpha)',
    valmin=0.1,
    valmax=30,
    valinit=init_alpha,
)

# Make a vertically oriented slider to control the amplitude
beta_list = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
beta_slider = Slider(
    ax=beta_list,
    label="Intercept (Beta)",
    valmin=0,
    valmax=80,
    valinit=init_beta,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):

    # Clearing the old datapoints
    ax1.clear()
    ax2.clear()

    # Creation of PD
    diags = create_line_pd(alpha_slider.val, beta_slider.val, pts)

    ax1.set_title('Persistence Diagram')
    ax1.plot(t,t,color="k")
    ax1.scatter(diags[0][:, 0],diags[0][:, 1], label="H0")
    if len(diags[1]) != 0:
        ax1.scatter(diags[1][:, 0],diags[1][:, 1], label="H1")
    ax1.set_xlim(0, xlim)
    ax1.set_ylim(0, ylim)
    ax1.set_xlabel('Birth time')
    ax1.set_ylabel('Death time')
    ax1.legend()
    ax1.grid(True)
    fig.canvas.draw_idle()

    # Creation of line in bipersistence diagram
    ax2.set_title('Bipersistence Module')
    ax2.plot(t, alpha_slider.val*t + beta_slider.val)
    ax2.grid(True)
    ax2.set_xlim(0, xlim)
    ax2.set_ylim(0, ylim)
    ax2.set_xlabel("Radius of ball")
    ax2.set_ylabel("Density")


# register the update function with each slider
alpha_slider.on_changed(update)
beta_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    alpha_slider.reset()
    beta_slider.reset()
button.on_clicked(reset)



plt.show()