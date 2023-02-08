import matplotlib.pyplot as plt
import torch
import numpy as np
from inverse_optim import gen_data
from inverse_optim import research_plot
from inverse_optim import sancho
import tadasets
import tqdm


# Load dataset
cat = np.load(f'/Users/sliemela/Downloads/Sancho/fiducial_HOD_fid_NFW_sample0_1Gpc_z0.50_RSD3_run0.npz')
pos = cat['pos']        # shape: (N_galaxies, 3) --> X,Y,Z position of each galaxy in Mpc/h
vel = cat['vel']        # shape: (N_galaxies, 3) --> Vx, Vy, Vz velocity of the galaxy in km/s
gtype = cat['gtype']

# Split up the dataset
split = (2,2,2)
bins = sancho.bin(pos, split)

# Plotting the bins
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for x in bins:
    if len(x) != 0:
        ax.scatter3D(x[:, 0], x[:, 1], x[:, 2])

# Calculating the statistics of the wasserstein distances of sancho
list_of_wasser_dist = sancho.compare_wasser_alpha(bins)

wasser_mean = np.mean(list_of_wasser_dist)
wasser_std = np.std(list_of_wasser_dist)

print(wasser_mean)
print(wasser_std)

# NOTE: the code of compare_wasser_alpha has been changed to only consider the first 3000. In the future, we may consider all pairs. 