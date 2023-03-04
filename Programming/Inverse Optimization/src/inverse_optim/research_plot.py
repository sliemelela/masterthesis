import matplotlib.pyplot as plt
from inverse_optim import gen_data
import torch
import numpy as np

def research_lr(lr_list, goal_pd, amount, dim, epochs, decay_speed=10, sliced=False, thetas=torch.tensor([k/4 * np.pi for k in range(5)]), filtr="alpha_rips_hybrid"):
    """
    Args:
        lr_list             : list of floats that represent learning rates that you would want to compare
        goal_pd             : persistence diagram that we want to 'approximate' in the sense that we want to 
                              generate a new dataset that approximately has the same persistence diagram
        amount (int)        : amount of data points
        dim (int)           : dimension of data points
        epochs (int)        : amount of learning steps
        decay_speed         : parameter that tunes how fast the learning rates decays (note that higer values correspond with 
                              lower decay speed).
        sliced              : if set to true, it will use the sliced wasserstein distance
        thetas              : the angles used for the sliced wasserstein distance
    
    Produces:
        - Plot of loss vs epochs for each initial learning rate that decays with some parameter decay_speed
    """

    for lr in lr_list:
        _, loss_list, _, _, _ = \
            gen_data.generate_data(goal_pd=goal_pd, amount=amount, dim=dim,\
                 lr=lr, epochs=epochs, decay_speed=decay_speed, investigate=True, sliced=sliced, thetas=thetas, filtr=filtr)
        
        # Loss research
        plt.plot(range(epochs), loss_list, label=f"lr = {lr}", alpha=0.9)

    # Loss research
    plt.title("Wasserstein distance between goal PD and initially random PD")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()

def research_move(lr, goal_pd, amount, dim, epochs, decay_speed=10, filtr="alpha_rips_hybrid"):
    """
    Args:
        lr                  : learning rate
        goal_pd             : persistence diagram that we want to 'approximate' in the sense that we want to 
                              generate a new dataset that approximately has the same persistence diagram
        amount (int)        : amount of data points
        dim (int)           : dimension of data points
        epochs (int)        : amount of learning steps
        decay_speed         : parameter that tunes how fast the learning rates decays (note that higer values correspond with 
                              lower decay speed).
    
    Produces:
        - Plots of max/min/avg movement (=|learning rate * gradient|) per point vs epochs
    """

    _, _, max_mag_list, min_mag_list, avg_mag_list = \
            gen_data.generate_data(goal_pd=goal_pd, amount=amount, dim=dim,\
                 lr=lr, epochs=epochs, decay_speed=decay_speed, investigate=True, filtr=filtr)

    plt.plot(range(epochs), max_mag_list, label=f"Max Mag, lr={lr}")
    plt.plot(range(epochs), min_mag_list, label=f"Min Mag, lr={lr}")
    plt.plot(range(epochs), avg_mag_list, label=f"Avg Mag, lr={lr}")

    # Mag research
    plt.title("Movement of points from initial PD")
    plt.xlabel("Epochs")
    plt.ylabel("Movement")
    plt.legend(loc="upper right")
    plt.show()