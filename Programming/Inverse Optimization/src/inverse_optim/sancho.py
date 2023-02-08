import numpy as np
from scipy import sparse
from inverse_optim import gen_data
from gudhi.wasserstein import wasserstein_distance
from tqdm import tqdm

def sort_to_bins_sparse(idx, data, mx=-1):
    if mx == -1:
        mx = idx.max() + 1    
    aux = sparse.csr_matrix((data, idx, np.arange(len(idx)+1)), (len(idx), mx)).tocsc()
    return np.split(aux.data, aux.indptr[1:]), \
        np.split(aux.indices, aux.indptr[1:])

def bin(data, bincounts):
    """
    Args:
        data       : dataset that we want to split up. The format of the dataset is a matrix where each 
                     row corresponds to a coordinate.
        bincounts  : how you want to split the dataset in terms per axis, 
                     e.g. (8,3,2) splits x-axis in 8 parts, y-axis in 3 parts 
                     and z-axis in 2 parts. If the set is higher or lower dimensional, 
                     simply pass a lower dimensional tuple.

    Returns:
        - List of matrices with rows corresponding to the coordinates of the point, 
          where each matrix represents the bin the coordinates fall in.
    """
    data = np.asanyarray(data)
    data = np.transpose(data)
    idx = [np.digitize(d, np.linspace(d.min(), d.max(), b, endpoint=False)) - 1
           for d, b in zip(data, bincounts)]
    flat = np.ravel_multi_index(idx, bincounts)
    _, idx = sort_to_bins_sparse(flat, data[0])
    bins = [data[:,i] for i in idx]
    for i,_ in enumerate(bins):
        bins[i] = np.array(bins[i])
        bins[i] = np.transpose(bins[i])
    return bins

def compare_wasser_alpha(bins):
    """
    Args:
        bins: list of matrices where each matrix represents a point set with the rows being 
              the coordinates of points 
    
    Returns:
        - List of Wasserstein distance between persistence diagrams of pairs in bins created by 
          alpha filtrations
    """

    # Creating all unique pairs of all bins (where the order (x,y) and (y,x) are identified)
    # and also where the empty bins are ignored
    bin_pairs = [(bins[p1], bins[p2]) for p1 in range(len(bins)) if len(bins[p1] !=0) for p2 in range(p1+1,len(bins)) if len(bins[p2] !=0)]

    list_wasserstein_distances = []
    for bin1, bin2 in tqdm(bin_pairs[:3000]):

        # Ensuring the size of the datasets are large by removing random data points of the larger set
        rng = np.random.default_rng()
        if len(bin1) > len(bin2):
            bin1 = rng.choice(bin1, len(bin2))
        elif len(bin2) > len(bin1):
            bin2 = rng.choice(bin2, len(bin1))

        # Creation of pd's
        diag1 = gen_data.create_alpha_pd(bin1)
        diag2 = gen_data.create_alpha_pd(bin2)

        # Calculating the wasserstein distance
        was_dist = gen_data.wasserstein_distance(diag1, diag2, order=1, enable_autodiff=True, keep_essential_parts=False)
        list_wasserstein_distances.append(was_dist)

    return list_wasserstein_distances
