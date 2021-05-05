"""
Collection of shower metrics for plotting.

Defines function to calculate specific metric.
"""
import torch

def sparsity(data):
    return total_energy(data)

def depth():
    """Return count of layers with an energy deposit.
    """
    # maxdepth = 2 * (d['layer_2'][:].sum(axis=(1,2)) != 0)
    # maxdepth[maxdepth == 0] = 1 * (d['layer_1'][:][maxdepth == 0].sum(axis=(1,2)) != 0)
    # return maxdepth
    return

def total_energy(data):
    """Total deposited energy per event"""
    tot_energy=0
    for layer in range(3):
        tot_energy+=energy(data,layer)
    return tot_energy

def energy(data, layer):
    """Deposited energy per layer per event"""
    return torch.sum(data[layer],dim=(1,2))
# def efrac(elayer, total_energy):
        #     return  elayer / total_energy


def lateral_depth(d):
    # '''
    # Sum_{i} E_i * d_i
    # '''
    # return (d['layer_2'][:] * 2).sum(axis=(1,2)) + (d['layer_1'][:]).sum(axis=(1,2))
    return

# def lateral_depth2(d):
#     '''
#     Sum_{i} E_i * d_i^2
#     '''
#     return (d['layer_2'][:] * 2 * 2).sum(axis=(1,2)) + (d['layer_1'][:]).sum(axis=(1,2))


# def shower_depth(lateral_depth, total_energy):
#     '''
#     lateral_depth / total_energy
#     Args:
#     -----
# lateral_depth: float, Sum_{i} E_i * d_i
# total_energy: float, total energy per event
#     '''
#     return lateral_depth / total_energy


# def shower_depth_width(lateral_depth, lateral_depth2, total_energy):
#     '''
#     sqrt[lateral_depth2 / total_energy - (lateral_depth / total_energy)^2]
#     Args:
#     -----
# lateral_depth: float, Sum_{i} E_i * d_i
# lateral_depth2: float, Sum_{i} E_i * d_i * d_i
# total_energy: float, total energy per event
#     '''
#     return np.sqrt((lateral_depth2 / total_energy) - (lateral_depth / total_energy)**2)


# def layer_lateral_width(layer, d):
#     '''
#     Args:
#     -----
# layer: int in {0, 1, 2} that labels the layer
# d: an h5py File with fields 'layer_2', 'layer_1', 'layer_0'
#    that represent the 2d cell grids and the corresponding
#    E depositons.
#     '''
#     e = energy(layer, d)
#     eta_cells = {'layer_0' : 3, 'layer_1' : 12, 'layer_2' : 12}
#     eta_bins = np.linspace(-240, 240, eta_cells['layer_' + str(layer)] + 1)
#     bin_centers = (eta_bins[1:] + eta_bins[:-1]) / 2.
#     x = (d['layer_{}'.format(layer)] * bin_centers.reshape(-1, 1)).sum(axis=(1,2))
#     x2 = (d['layer_{}'.format(layer)] * (bin_centers.reshape(-1, 1) ** 2)).sum(axis=(1,2))
#     return np.sqrt((x2 / (e+myepsilon)) - (x / (e+myepsilon)) ** 2)

# def eratio(images):
#     top2 = np.array([np.sort(row.ravel())[::-1][:2] for row in images])
#     return (top2[:, 0] - top2[:, 1]) / (myepsilon+top2[:, 0] + top2[:, 1])
