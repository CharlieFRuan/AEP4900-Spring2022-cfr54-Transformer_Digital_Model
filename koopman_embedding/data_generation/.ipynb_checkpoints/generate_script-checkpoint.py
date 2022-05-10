import torch
import numpy as np
import matplotlib.pyplot as plt
from exp import CoupledPendula
import h5py

def simulate_pendula(n_samples, n_pendula, sanity_check = False):
    # simulate a chain of coupled pendula with the following parameters
    # the parameters are explained in the exp.py file
    pend = CoupledPendula(Tmax = 2)
    
    ω0 = torch.tensor([10.]*n_pendula).repeat(n_samples,1)
    ωd = torch.tensor([0.]*n_pendula).repeat(n_samples,1)
    Ad = torch.tensor([0.]*n_pendula).repeat(n_samples,1)
    v0 = torch.tensor([0.]*n_pendula).repeat(n_samples,1)
    coupling = torch.tensor([10.]*n_pendula).repeat(n_samples,1)
    γ = torch.tensor([0.]*n_pendula).repeat(n_samples,1)
    encoding_amplitude = torch.tensor(1.).repeat(n_samples,1)
    phid = torch.tensor([0.]*n_pendula).repeat(n_samples,1)

    # set initial angles (in rad)
    if not sanity_check:
        # i.e. initial theta uniformly random between -pi/2 to pi/2
        theta_initial = torch.rand([n_samples,n_pendula])*np.pi - np.pi/2
    else:
        # i.e. initial theta uniformly random between -pi/10 and pi/10
        theta_initial = (torch.rand([n_samples,n_pendula])*np.pi - np.pi/2) / 5

    # propagate through setup. The "full=True" parameter determines that the full
    # time evolution of the pendula is returned instead of just the final angle.
    theta = pend(theta_initial, ω0, ωd, Ad, v0, coupling, γ, encoding_amplitude, phid, full = True)
    return theta

def gen_n_pendula(n_samples, n_pendula, sanity_check=False):
    theta = simulate_pendula(n_samples, n_pendula, sanity_check=sanity_check)
    with h5py.File('pendula_{}.hdf5'.format(n_pendula), 'w') as f:
        dset = f.create_dataset('pendula_data', data=theta)
    print("created data pendula_{}.hdf5".format(n_pendula))
