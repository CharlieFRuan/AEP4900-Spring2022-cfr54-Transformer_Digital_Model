"""
Generate an originally T x N x P x 2 dataset; by default T=60, N=5000, P=10
Transposed into N x T x 2P, which in default is 5000 x 60 x 20
Each of the N dataset is a dictionary entry for the hdf5 file, key is index, value is the 60x20 data

We create two hdf5 files, one for training and one for testing, both of the same size.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from exp import CoupledPendula
import h5py
import argparse
import os

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

def gen_n_pendula(n_samples, n_pendula, data_dir, file_name, sanity_check):
    # 1. First generate the training data
    theta = simulate_pendula(n_samples, n_pendula, sanity_check=sanity_check)
    # We want to transpose the T x N x P x 2 into  N x T x 2P
    theta = np.transpose(theta, (1, 0, 2, 3)).reshape(n_samples, 60, 20)
    
    file_add = os.path.join(data_dir, 'train_'+file_name)
    with h5py.File(file_add, 'w') as f:
        for i in range(n_samples):
            # we save each data (each initial condition) as a value for a key of the hdf5 file
            dset = f.create_dataset(str(i), data=theta[i])
        f.close()
    print("created data {}".format('train_'+file_name))

    # 2. Then generate the testing data seperately
    theta = simulate_pendula(n_samples, n_pendula, sanity_check=sanity_check)
    # We want to transpose the T x N x P x 2 into  N x T x 2P
    theta = np.transpose(theta, (1, 0, 2, 3)).reshape(n_samples, 60, 20)
    
    file_add = os.path.join(data_dir, 'test_'+file_name)
    with h5py.File(file_add, 'w') as f:
        for i in range(n_samples):
            # we save each data (each initial condition) as a value for a key of the hdf5 file
            dset = f.create_dataset(str(i), data=theta[i])
        f.close()
    print("created data {}".format('test_'+file_name))

def get_args():
    parser = argparse.ArgumentParser(description='Pendulum Data Generation')
    
    # data generation
    parser.add_argument('--sanity_check', type=bool, default=False, help='weather generate easy data')
    parser.add_argument('--n_samples', type=int, default=5000, help='number of data (number of initial conditions')
    parser.add_argument('--n_pendula', type=int, default=10, help='number of pendula for system')

    # file saving
    parser.add_argument('--data_dir', type=str, default='/Users/charlieruan/projects_data/McMahon/data', help='directory to save generated data')
    parser.add_argument('--file_name', type=str, help='directory to save generated data')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    gen_n_pendula(args.n_samples, args.n_pendula, args.data_dir, args.file_name, args.sanity_check)
