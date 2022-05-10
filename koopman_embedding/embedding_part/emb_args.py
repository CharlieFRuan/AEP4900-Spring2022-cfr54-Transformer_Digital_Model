"""
A class that manages the arguments for embedding part.
"""

import numpy as np
import argparse 
import random
import torch
import json, errno, os
import logging

HOME = os.getcwd()
logger = logging.getLogger(__name__)

class EmbeddingArgs(argparse.ArgumentParser):
    """
    Arguments for training embedding models
    """
    def __init__(self):
        super().__init__(description='Arguments for training the embedding models')
        self.add_argument('--output_dir', type=str, default="./outputs", help='directory to save experiments')
        self.add_argument('--exp_name', type=str, default="10_pendula", help='experiment name')        

        # data
        self.add_argument('--training_h5_file', type=str, default=None, help='file path to the training data hdf5 file')
        self.add_argument('--eval_h5_file', type=str, default=None, help='file path to the evaluation data hdf5 file')
        self.add_argument('--n_train', type=int, default=2048, help='number of training data')
        self.add_argument('--n_eval', type=int, default=128, help='number of testing data')
        self.add_argument('--stride', type=int, default=1, help='take all timesteps, or take 1 from every 3 timesteps, etc.')
        self.add_argument('--block_size', type=int, default=60, help='number of time-steps as encoder input')
        self.add_argument('--batch_size', type=int, default=64, help='batch size for training')

        # training
        self.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate')
        self.add_argument('--seed', type=int, default=12345, help='manual seed used in PyTorch and Numpy')
        
        # logging
        self.add_argument('--plot_freq', type=int, default=25, help='how many epochs to wait before plotting test output')
        self.add_argument('--test_freq', type=int, default=10, help='how many epochs to test the model')
        self.add_argument('--save_steps', type=int, default=100, help='how many epochs to wait before saving model')

    def mkdirs(self, *directories: str) -> None:
        """Makes a directory if it does not exist

        Args:
           directories (str...): a sequence of directories to create

        Raises:
            OSError: if directory cannot be created
        """
        for directory in list(directories):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def parse(self):
        args = self.parse_args()
        logger.info(HOME)

        # Set random seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

        args.run_dir = os.path.join(HOME, args.output_dir, "embedding_{}".format(args.exp_name), 
                    "ntrain{}_epochs{:d}_batch{:d}".format(args.n_train, args.epochs, args.batch_size))
        args.ckpt_dir = os.path.join(args.run_dir,"checkpoints")
        args.plot_dir = os.path.join(args.run_dir, "predictions")

        self.mkdirs(args.run_dir, args.ckpt_dir, args.plot_dir)
        with open(os.path.join(args.run_dir, "args.json"), 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        return args


