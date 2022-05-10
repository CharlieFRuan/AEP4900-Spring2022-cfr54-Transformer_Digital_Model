import os
import matplotlib
import torch
import numpy as np
from typing import Optional

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

from viz_model import Viz

Tensor = torch.Tensor

class TenPendulaViz(Viz):
    """
    Visualization class for 10 Pendula System

    Args:
        plot_dir (str, optional): Directory to save visualizations in. Defaults to None.
    """
    def __init__(self, plot_dir: str = None) -> None:
        super().__init__(plot_dir=plot_dir)

    def plotPrediction(self,
        y_pred: Tensor,
        y_target: Tensor,
        plot_dir: str = None,
        epoch: int = None,
        pid: int = 0
    ) -> None:
        """Plots a 3D line of a single Lorenz prediction

        Args:
            y_pred (Tensor): [T, 3] Prediction tensor.
            y_target (Tensor): [T, 3] Target tensor.
            plot_dir (str, optional): Directory to save figure, overrides plot_dir one if provided. Defaults to None.
            epoch (int, optional): Current epoch, used for file name. Defaults to None.
            pid (int, optional): Optional plotting id for indexing file name manually. Defaults to 0.
        """

        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        y_target = y_target.detach().cpu().numpy()

        plt.close('all')

        # 1. first plot position
        fig, axs = plt.subplots(nrows = 5, ncols = 2, figsize = [10,8], dpi = 200, sharex=True, sharey=True)
        for i_loop in range(10):
            i = i_loop * 2 # since we only want the position features
            plt.sca(axs.flatten()[i_loop])
            plt.plot(y_target[:,i], '.-', lw = 1, c = 'k', alpha = 0.5, label = 'ground truth')
            plt.plot(y_pred[:,i], '.-', lw = 1, c = 'b', alpha = 0.5, label = 'NN prediction')
            plt.xlabel('Timestep')
            plt.ylabel('Position')
            plt.title(f'Pendulum {i_loop}')
            plt.grid()
        fig.suptitle("Position Comparison")
        plt.legend()
        plt.tight_layout()
        if(not epoch is None):
            file_name = 'tenPendula_pos{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'tenPendula_pos{:d}'.format(pid)
        self.saveFigure(plot_dir, file_name)
        
        plt.close('all')
        # 2. then plot velocity
        fig, axs = plt.subplots(nrows = 5, ncols = 2, figsize = [10,8], dpi = 200, sharex=True, sharey=True)
        for i_loop in range(10):
            i = i_loop * 2 + 1 # since we only want the velocity features
            plt.sca(axs.flatten()[i_loop])
            plt.plot(y_target[:,i], '.-', lw = 1, c = 'k', alpha = 0.5, label = 'ground truth')
            plt.plot(y_pred[:,i], '.-', lw = 1, c = 'b', alpha = 0.5, label = 'NN prediction')
            plt.xlabel('Timestep')
            plt.ylabel('Velocity')
            plt.title(f'Pendulum {i_loop}')
            plt.grid()
        fig.suptitle("Velocity Comparison")
        plt.legend()
        plt.tight_layout()
        if(not epoch is None):
            file_name = 'tenPendula_vel{:d}_{:d}'.format(pid, epoch)
        else:
            file_name = 'tenPendula_vel{:d}'.format(pid)
        self.saveFigure(plot_dir, file_name)