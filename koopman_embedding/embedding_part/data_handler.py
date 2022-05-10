import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Dict, Tuple

class PendulumDataHandler(object):
    """
    Handles the hdf5 data that generate_script creates
    """
    class PendulumDataset(Dataset):
        """
        A custum dataset using torch's dataset, for loading pendulum data
        """
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {"states": self.data[i]}
    
    def createTrainingLoader(self, 
        file_path: str,  #hdf5 file
        # block_size: int, # Length of time-series
        ndata: int = 2048, # number of initial conditions we want
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:

        examples = []
        cur_num = 0
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                data_series = torch.Tensor(np.array(f[key]))
                examples.append(data_series.unsqueeze(0))
                cur_num += 1
                if (cur_num >= ndata):
                    break
        data = torch.cat(examples, dim=0)
        self.mu = torch.mean(data, dim=(0,1)) # mean for each state across data and time
        self.std = torch.std(data, dim=(0,1)) # std for each state across data and time

        dataset = self.PendulumDataset(data)
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # now when we enumerate through the loader, it returns a [B, 60, 20], where B is batchsize
        return training_loader

    def createTestingLoader(self, 
        file_path: str,  #hdf5 file
        # block_size: int, # Length of time-series
        ndata: int = 128, # number of initial conditions we want
        batch_size: int = 32,
        shuffle: bool = False,
    ) -> DataLoader:
        # basically the same thing
        examples = []
        cur_num = 0
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                data_series = torch.Tensor(np.array(f[key]))
                examples.append(data_series.unsqueeze(0))
                cur_num += 1
                if (cur_num >= ndata):
                    break
        data = torch.cat(examples, dim=0)

        dataset = self.PendulumDataset(data)
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # now when we enumerate through the loader, it returns a [B, 60, 20], where B is batchsize
        return testing_loader

    @property
    def norm_params(self) -> Tuple:
        """Get normalization parameters

        Raises:
            ValueError: If normalization parameters have not been initialized

        Returns:
            (Tuple): mean and standard deviation
        """
        if self.mu is None or self.std is None:
            raise ValueError("Normalization constants set yet!")
        return self.mu, self.std