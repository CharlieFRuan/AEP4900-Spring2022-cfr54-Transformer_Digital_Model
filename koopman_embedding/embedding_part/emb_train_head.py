import torch
import torch.nn as nn
import numpy as np
import logging
from torch.autograd import Variable
import os
from typing import Tuple
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
# Custom types
Tensor = torch.Tensor
TensorTuple = Tuple[torch.Tensor]
FloatTuple = Tuple[float]

class EmbeddingModel(nn.Module):
    """
    Embedding NN model, responsible for one timestep processing
    """
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.model_name = "embedding_model"

        self.observableNet = nn.Sequential(
            nn.Linear(config.state_dims[0], config.hidden_states),
            nn.ReLU(),
            nn.Linear(config.hidden_states, config.n_embd),
            nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
            nn.Dropout(config.embd_pdrop)
        )
        
        self.recoveryNet = nn.Sequential(
            nn.Linear(config.n_embd, config.hidden_states),
            nn.ReLU(),
            nn.Linear(config.hidden_states, config.state_dims[0])
        )
        
        # added a layer 4/21 Charlie
#         self.observableNet = nn.Sequential(
#             nn.Linear(config.state_dims[0], 714),
#             nn.ReLU(),
#             nn.Linear(714, 605),
#             nn.ReLU(),
#             nn.Linear(605, 417),
#             nn.ReLU(),
#             nn.Linear(417, config.n_embd),
#             nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
#             nn.Dropout(config.embd_pdrop)
#         )
        
#         self.recoveryNet = nn.Sequential(
#             nn.Linear(config.n_embd, 417),
#             nn.ReLU(),
#             nn.Linear(417, 605),
#             nn.ReLU(),
#             nn.Linear(605, 714),
#             nn.ReLU(),
#             nn.Linear(714, config.state_dims[0])
#         )

        # Learned Koopman operator
        self.obsdim = config.n_embd
        self.kMatrixDiag = nn.Parameter(torch.linspace(1, 0, config.n_embd)) # initialize K Operator

        # Off-diagonal indices; did not quite get this part
        xidx = []
        yidx = []
        for i in range(1, 3):
            yidx.append(np.arange(i, config.n_embd))
            xidx.append(np.arange(0, config.n_embd-i))

        self.xidx = torch.LongTensor(np.concatenate(xidx))
        self.yidx = torch.LongTensor(np.concatenate(yidx))
        self.kMatrixUT = nn.Parameter(0.1*torch.rand(self.xidx.size(0)))

        # Normalization occurs inside the model
        self.register_buffer('mu', torch.tensor([0.] * 20)) # for 10-pendula
        self.register_buffer('std', torch.tensor([1.] * 20)) # for 10-pendula
        # self.register_buffer('mu', torch.tensor([0.] * 3)) # for lorenz
        # self.register_buffer('std', torch.tensor([1.] * 3)) # for lorenz
        logger.info('Number of embedding parameters: {}'.format(self.num_parameters))

    def forward(self, x: Tensor) -> TensorTuple:
        """Forward pass

        Args:
            x (Tensor): [B, 3] Input feature tensor

        Returns:
            TensorTuple: Tuple containing:

                | (Tensor): [B, config.n_embd] Koopman observables
                | (Tensor): [B, 3] Recovered feature tensor
        """
        # only checks whether observableNet and recoveryNet are inverse function to each other
        # this step does not use Koopman
        # Encode
        x = self._normalize(x)
        g = self.observableNet(x)
        # Decode
        out = self.recoveryNet(g)
        xhat = self._unnormalize(out)
        return g, xhat

    def embed(self, x: Tensor) -> Tensor:
        """Embeds tensor of state variables to Koopman observables

        Args:
            x (Tensor): [B, 3] Input feature tensor

        Returns:
            Tensor: [B, config.n_embd] Koopman observables
        """
        x = self._normalize(x)
        g = self.observableNet(x)
        return g

    def recover(self, g: Tensor) -> Tensor:
        """Recovers feature tensor from Koopman observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables

        Returns:
            Tensor: [B, 3] Physical feature tensor
        """
        out = self.recoveryNet(g)
        x = self._unnormalize(out)
        return x

    def koopmanOperation(self, g: Tensor) -> Tensor:
        """Applies the learned Koopman operator on the given observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables

        Returns:
            (Tensor): [B, config.n_embd] Koopman observables at the next time-step
        """
        # Koopman operator
        kMatrix = Variable(torch.zeros(self.obsdim, self.obsdim)).to(self.kMatrixUT.device)
        # Populate the off diagonal terms
        kMatrix[self.xidx, self.yidx] = self.kMatrixUT
        kMatrix[self.yidx, self.xidx] = -self.kMatrixUT

        # Populate the diagonal
        ind = np.diag_indices(kMatrix.shape[0])
        kMatrix[ind[0], ind[1]] = self.kMatrixDiag

        # Apply Koopman operation
        gnext = torch.bmm(kMatrix.expand(g.size(0), kMatrix.size(0), kMatrix.size(0)), g.unsqueeze(-1))
        self.kMatrix = kMatrix
        return gnext.squeeze(-1) # Squeeze empty dim from bmm

    @property
    def koopmanOperator(self, requires_grad: bool =True) -> Tensor:
        """Current Koopman operator

        Args:
            requires_grad (bool, optional): If to return with gradient storage. Defaults to True

        Returns:
            (Tensor): Full Koopman operator tensor
        """
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    def _normalize(self, x):
        return (x - self.mu.unsqueeze(0))/self.std.unsqueeze(0)

    def _unnormalize(self, x):
        return self.std.unsqueeze(0)*x + self.mu.unsqueeze(0)

    @property
    def koopmanDiag(self):
        return self.kMatrixDiag

    @property
    def num_parameters(self):
        """Get number of learnable parameters in model
        """
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count

    @property
    def devices(self):
        """Get list of unique device(s) model exists on
        """
        devices = []
        for param in self.parameters():
            if(not param.device in devices):
                devices.append(param.device)
        for buffer in self.buffers():
            if (not buffer.device in devices):
                devices.append(buffer.device)
        return devices

    @property
    def input_dims(self):
        return self.config.state_dims

    def save_model(self, save_directory: str, epoch: int = 0) -> None:
        """Saves embedding model to the specified directory.

        Args:
            save_directory (str): Folder directory to save state dictionary to.
            epoch (int, optional): Epoch of current model for file name. Defaults to 0.
        
        Raises:
            FileNotFoundError: If provided path is a file
        """
        if os.path.isfile(save_directory):
            raise FileNotFoundError("Provided path ({}) should be a directory, not a file".format(save_directory))

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "{}{:d}.pth".format(self.model_name, epoch))
        # Save pytorch model to file
        torch.save(self.state_dict(), output_model_file)

    def load_model(self, file_or_path_directory: str, epoch: int = 0) -> None:
        """Load a embedding model from the specified file or path
        
        Args:
            file_or_path_directory (str): File or folder path to load state dictionary from.
            epoch (int, optional): Epoch of current model for file name, used if folder path is provided. Defaults to 0.
        
        Raises:
            FileNotFoundError: If provided file or directory could not be found.
        """
        if os.path.isfile(file_or_path_directory):
            logger.info('Loading embedding model from file: {}'.format(file_or_path_directory))
            self.load_state_dict(torch.load(file_or_path_directory, map_location=lambda storage, loc: storage))
        elif  os.path.isdir(file_or_path_directory):
            file_path = os.path.join(file_or_path_directory, "{}{:d}.pth".format(self.model_name, epoch))
            logger.info('Loading embedding model from file: {}'.format(file_path))
            self.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
        else:
            raise FileNotFoundError("Provided path or file ({}) does not exist".format(file_or_path_directory))

class EmbeddingTrainingHead(nn.Module):
    """
    Training head for the embedding model, responsible for entire time-series processing; 
    contains method for training model for a single epoch
    """
    def __init__(self, config):
        super().__init__()
        self.embedding_model = EmbeddingModel(config)

    def forward(self, states: Tensor) -> FloatTuple:
        """Trains model for a single epoch

        Args:
            states (Tensor): [B, T, 3] Time-series feature tensor

        Returns:
            FloatTuple: Tuple containing:
            
                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        self.embedding_model.train()
        device = self.embedding_model.devices[0]

        loss_reconstruct = 0
        loss_dynamic = 0
        mseLoss = nn.MSELoss()

        # coefficients for the three loss terms
        recons_lmbd = self.embedding_model.config.recons_lmbd
        dynamic_lmbd = self.embedding_model.config.dynamic_lmbd
        decay_lmbd = self.embedding_model.config.decay_lmbd

        xin0 = states[:,0].to(device) # Time-step

        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0) # just to see whether G and F are inverse function to each other
        loss = recons_lmbd * mseLoss(xin0, xRec0) # loss keeps track of all 3 types of losses
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach() # loss_reconstruct only keeps track of 1 loss

        g1_old = g0 # F(xin0)
        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            # note that g1_old never uses the true value, but keeps building up error
            # It becomes K(...K(K(K(F(x_0))))...)
            xin0 = states[:,t0,:].to(device) # Next time-step
            _, xRec1 = self.embedding_model(xin0)

            g1Pred = self.embedding_model.koopmanOperation(g1_old) # K(g1_old)
            xgRec1 = self.embedding_model.recover(g1Pred) # predicted next time step x

            # below 3 losses are dynamics loss, reconstruction loss, decay loss
            loss = loss + dynamic_lmbd*mseLoss(xgRec1, xin0) + recons_lmbd*mseLoss(xRec1, xin0) \
                + decay_lmbd*torch.sum(torch.pow(self.embedding_model.koopmanOperator, 2))

            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            loss_dynamic = loss_dynamic + mseLoss(xgRec1, xin0)
            g1_old = g1Pred

        return loss, loss_reconstruct, loss_dynamic

    # Charlie 4/21/2022 below is the orignal evaluate method, it is in a different scale then the training, but still could be helpful
    # def evaluate(self, states: Tensor) -> Tuple[float, Tensor, Tensor]:
    #     """Evaluates the embedding models reconstruction error and returns its
    #     predictions. Evaluates 2 things: how good the reconstruction is, and how
    #     good the one-step prediction is.

    #     Args:
    #         states (Tensor): [B, T, 3] Time-series feature tensor

    #     Returns:
    #         Tuple[Float, Tensor, Tensor]: Test error, Predicted states, Target states
    #     """
    #     self.embedding_model.eval()
    #     device = self.embedding_model.devices[0]

    #     mseLoss = nn.MSELoss()

    #     total_recons_loss = 0

    #     # Pull out targets from prediction dataset
    #     yTarget = states[:,1:].to(device) # target is t_1 to t_N
    #     xInput = states[:,:-1].to(device) # input is t_0 to t_N-1
    #     yPred = torch.zeros(yTarget.size()).to(device)

    #     # Test accuracy of one time-step
    #     for i in range(xInput.size(1)):
    #         xInput0 = xInput[:,i].to(device)
    #         g0 = self.embedding_model.embed(xInput0)
    #         yPred0 = self.embedding_model.recover(g0)
    #         yPred[:,i] = yPred0.squeeze().detach()

    #         # added by Charlie 4/20
    #         recons_loss = mseLoss(xInput0, self.embedding_model(xInput0)[1]) # G(F(x)), should be x
    #         total_recons_loss = total_recons_loss + recons_loss

    #     dynamic_loss = mseLoss(yTarget, yPred)

    #     total_recons_loss = total_recons_loss / xInput.size(1) # average over time steps
    #     return dynamic_loss, yPred, yTarget, total_recons_loss

    def evaluate(self, states: Tensor) -> Tuple[float, Tensor, Tensor]:
        """Evaluates the embedding models reconstruction error and returns its
        predictions. Evaluates 2 things: how good the reconstruction is, and how
        good the one-step prediction is.

        Args:
            states (Tensor): [B, T, 3] Time-series feature tensor

        Returns:
            Tuple[Float, Tensor, Tensor]: Test error, Predicted states, Target states
        """
        self.embedding_model.eval()
        device = self.embedding_model.devices[0]

        mseLoss = nn.MSELoss()

        loss_reconstruct = 0
        loss_dynamic = 0
        mseLoss = nn.MSELoss()

        # coefficients for the three loss terms
        recons_lmbd = self.embedding_model.config.recons_lmbd
        dynamic_lmbd = self.embedding_model.config.dynamic_lmbd
        decay_lmbd = self.embedding_model.config.decay_lmbd

        xin0 = states[:,0].to(device) # Time-step

        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0) # just to see whether G and F are inverse function to each other
        loss = recons_lmbd * mseLoss(xin0, xRec0) # loss keeps track of all 3 types of losses
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach() # loss_reconstruct only keeps track of 1 loss

        g1_old = g0 # F(xin0)
        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            # note that g1_old never uses the true value, but keeps building up error
            # It becomes K(...K(K(K(F(x_0))))...)
            xin0 = states[:,t0,:].to(device) # Next time-step
            _, xRec1 = self.embedding_model(xin0)

            g1Pred = self.embedding_model.koopmanOperation(g1_old) # K(g1_old)
            xgRec1 = self.embedding_model.recover(g1Pred) # predicted next time step x

            # below 3 losses are dynamics loss, reconstruction loss, decay loss
            loss = loss + dynamic_lmbd*mseLoss(xgRec1, xin0) + recons_lmbd*mseLoss(xRec1, xin0) \
                + decay_lmbd*torch.sum(torch.pow(self.embedding_model.koopmanOperator, 2))

            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach() # did not average over timesteps, but it's fine
            loss_dynamic = loss_dynamic + mseLoss(xgRec1, xin0)
            g1_old = g1Pred

        return loss, loss_reconstruct, loss_dynamic

    def save_model(self, *args, **kwargs):
        """
        Saves the embedding model
        """
        assert not self.embedding_model is None, "Must initialize embedding model before saving."

        self.embedding_model.save_model(*args, **kwargs)


    def load_model(self, *args, **kwargs):
        """Load the embedding model
        """
        assert not self.embedding_model is None, "Must initialize embedding model before loading."

        self.embedding_model.load_model(*args, **kwargs)

    
