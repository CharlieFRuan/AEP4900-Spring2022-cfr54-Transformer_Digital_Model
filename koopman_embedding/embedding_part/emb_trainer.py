import torch 
import numpy as np
import logging
from emb_train_head import EmbeddingModel, EmbeddingTrainingHead
import argparse
from typing import Tuple, Dict
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

Optimizer = torch.optim.Optimizer
Scheduler = torch.optim.lr_scheduler._LRScheduler

def set_seed(seed: int) -> None:
    """Set random seed

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EmbeddingTrainer:
    """Trainer for Koopman embedding model

    Args:
        model (EmbeddingTrainingHead): Embedding training model
        args (TrainingArguments): Training arguments
        optimizers (Tuple[Optimizer, Scheduler]): Tuple of Pytorch optimizer and lr scheduler.
        viz (Viz, optional): Visualization class. Defaults to None.
    """
    def __init__(self,
        model: EmbeddingTrainingHead,
        args: argparse.ArgumentParser,
        optimizers: Tuple[Optimizer, Scheduler]
    ) -> None:
        """Constructor
        """
        self.model = model.to(args.device)
        self.args = args
        self.optimizers = optimizers

        set_seed(self.args.seed)

    def train(self, training_loader:DataLoader, eval_dataloader:DataLoader) -> None:
        """Training loop for the embedding model

        Args:
            training_loader (DataLoader): Training dataloader
            eval_dataloader (DataLoader): Evaluation dataloader
        """
        optimizer = self.optimizers[0]
        lr_scheduler = self.optimizers[1]

        train_losses = []
        test_losses = []

        
        # Loop over epochs
        for epoch in range(1, self.args.epochs + 1):
              
            loss_total = 0.0 # sum of all 3 losses: reconstruct, dynamic, and decay
            loss_reconstruct = 0.0
            loss_dynamic = 0.0
            self.model.zero_grad()
            for mbidx, inputs in enumerate(training_loader):

                loss0, loss_reconstruct0, loss_dynamic0 = self.model(**inputs)
                loss0 = loss0.sum()

                loss_reconstruct = loss_reconstruct + loss_reconstruct0.sum()
                loss_dynamic = loss_dynamic + loss_dynamic0.sum()
                loss_total = loss_total + loss0.detach()
                # Backwards!
                loss0.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()

                # if (mbidx+1) % 10 == 0:
                #     logger.info('Epoch {:d}: Completed mini-batch {}/{}.'.format(epoch, mbidx+1, len(training_loader)))

            # Progress learning rate scheduler
            lr_scheduler.step()
            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
                break
            loss_total = loss_total / len(training_loader) # average over each batch
            loss_dynamic = loss_dynamic / len(training_loader) # average over each batch
            loss_reconstruct = loss_reconstruct / len(training_loader) # average over each bathc
            logger.info("Epoch {:d}: Training loss {:.03f}, of wich Dynamic loss {:.03f}, Recons loss {:.03f}, Lr {:.05f}".format(epoch, loss_total, loss_dynamic, loss_reconstruct, cur_lr))

            # Evaluate current model
            if(epoch%5 == 0 or epoch == 1):
                output = self.evaluate(eval_dataloader, epoch=epoch)
                logger.info('Epoch {:d} Test Loss: {:.04f}, of which Dynamic loss: {:.04f}, Recons loss: {:.04f}'.format(epoch, output['total_loss'], output['dynamic_loss'], output['recons_loss']))
                test_losses.append(output['total_loss'])

            # Save model checkpoint
            if epoch % self.args.save_steps == 0:
                logger.info("Checkpointing model, optimizer and scheduler.")
                # Save model checkpoint
                self.model.save_model(self.args.ckpt_dir, epoch=epoch)
                torch.save(optimizer.state_dict(), os.path.join(self.args.ckpt_dir, "optimizer{:d}.pt".format(epoch)))
                torch.save(lr_scheduler.state_dict(), os.path.join(self.args.ckpt_dir, "scheduler{:d}.pt".format(epoch)))

            train_losses.append(loss_total)

        return train_losses, test_losses
        

    # below is the origianl evaluate, changed 4/21 by Charlie, since test loss is in different scale than training
    # @torch.no_grad()
    # def evaluate(self, eval_dataloader: DataLoader, epoch: int = 0) -> Dict[str, float]:
    #     """Run evaluation, plot prediction and return metrics.

    #     Args:
    #         eval_dataset (Dataset): Evaluation dataloader
    #         epoch (int, optional): Current epoch, used for naming figures. Defaults to 0.

    #     Returns:
    #         Dict[str, float]: Dictionary of prediction metrics
    #     """
    #     total_dynamic_loss = 0
    #     total_recons_loss = 0
    #     for mbidx, inputs in enumerate(eval_dataloader):
    #         # inputs['states'] is in dimension (batch, time_series_len, states_num)
    #         dynamic_loss, state_pred, state_target, recons_loss = self.model.evaluate(**inputs)
    #         total_dynamic_loss = total_dynamic_loss + dynamic_loss
    #         total_recons_loss = total_recons_loss + recons_loss
    #     return {'dynamic_loss': total_dynamic_loss/len(eval_dataloader),
    #             'recons_loss': total_recons_loss/len(eval_dataloader)} # averaged over each batch

    @torch.no_grad()
    def evaluate(self, eval_dataloader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """Run evaluation, plot prediction and return metrics.

        Args:
            eval_dataset (Dataset): Evaluation dataloader
            epoch (int, optional): Current epoch, used for naming figures. Defaults to 0.

        Returns:
            Dict[str, float]: Dictionary of prediction metrics
        """
        total_dynamic_loss = 0
        total_recons_loss = 0
        total_loss = 0

        for mbidx, inputs in enumerate(eval_dataloader):
            # inputs['states'] is in dimension (batch, time_series_len, states_num)
            loss, loss_reconstruct, loss_dynamic = self.model.evaluate(**inputs)
            total_dynamic_loss = total_dynamic_loss + loss_dynamic
            total_recons_loss = total_recons_loss + loss_reconstruct
            total_loss = total_loss + loss

        return {'total_loss': total_loss/len(eval_dataloader),
                'dynamic_loss': total_dynamic_loss/len(eval_dataloader),
                'recons_loss': total_recons_loss/len(eval_dataloader)} # averaged over each batch