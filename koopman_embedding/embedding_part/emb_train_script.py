import sys
import logging
import torch
from torch.optim.lr_scheduler import ExponentialLR
from emb_args import EmbeddingArgs
from emb_train_head import EmbeddingModel, EmbeddingTrainingHead
from data_handler import PendulumDataHandler
from emb_trainer import EmbeddingTrainer
import matplotlib.pyplot as plt
import numpy as np
# sys.path.append('../')
from config import Config

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # for 10 pendula
    # sys.argv = sys.argv + ["--exp_name", "10_pendula"]
    # sys.argv = sys.argv + ["--training_h5_file", "train_10_pendula.hdf5"]
    # sys.argv = sys.argv + ["--eval_h5_file", "test_10_pendula.hdf5"]
    # sys.argv = sys.argv + ["--batch_size", "64"]
    # sys.argv = sys.argv + ["--n_train", "4096"]
    # sys.argv = sys.argv + ["--save_steps", "100"]
    # sys.argv = sys.argv + ["--epochs", "750"]
    # sys.argv = sys.argv + ["--lr", 0.002] # added 4/21 by Charlie, thinking that learnign rate is too high towards the end

    # for lorenz
    sys.argv = sys.argv + ["--exp_name", "lorenz"]
    sys.argv = sys.argv + ["--training_h5_file", "/Users/charlieruan/projects_data/McMahon/data/lorenz_training_rk.hdf5"]
    sys.argv = sys.argv + ["--eval_h5_file", "/Users/charlieruan/projects_data/McMahon/data/lorenz_valid_rk.hdf5"]
    sys.argv = sys.argv + ["--batch_size", "512"]
    sys.argv = sys.argv + ["--block_size", "16"]
    sys.argv = sys.argv + ["--n_train", "2048"]
    sys.argv = sys.argv + ["--save_steps", "100"]
    sys.argv = sys.argv + ["--epochs", "200"]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    args = EmbeddingArgs().parse()  
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Torch device: {}".format(args.device))

    config = Config()
    data_handler = PendulumDataHandler()

    # Set up data-loaders
    training_loader = data_handler.createTrainingLoader(args.training_h5_file, ndata=args.n_train, batch_size=args.batch_size)
    testing_loader = data_handler.createTestingLoader(args.eval_h5_file, ndata=args.n_eval, batch_size=8)

    # Set up model
    model = EmbeddingTrainingHead(config)
    mu, std = data_handler.norm_params
    model.embedding_model.mu = mu.to(args.device)
    model.embedding_model.std = std.to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*0.995**(-1), weight_decay=1e-8)
    scheduler = ExponentialLR(optimizer, gamma=0.995)

    trainer = EmbeddingTrainer(model, args, (optimizer, scheduler))
    train_losses, test_losses = trainer.train(training_loader, testing_loader)

    f, ax = plt.subplots(1, figsize=(8.5, 8.5))
    plt.plot(np.arange(1,args.epochs+1), train_losses)
    plt.plot(np.arange(0,args.epochs+1, 5), test_losses)
    plt.ylim(top=1000)

    plt.savefig("./structure_tuning_training_plots/original_lorenz_struct.png", dpi=300)
    plt.clf()