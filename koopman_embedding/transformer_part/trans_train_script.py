import sys 
import logging
import torch

from torch import embedding
from hf_arg_parser import HfArgumentParser
from args import ModelArguments, TrainingArguments, DataArguments, ArgUtils
from phys_transformer_gpt2 import PhysformerGPT2
from phys_transformer_helpers import PhysformerTrain
from embed_data_for_trans import PhysicalDataset
from trainer import Trainer
from config import Config

from viz_lorenz import LorenzViz
from viz_10_pendula import TenPendulaViz

sys.path.append('../')
from embedding_part.emb_train_head import EmbeddingModel

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # 10 pendula
    sys.argv = sys.argv + ["--init_name", "10_pendula"]
    sys.argv = sys.argv + ["--embedding_file_or_path", "../embedding_part/outputs/embedding_10_pendula/ntrain4096_epochs750_batch64_dynamic_focus/checkpoints/embedding_model700.pth"]
    sys.argv = sys.argv + ["--training_h5_file","../data/train_10_pendula.hdf5"]
    sys.argv = sys.argv + ["--eval_h5_file","../data/test_10_pendula.hdf5"]
    sys.argv = sys.argv + ["--train_batch_size", "64"]
    sys.argv = sys.argv + ["--stride", "64"] # will play no role since one block size (60) makes the entire training data 
    sys.argv = sys.argv + ["--n_train", "2048"]
    sys.argv = sys.argv + ["--save_steps", "100"]
    sys.argv = sys.argv + ["--n_eval", "16"]
    sys.argv = sys.argv + ["--epochs", "200"]

    # lorenz
    # sys.argv = sys.argv + ["--init_name", "lorenz"]
    # sys.argv = sys.argv + ["--embedding_file_or_path", "../embedding_part/outputs/embedding_lorenz/ntrain2048_epochs200_batch512/checkpoints/embedding_model200.pth"]
    # sys.argv = sys.argv + ["--training_h5_file","../data/lorenz_training_rk.hdf5"]
    # sys.argv = sys.argv + ["--eval_h5_file","../data/lorenz_valid_rk.hdf5"]
    # sys.argv = sys.argv + ["--train_batch_size", "16"]
    # sys.argv = sys.argv + ["--stride", "64"]
    # sys.argv = sys.argv + ["--n_train", "2048"]
    # sys.argv = sys.argv + ["--save_steps", "25"]
    # sys.argv = sys.argv + ["--n_eval", "16"]

    # Parse arguments using the hugging face argument parser
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # Parse the sys.argv commandline args into the specified arg instances
    model_args, data_args, training_args = parser.parse_args_into_dataclasses() 

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    
    # Configure arguments after intialization 
    model_args, data_args, training_args = ArgUtils.config(model_args, data_args, training_args)

    # Load model configuration, same as the embedding part
    config = Config()

    # Load embedding model
    embedding_model = EmbeddingModel(config)
    embedding_model.load_model(model_args.embedding_file_or_path)
    embedding_model.to(training_args.src_device)

    # viz = LorenzViz(plot_dir=training_args.plot_dir) # lorenz
    viz = TenPendulaViz(plot_dir=training_args.plot_dir) # 10 pendula
    
    
    # Init transformer model
    transformer = PhysformerGPT2(config, model_args.model_name)
    model = PhysformerTrain(config, transformer)

    # Initialize training and validation datasets using the trained Embedding model
    training_data = PhysicalDataset(
        embedding_model, 
        data_args.training_h5_file, 
        block_size=config.n_ctx, 
        stride=data_args.stride,
        ndata=data_args.n_train, 
        overwrite_cache=data_args.overwrite_cache)
    print(training_data)

    eval_data = PhysicalDataset(
        embedding_model, 
        data_args.eval_h5_file, 
        # block_size=256, # for lorenz
        block_size=60, # for 10 pendula
        stride=1024, # does not play a role for now since one data entry takes only one block size (size 60)
        ndata=data_args.n_eval, 
        eval = True,
        overwrite_cache=data_args.overwrite_cache)
    
    # Train
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.lr, weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 14, 2, eta_min=1e-9)
    trainer = Trainer(
        model, 
        training_args, 
        (optimizer, scheduler), 
        train_dataset = training_data, 
        eval_dataset = eval_data, 
        embedding_model = embedding_model,
        viz=viz)

    trainer.train()
