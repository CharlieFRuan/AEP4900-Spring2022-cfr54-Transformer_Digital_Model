# AEP4900-Spring2022-cfr54-Transformer_Digital_Model

This repository contains all the necessary code for an undergraduate research project in the McMahon Lab. The project aims to use Transformer-based neural network to improve the digital model in the Physics-Aware Training algorithm proposed in https://arxiv.org/abs/2104.13386. 

The koopman_embedding/ folder consists of the necessary code for training the Koopman NN (koopman_embedding/embedding_part/emb_train_script) and Transformer NN (koopman_embedding/transformer_part/trans_train_script.py).

Under the same folder, koopman_embedding/data_generation/generate_script.py is able to generate the 10-pendula dataset for training and evaluating.

koopman_embedding_og_paper_code/transformer-physx/ consists of the original code from the Koopman-based approach paper https://arxiv.org/abs/2010.03957. 

The fifty_nine_transformer/fifty_nine_transformer.ipynb consists of the corresponding code for 59_transformers' implementation.

The file scheduled_sampling_playground.ipynb consists of the corresponding code for scheduled_sampling's implementation. 
