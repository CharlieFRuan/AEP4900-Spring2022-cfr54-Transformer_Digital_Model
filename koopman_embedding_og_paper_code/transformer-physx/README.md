# Transformer PhysX
 [![PyPI version](https://badge.fury.io/py/trphysx.svg)](https://pypi.org/project/trphysx/) [![CircleCI](https://circleci.com/gh/zabaras/transformer-physx.svg?style=shield&circle-token=1d8e1117961b1d63d754f90e6b526649607aad34&branch=main)](https://circleci.com/gh/zabaras/transformer-physx) [![Documentation Status](https://readthedocs.org/projects/transformer-physx/badge/?version=latest)](https://transformer-physx.readthedocs.io/en/latest/?badge=latest) [![Website](https://img.shields.io/website?url=https%3A%2F%2Fzabaras.github.io%2Ftransformer-physx%2F)](https://zabaras.github.io/transformer-physx/) ![liscense](https://img.shields.io/github/license/zabaras/transformer-physx)


Transformer PhysX is a Python packaged modeled after the [Hugging Face repository](https://github.com/huggingface/transformers) designed for the use of transformers for modeling physical systems. Transformers have seen recent success in both natural language processing and vision fields but have yet to fully permute other machine learning areas. Originally proposed in [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957), this projects goal is to make these deep learning advances including self-attention and Koopman embeddings more accessible for the scientific machine learning community.

[Website](https://zabaras.github.io/transformer-physx/) | [Documentation](https://transformer-physx.readthedocs.io) |[Getting Started](https://transformer-physx.readthedocs.io/en/latest/install.html) | [Data](https://www.doi.org/10.5281/zenodo.5148523)

### Associated Papers

Transformers for Modeling Physical Systems [ [ArXiV](https://arxiv.org/abs/2010.03957) ]

### Colab Quick Start

| | Embedding Model | Transformer |
|---|:-------:|:-------:|
| Lorenz | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zabaras/transformer-physx/blob/main/examples/lorenz/train_lorenz_enn.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zabaras/transformer-physx/blob/main/examples/lorenz/train_lorenz_transformer.ipynb) |
| Cylinder Flow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zabaras/transformer-physx/blob/main/examples/cylinder/train_cylinder_enn.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zabaras/transformer-physx/blob/main/examples/cylinder/train_cylinder_transformer.ipynb) |
| Gray-Scott | - | - |
| Rossler | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zabaras/transformer-physx/blob/main/examples/rossler/train_rossler_enn.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zabaras/transformer-physx/blob/main/examples/rossler/train_rossler_transformer.ipynb) |

### Road Map

This is an on going project, hence many parts are not fully developed and will be added in the near future. If you have any particular questions or features you are interested in, please make a issue request so it can be prioritized!
Thanks for understanding.

- Tutorials/ blog post to help with introduction
- Info on each example system in docs
- Additional Unit Testing for Better Code Coverage
- Parallel Data training for transformer
- Unsupervised pretraining physics

### Additional Resources

- [Huggingface Repository](https://github.com/huggingface/transformers)
- [Transformer illustrated blog post](https://jalammar.github.io/illustrated-transformer/)
- [Deep learning Koopman dynamics blog post](https://nicholasgeneva.com/deep-learning/koopman/dynamics/2020/05/30/intro-to-koopman.html)


### Contact
Open an issue on the Github repository if you have any questions/concerns.

### Citation
Find this useful or like this work? Cite us with:

```latex
@article{geneva2020transformers,
    title={Transformers for Modeling Physical Systems},
    author={Geneva, Nicholas and Zabaras, Nicholas},
    journal={arXiv preprint arXiv:2010.03957},
    year={2020}
}
```