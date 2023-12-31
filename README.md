# NLA_project

## Introduction 

Knowledge graphs (KGs) are large graph-structured databases that store facts in triple form, representing interlinked descriptions of entities, relationships, and semantic descriptions. These KGs provide structured knowledge that plays a vital role in knowledge-aware tasks such as recommendation systems, intelligent question-answering, recommender systems, and social network analysis. The effectiveness of these tasks heavily relies on the quality and completeness of the KGs. However, KGs often suffer from incompleteness, resulting in significant knowledge gaps. To address this, it is crucial to enhance existing KGs by supplementing missing knowledge to improve their overall quality and usefulness.

## Our project - Leveraging TuckER for Knowledge Graph Link Prediction

For this project, we utilize TuckER, a robust linear model based on Tucker decomposition of the binary tensor representation of knowledge graph triples. The implementation of TuckER can be found on GitHub at https://github.com/johanDDC/R-TuckER/tree/master. 

To accomplish the knowledge graph link prediction task, we employ the following optimization tools from the GitHub repository https://github.com/johanDDC/tucker_riemopt/tree/master:

- Riemann gradient
- Vector transport
- Retraction map

Additionally, we explore a range of optimization algorithms to enhance the performance of the knowledge graph link prediction task, namely 

- RGD
- RSGD with Momentum
- Adam

## Contribution to tackling the KG completion problem

Our contribution consisted in implementing and extending several optimization methods using Riemann gradient, namely

- RMSprop
- Nesterov momentum SGD
- AdamW
- AdaDelta

Moreover, we incorporated a smooth L1 loss, building upon the principles of Robust Low-Rank Matrix Completion through Riemannian Optimization. 


## Requirements 
Check `requirements.txt`

## Tutorial (short example)
1. Add wandb login and project to `rtucker/configs/base_config.py`
2. `cd rtrucker`
3. `python train.py --mode symmetric --optim rgd` (or other optimizers, which can be found in `train.py`)

More examples can be found in `notebooks/experiment.ipynb`

## References
I. Balažević, C. Allen, T. M. Hospedales. TuckER: Tensor Factorization for Knowledge Graph Completion. 2019. https://arxiv.org/pdf/1901.09590.pdf

B. Vandereycken. Low-rank matrix completion by Riemannian optimization—extended version. 2012. https://arxiv.org/pdf/1209.3834.pdf

P.-A. Absil, R. Mahony, R. Sepulchre. Vector Transport On Manifolds. 2007. https://perso.uclouvain.be/pa.absil/Talks/ICIAM070717_oom_05.pdf

T. Dettmers. Convolutional 2D Knowledge Graph Embeddings. 2018. https://arxiv.org/pdf/1707.01476.pdf

B. Mishra, H. Kasai. Low-rank tensor completion: a Riemannian manifold preconditioning approach. http://proceedings.mlr.press/v48/kasai16.pdf
