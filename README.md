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


# Requirements 
## Dataset Requirements

1. FB15k: This dataset is a subset of Freebase, a comprehensive real-world fact database. It comprises 592,213 triplets, involving 14,951 entities and 1,345 relationship types.

2. FB15k-237: Derived from FB15k, this dataset is specifically designed to challenge simple models by removing the inverse of many relations present in the training set from the validation and test sets. It consists of 310,116 triplets, featuring 14,541 entities and 237 relation types.
