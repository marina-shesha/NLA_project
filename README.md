# NLA_project

Knowledge graphs (KGs) are large graph-structured databases that store facts in triple form, representing interlinked descriptions of entities, relationships, and semantic descriptions. These KGs provide structured knowledge that plays a vital role in knowledge-aware tasks such as recommendation systems, intelligent question-answering, recommender systems, and social network analysis. The effectiveness of these tasks heavily relies on the quality and completeness of the KGs. However, KGs often suffer from incompleteness, resulting in significant knowledge gaps. To address this, it is crucial to enhance existing KGs by supplementing missing knowledge to improve their overall quality and usefulness.

For this project, we utilize TuckER, a robust linear model based on Tucker decomposition of the binary tensor representation of knowledge graph triples. The implementation of TuckER can be found on GitHub at https://github.com/johanDDC/R-TuckER/tree/master. 

To accomplish the knowledge graph link prediction task, we employ the following optimization tools from the GitHub repository https://github.com/johanDDC/tucker_riemopt/tree/master:

- Riemann gradient
- Vector transport
- Retraction map

Additionally, we explore different optimization algorithms, including 

- RGD
- RSGD with Momentum
- Adam

to enhance the performance of the knowledge graph link prediction task.
