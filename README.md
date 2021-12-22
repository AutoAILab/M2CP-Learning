# Multimodal Semi-Supervised Learning for 3D Objects

Our paper has been accepted by BMVC 2021

Arvix version [here](https://arxiv.org/abs/2110.11601)

![image](https://github.com/AutoAILab/Multimodal-Semi-Learning/blob/main/Framework.png)

## Abstract

We propose a novel multimodal semi-supervised learning framework by introducing instance-level consistency constraint and  a novel multimodal contrastive prototype (M2CP) loss. The instance-level consistency enforces the network to generate consistent representations for multimodal data of the same object regardless of its modality. The M2CP maintains a multimodal prototype for each class and learns features with small intra-class variations by minimizing the feature distance of each object to its prototype while maximizing the distance to the others. Our proposed framework significantly outperforms all the state-of-the-art counterparts for both classification and retrieval tasks by a large margin on the modelNet10 and ModelNet40 datasets.

