# Multimodal Semi-Supervised Learning for 3D Objects

Our paper has been accepted by BMVC 2021

Arvix version [here](https://arxiv.org/abs/2110.11601)

![image](https://github.com/AutoAILab/Multimodal-Semi-Learning/blob/main/Framework.png)

## Abstract

We propose a novel multimodal semi-supervised learning framework by introducing instance-level consistency constraint and  a novel multimodal contrastive prototype (M2CP) loss. The instance-level consistency enforces the network to generate consistent representations for multimodal data of the same object regardless of its modality. The M2CP maintains a multimodal prototype for each class and learns features with small intra-class variations by minimizing the feature distance of each object to its prototype while maximizing the distance to the others. Our proposed framework significantly outperforms all the state-of-the-art counterparts for both classification and retrieval tasks by a large margin on the modelNet10 and ModelNet40 datasets.

## Download Dataset
[Image](https://drive.google.com/file/d/1-Gz4HMwM96zEnR3Hji4UEAGB4FPrd5Xo/view?usp=sharing)

[Mesh](https://drive.google.com/file/d/1pcOK10d4NrgJzAbqLJFF9hIA43pwiVEH/view?usp=sharing)

[Point Clouds](https://drive.google.com/file/d/1VyXh4Q7EJK0uUM6xAgAHyzwpFFui_se4/view?usp=sharing)


## Installation
Install ```Python``` -- This repo is tested with Python 3.7.6.

Install ```NumPy``` -- This repo is tested with NumPy 1.18.5. Please make sure your NumPy version is at least 1.18.

Install ```PyTorch``` with CUDA -- This repo is tested with PyTorch 1.5.1, CUDA 10.2. It may work with newer versions, but that is not guaranteed. A lower version may be problematic.

Install ```TensorFlow``` (for TensorBoard) -- This repo is tested with TensorFlow 2.2.0.


## Training
This netowrk is trained with two 16G Tesla V100 GPU

Remember change the dataloader root before training
```
python train.py
```

## Testing
The trained model is save in checkpoints file. Remember change the test model name in ```test_12views.py``` before testing.

```
python test_12views.py

```
## Pretrained model
We provide the pre-trained models of [ModelNet40-10%](https://www.dropbox.com/s/wptl91al4jhnwqo/ModelNet40_p10_nt_xw2_aw0_cw9_baseshare_Cos_12views_imgsize224_10w.zip?dl=0)

## Citation
```
@article{chen2021multimodal,
  title={Multimodal Semi-Supervised Learning for 3D Objects},
  author={Chen, Zhimin and Jing, Longlong and Liang, Yang and Tian, YingLi and Li, Bing},
  journal={arXiv preprint arXiv:2110.11601},
  year={2021}
}
```

