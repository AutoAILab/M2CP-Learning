# from models.MVCNN import MVCNN
from __future__ import division, absolute_import
from models.dgcnn import DGCNN
from models.resnet import resnet18
from models.meshnet import MeshNet
from tools.dual_dataloader import SingleViewDataloader, MultiViewDataloader
from tools.utils import calculate_accuracy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse
import torch.optim as optim
import time

# The network for extracting image features
class SingleViewNet(nn.Module):

    def __init__(self, pre_trained = None):
        super(SingleViewNet, self).__init__()

        # if pre_trained:
        #     self.img_net = torch.load(pre_trained)
        # else:
        #     # print('--------- The ImageNet is loading from ImageNet models --------- ')
        resnet18 = models.resnet18(pretrained = pre_trained)
        resnet18 = list(resnet18.children())[:-1]
        self.img_net = nn.Sequential(*resnet18)

        #head
        self.linear1 = nn.Linear(512, 512, bias=False)

        # self.bn6 = nn.BatchNorm1d(512)
        # self.linear2 = nn.Linear(512, 256)

        self.pred = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(),      
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 40)
        )

        self.feature = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(),      
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )


    def forward(self, img, imgV):

        img_feat = self.img_net(img)
        img_feat = img_feat.squeeze(3)
        img_feat = img_feat.squeeze(2)

        img_featV = self.img_net(imgV)
        img_featV = img_featV.squeeze(3)
        img_featV = img_featV.squeeze(2)

        img_base1 = self.linear1(img_feat)
        img_base2 = self.linear1(img_featV)

        img_base = torch.max(img_base1, img_base2)

        img_pred = self.pred(img_base)
        img_feat = self.feature(img_base)

        return img_pred, img_feat, img_base

# The networks for training contrastive
class Semi3D(nn.Module):
    def __init__(self, img_net, cloud_net, meshnet):
        super(Semi3D, self).__init__()

        self.img_net = img_net
        self.cloud_net = cloud_net
        self.meshnet = meshnet

        self.pred = nn.Sequential(   
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 40)
        )

    
    # pt1, img1, centers1, corners1, normals1, neighbor_index1
    def forward(self, pt, img, imgV, centers, corners, normals, neighbor_index):

        # get the representations for point cloud data
        pt_pred, pt_feat, pt_base = self.cloud_net(pt)  # [N,C]
        # print(cloud_ris.size(), cloud_zis.size())

        mesh_pred, mesh_feat, mesh_base = self.meshnet(centers, corners, normals, neighbor_index)

        # get the representations and the projections
        img_pred, img_feat, img_base = self.img_net(img, imgV)  # [N,C]

        # print(pt_base.size(), mesh_base.size())
        concatenate_feature = torch.cat([pt_base, mesh_base], dim = 1)

        # print(concatenate_feature.size())
        fused_pred = self.pred(concatenate_feature)


        return pt_pred, mesh_pred, img_pred, fused_pred, pt_feat, mesh_feat, img_feat