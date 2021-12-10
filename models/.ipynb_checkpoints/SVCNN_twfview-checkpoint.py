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
        
        print(pre_trained, '**********pre_trained************')

        # if pre_trained:
        #     self.img_net = torch.load(pre_trained)
        # else:
        #     # print('--------- The ImageNet is loading from ImageNet models --------- ')
        
        resnet50 = models.resnet50(pretrained = pre_trained)
        resnet50 = list(resnet50.children())[:-1]
        
        resnet18 = models.resnet18(pretrained = pre_trained)
        resnet18 = list(resnet18.children())[:-1]
        self.img_net = nn.Sequential(*resnet18)

        #head
#         self.linear1 = nn.Linear(2048, 512, bias=False)

        self.linear1 = nn.Linear(512, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)

        # self.bn6 = nn.BatchNorm1d(512)
        # self.linear2 = nn.Linear(512, 256)

#         self.pred = nn.Sequential(
#             nn.BatchNorm1d(512),
#             nn.ReLU(),   
# #             nn.Dropout(p=0.5),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
# #             nn.Dropout(p=0.5),
#             nn.Linear(256, 40)
#         )

#         self.feature = nn.Sequential(
#             nn.BatchNorm1d(512),
#             nn.ReLU(),  
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 256)
#         )
#         self.groupfeature = nn.Sequential(
#             nn.BatchNorm1d(512),
#             nn.ReLU(),  
#             nn.Linear(512, 256),
# #             nn.BatchNorm1d(256),
# #             nn.ReLU(),
# #             nn.Linear(256, 256)
#         )


    def forward(self, img, imgV, img1V3, img1V4, img1V5, img1V6, img1V7, img1V8, img1V9, img1V10, img1V11,img1V12):

        img_feat = self.img_net(img)
        img_feat = img_feat.squeeze(3)
        img_feat = img_feat.squeeze(2)

        img_featV = self.img_net(imgV)
        img_featV = img_featV.squeeze(3)
        img_featV = img_featV.squeeze(2)
        
        img_feat1V3 = self.img_net(img1V3)
        img_feat1V3 = img_feat1V3.squeeze(3)
        img_feat1V3 = img_feat1V3.squeeze(2)

        img_feat1V4 = self.img_net(img1V4)
        img_feat1V4 = img_feat1V4.squeeze(3)
        img_feat1V4 = img_feat1V4.squeeze(2)
        
        img_feat1V5 = self.img_net(img1V5)
        img_feat1V5 = img_feat1V5.squeeze(3)
        img_feat1V5 = img_feat1V5.squeeze(2)

        img_feat1V6 = self.img_net(img1V6)
        img_feat1V6 = img_feat1V6.squeeze(3)
        img_feat1V6 = img_feat1V6.squeeze(2)
        
        img_feat1V7 = self.img_net(img1V7)
        img_feat1V7 = img_feat1V7.squeeze(3)
        img_feat1V7 = img_feat1V7.squeeze(2)
        
        img_feat1V8 = self.img_net(img1V8)
        img_feat1V8 = img_feat1V8.squeeze(3)
        img_feat1V8 = img_feat1V8.squeeze(2)
        
        img_feat1V9 = self.img_net(img1V9)
        img_feat1V9 = img_feat1V9.squeeze(3)
        img_feat1V9 = img_feat1V9.squeeze(2)
        
        img_feat1V10 = self.img_net(img1V10)
        img_feat1V10 = img_feat1V10.squeeze(3)
        img_feat1V10 = img_feat1V10.squeeze(2)
        
        img_feat1V11 = self.img_net(img1V11)
        img_feat1V11 = img_feat1V11.squeeze(3)
        img_feat1V11 = img_feat1V11.squeeze(2)
        
        img_feat1V12 = self.img_net(img1V12)
        img_feat1V12 = img_feat1V12.squeeze(3)
        img_feat1V12 = img_feat1V12.squeeze(2)
        
        img_base1 = self.linear1(img_feat)
        img_base2 = self.linear1(img_featV)
        img_base3 = self.linear1(img_feat1V3)
        img_base4 = self.linear1(img_feat1V4)
        img_base5 = self.linear1(img_feat1V5)
        img_base6 = self.linear1(img_feat1V6)
        img_base7 = self.linear1(img_feat1V7)
        img_base8 = self.linear1(img_feat1V8)
        img_base9 = self.linear1(img_feat1V9)
        img_base10 = self.linear1(img_feat1V10)
        img_base11 = self.linear1(img_feat1V11)
        img_base12 = self.linear1(img_feat1V12)
        
        img_base1 =  F.relu(self.bn6(img_base1))
        img_base2 =  F.relu(self.bn6(img_base2))
        img_base3 =  F.relu(self.bn6(img_base3))
        img_base4 =  F.relu(self.bn6(img_base4))
        img_base5 =  F.relu(self.bn6(img_base5))
        img_base6 =  F.relu(self.bn6(img_base6))
        img_base7 =  F.relu(self.bn6(img_base7))
        img_base8 =  F.relu(self.bn6(img_base8))
        img_base9 =  F.relu(self.bn6(img_base9))
        img_base10 =  F.relu(self.bn6(img_base10))
        img_base11 =  F.relu(self.bn6(img_base11))
        img_base12 =  F.relu(self.bn6(img_base12))
        
        img_base = torch.max(img_base1, img_base2)
        img_base = torch.max(img_base, img_base3)
        img_base = torch.max(img_base, img_base4)
        img_base = torch.max(img_base, img_base5)
        img_base = torch.max(img_base, img_base6)
        img_base = torch.max(img_base, img_base7)
        img_base = torch.max(img_base, img_base8)
        img_base = torch.max(img_base, img_base9)
        img_base = torch.max(img_base, img_base10)
        img_base = torch.max(img_base, img_base11)
        img_base = torch.max(img_base, img_base12)


#         img_pred = self.pred(img_base)
#         img_feat = self.feature(img_base)
#         img_gfeat = self.groupfeature(img_base)

        return img_base


# The networks for training contrastive
class FusionHead(nn.Module):
    def __init__(self):
        super(FusionHead, self).__init__()
       
        self.pred = nn.Sequential(
#             nn.BatchNorm1d(512),
#             nn.ReLU(),   
#             nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
#             nn.Dropout(p=0.5),
            nn.Linear(256, 40)
        )

        self.feature = nn.Sequential(
#             nn.BatchNorm1d(512),
#             nn.ReLU(),  
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.groupfeature = nn.Sequential(
#             nn.BatchNorm1d(512),
#             nn.ReLU(),  
            nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 256)
        )

    # pt1, img1, centers1, corners1, normals1, neighbor_index1
    def forward(self, point_base, mesh_base, img_base):

        # print(pt_base.size(), mesh_base.size())
        img_pred = self.pred(img_base)
        img_feat = self.feature(img_base)
        img_gfeat = self.groupfeature(img_base)
        
        pt_pred = self.pred(point_base)
        pt_feat = self.feature(point_base)
        pt_gfeat = self.groupfeature(point_base)
        
        mesh_pred = self.pred(mesh_base)
        mesh_feat = self.feature(mesh_base)
        mesh_gfeat = self.groupfeature(mesh_base)

        # print(concatenate_feature.size())

        return img_pred, pt_pred, mesh_pred, img_feat, pt_feat, mesh_feat, img_gfeat, pt_gfeat, mesh_gfeat
    
    
    
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()

        self.pred = nn.Sequential(   
            nn.Linear(512 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 40)
        )

    # pt1, img1, centers1, corners1, normals1, neighbor_index1
    def forward(self, pt_base, mesh_base, img_base):

        # print(pt_base.size(), mesh_base.size())
        concatenate_feature = torch.cat([pt_base, mesh_base, img_base], dim = 1)

        # print(concatenate_feature.size())
        fused_pred = self.pred(concatenate_feature)

        return fused_pred


# The networks for training contrastive
class Semi3D(nn.Module):
    def __init__(self, img_net, cloud_net, meshnet, fusion_net, fusion_head):
        super(Semi3D, self).__init__()

        self.img_net = img_net
        self.cloud_net = cloud_net
        self.meshnet = meshnet
        self.fusion_net = fusion_net
        self.fusion_head = fusion_head

    # pt1, img1, centers1, corners1, normals1, neighbor_index1
    def forward(self, pt, img, imgV, img1V3, img1V4, img1V5, img1V6, img1V7, img1V8, img1V9, img1V10, img1V11,img1V12, centers, corners, normals, neighbor_index):

        # get the representations for point cloud data
        pt_base = self.cloud_net(pt)  # [N,C]
        # print(cloud_ris.size(), cloud_zis.size())

        mesh_base = self.meshnet(centers, corners, normals, neighbor_index)

        # get the representations and the projections
        img_base = self.img_net(img, imgV, img1V3, img1V4, img1V5, img1V6, img1V7, img1V8, img1V9, img1V10, img1V11,img1V12)  # [N,C]

        fused_pred = self.fusion_net(pt_base, mesh_base, img_base)
        
        img_pred, pt_pred, mesh_pred, img_feat, pt_feat, mesh_feat, img_gfeat, pt_gfeat, mesh_gfeat = self.fusion_head(pt_base, mesh_base, img_base) 
        

        return pt_pred, mesh_pred, img_pred, fused_pred, pt_feat, mesh_feat, img_feat, pt_base, mesh_base, img_base, pt_gfeat, mesh_gfeat, img_gfeat


