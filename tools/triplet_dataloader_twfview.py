#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""

import os
import sys
import glob
import h5py
import json
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch
import numpy as np

# from tools.visualize import showpoints


def load_data(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = '/scratch1/zhiminc/MVdata/dataset/'
    all_data = []
    all_label = []
    img_lst = []
    # print(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition))
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
#         print('*************************************************************')
#         print(glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)))
        split = h5_name[-4]
        jason_name = '/scratch1/zhiminc/MVdata/dataset/modelnet40_ply_hdf5_2048/ply_data_' + partition +'_' + split + '_id2file.json'

        # print('index name :',idx_name)
        with open(jason_name) as json_file:
            images = json.load(json_file)
        
        img_lst = img_lst + images

        f = h5py.File(h5_name)

        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        # for idx in label:
        #     print(idx)
        # print(label)
        # print(data.shape, label.shape)
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
    # print(len(all_data), len(all_label), len(img_lst))
    return all_data, all_label, img_lst


def load_modelnet10_data(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = '/scratch1/zhiminc/MVdata/dataset/'
    # print(DATA_DIR)
    all_data = []
    all_label = []
    img_lst = []
    # print(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition))
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet10_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        split = h5_name[-4]
        jason_name = '/scratch1/zhiminc/MVdata/dataset/modelnet10_ply_hdf5_2048/ply_data_' + partition +'_' + split + '_id2file.json'

        # print('index name :',idx_name)
        with open(jason_name) as json_file:
            images = json.load(json_file)
        # print(images)

        img_lst = img_lst + images
        # for img in img_lst:
        #     print(img)

        f = h5py.File(h5_name)

        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        # print(data.shape, label.shape)
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
    print(len(all_data), len(all_label), len(img_lst))
    return all_data, all_label, img_lst


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def random_scale(pointcloud, scale_low=0.8, scale_high=1.25):
    N, C = pointcloud.shape
    scale = np.random.uniform(scale_low, scale_high)
    pointcloud = pointcloud*scale
    return pointcloud

class TripletDataloader(Dataset):
    def __init__(self, dataset, num_points, partition='labeled', perceptange = 10):
        self.dataset = dataset

        if partition in  ['labeled', 'unlabeled']:
            self.data_split = 'train'
        else:
            self.data_split = 'test'

        if self.dataset == 'ModelNet40':
            data, label, img_lst = load_data(self.data_split)
        else:
            data, label, img_lst = load_modelnet10_data(self.data_split)

        # random.seed(1234)
        # zipped_lst = list(zip(data, label, img_lst))
        # random.shuffle(zipped_lst)
        # data, label, img_lst = zip(*zipped_lst)
        # data, label, img_lst = np.array(data), list(label), list(img_lst)
        # print('------------------   make sure the order are same ------------------')
        # print(label[0], label[1], label[2], label[3])


        if self.data_split == 'train':
            num_smaple = len(label)
            labeled_sample_num = int(len(label) * perceptange / 100.0)
            unlabeled_sample_num = len(label) - labeled_sample_num

            if partition == 'labeled':
                self.data, self.label, self.img_lst = data[unlabeled_sample_num:, :, :], label[unlabeled_sample_num:], img_lst[unlabeled_sample_num:]
            else:
                self.data, self.label, self.img_lst = data[:unlabeled_sample_num, :, :], label[:unlabeled_sample_num], img_lst[:unlabeled_sample_num]
        else:
            self.data, self.label, self.img_lst = data, label, img_lst

        print('the length of the dataset and label: ',len(label), len(img_lst))

        self.num_points = num_points

        self.img_train_transform = transforms.Compose([
            transforms.RandomCrop(224),
#             transforms.Resize(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.img_test_transform = transforms.Compose([
            transforms.CenterCrop(224),
#             transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def get_data(self, item):

        # Get Image Data first
        names = self.img_lst[item]
        names = names.split('/')
        
        #random select one image from the 12 images for each object
        
        img_idx_lst = [x for x in range(180)]
        random.shuffle(img_idx_lst)
        img_idx = img_idx_lst[0]
        img_names = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx)
        img = Image.open(img_names).convert('RGB')
        
        img_idx2 = img_idx_lst[1]
        img_name2 = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx2)
        img2 = Image.open(img_name2).convert('RGB')
        
        img_idx3 = img_idx_lst[2]
        img_name3 = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx3)
        img3 = Image.open(img_name3).convert('RGB')
        
        img_idx4 = img_idx_lst[3]
        img_name4 = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx4)
        img4 = Image.open(img_name4).convert('RGB')
        
        img_idx5 = img_idx_lst[4]
        img_name5 = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx5)
        img5 = Image.open(img_name5).convert('RGB')
        
        img_idx6 = img_idx_lst[5]
        img_name6 = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx6)
        img6 = Image.open(img_name6).convert('RGB')
        
        img_idx7 = img_idx_lst[6]
        img_name7 = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx7)
        img7 = Image.open(img_name7).convert('RGB')
        
        img_idx8 = img_idx_lst[7]
        img_name8 = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx8)
        img8 = Image.open(img_name8).convert('RGB')
        
        img_idx9 = img_idx_lst[8]
        img_name9 = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx9)
        img9 = Image.open(img_name9).convert('RGB')
        
        img_idx10 = img_idx_lst[9]
        img_name10 = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx10)
        img10 = Image.open(img_name10).convert('RGB')
        
        img_idx11 = img_idx_lst[10]
        img_name11 = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx11)
        img11 = Image.open(img_name11).convert('RGB')
        
        img_idx12 = img_idx_lst[11]
        img_name12 = '/scratch1/zhiminc/MVdata/dataset/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx12)
        img12 = Image.open(img_name12).convert('RGB')
        

        label = self.label[item]
        label = torch.LongTensor(label)


        pointcloud = self.data[item]
        choice1 = np.random.choice(len(pointcloud), self.num_points, replace=True)
        pt1 = pointcloud[choice1, :]

        choice2 = np.random.choice(len(pointcloud), self.num_points, replace=True)
        pt2 = pointcloud[choice2, :]


        if self.data_split == 'train':
            pt1 = random_scale(pt1, scale_low=0.8, scale_high=1.2)
            pt1 = translate_pointcloud(pt1)
            pt1 = rotate_pointcloud(pt1)
            pt1 = jitter_pointcloud(pt1)
            np.random.shuffle(pt1)

            pt2 = random_scale(pt2, scale_low=0.8, scale_high=1.2)
            pt2 = translate_pointcloud(pt2)
            pt2 = rotate_pointcloud(pt2)
            pt2 = jitter_pointcloud(pt2)
            np.random.shuffle(pt2)

            img = self.img_train_transform(img)
            img2 = self.img_train_transform(img2)
            img3 = self.img_train_transform(img3)
            img4 = self.img_train_transform(img4)
            img5 = self.img_train_transform(img5)
            img6 = self.img_train_transform(img6)
            img7 = self.img_train_transform(img7)
            img8 = self.img_train_transform(img8)
            img9 = self.img_train_transform(img9)
            img10 = self.img_train_transform(img10)
            img11 = self.img_train_transform(img11)
            img12 = self.img_train_transform(img12)

        else:
            img = self.img_test_transform(img)
            img2 = self.img_test_transform(img2)
            img3 = self.img_test_transform(img3)
            img4 = self.img_test_transform(img4)
            img5 = self.img_test_transform(img5)
            img6 = self.img_test_transform(img6)
            img7 = self.img_test_transform(img7)
            img8 = self.img_test_transform(img8)
            img9 = self.img_test_transform(img9)
            img10 = self.img_test_transform(img10)
            img11 = self.img_test_transform(img11)
            img12 = self.img_test_transform(img12)
        # print(pointcloud.shape, label, img.size())
        return pt1, pt2, label, img, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11,img12
#         return pt1, pt2, label, img, img2


    def get_mesh_raw_data(self, face, neighbor_index):
        # fill for n < max_faces with randomly picked faces
        max_faces = 1024
        num_point = len(face)
        if num_point < max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index



    def get_mesh(self, item):
        names = self.img_lst[item]
        names = names.split('/')
        mesh_path = '/scratch1/zhiminc/MVdata/dataset/ModelNet40_Mesh/' + names[0] + '/train/' + names[1][:-4] + '.npz'

        data = np.load(mesh_path)
        face = data['face']
        neighbor_index = data['neighbor_index']

        # data augmentation
        if self.data_split == 'train':
            sigma, clip = 0.01, 0.05
            jittered_data1 = np.clip(sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
            face1 = np.concatenate((face[:, :12] + jittered_data1, face[:, 12:]), 1)
        
        centers1, corners1, normals1, neighbor_index1 = self.get_mesh_raw_data(face1, neighbor_index)

        return centers1, corners1, normals1, neighbor_index1


    def check_exist(self, item):
        names = self.img_lst[item]
        names = names.split('/')
        # mesh_path = os.path.join('./dataset/ModelNet40_Mesh/', names[0], self.data_split,  names[1][:-4], '.npz' )
        mesh_path = '/scratch1/zhiminc/MVdata/dataset/ModelNet40_Mesh/' + names[0] + '/train/' + names[1][:-4] + '.npz'

        return os.path.isfile(mesh_path)

    def __getitem__(self, item):
        
        while not self.check_exist(item):
            idx = random.randint(0, len(self.data)-1)
            item = idx

        pt1, pt2, target, img, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11,img12 = self.get_data(item)
        # centers, corners, normals, neighbor_index = self.get_mesh(item)
        centers, corners, normals, neighbor_index  = self.get_mesh(item)

        pt1 = torch.from_numpy(pt1)
        pt2 = torch.from_numpy(pt2)

#         return pt1, pt2, img, img2, centers, corners, normals, neighbor_index, target
        return pt1, img, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11,img12, centers, corners, normals, neighbor_index, target

    def __len__(self):
        return self.data.shape[0]



if __name__ == '__main__':
    
    train_set = TripletDataloader(dataset = 'ModelNet40', num_points = 1024, partition='train')
    
    data_loader_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False,num_workers=1)
    
    cnt = 0
    for data in data_loader_loader:
        pt1, pt2, img1, img2, img1_v, img2_v, label1, label2, pos, neg  = data
        pt1 = pt1.numpy()
        pt1 = pt1[0,:,:]
        # print(pt1.shape)
        # print(pt1)
        # print(np.amin(pt1), np.amax(pt1))
        # showpoints(pt1)
