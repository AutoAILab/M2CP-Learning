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
from tools.visualize import showpoints
from tools.dual_dataloader import load_data, load_modelnet10_data


class TestDataloader(Dataset):
    def __init__(self, dataset, num_points, num_views, partition='test'):
        self.dataset = dataset
        if self.dataset == 'ModelNet40':
            self.data, self.label, self.img_lst = load_data(partition)
        else:
            self.data, self.label, self.img_lst = load_modelnet10_data(partition)

        self.num_points = num_points
        self.num_views = num_views
        self.partition = partition

        self.img_transform = transforms.Compose([
#             transforms.CenterCrop(224),
            transforms.Resize(224),
            # transforms.Resize((450, 800)),
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
        img_names = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx)
        img = Image.open(img_names).convert('RGB')
        
        img_idx2 = img_idx_lst[1]
        img_name2 = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx2)
        img2 = Image.open(img_name2).convert('RGB')
        
        img_idx3 = img_idx_lst[2]
        img_name3 = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx3)
        img3 = Image.open(img_name3).convert('RGB')
        
        img_idx4 = img_idx_lst[3]
        img_name4 = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx4)
        img4 = Image.open(img_name4).convert('RGB')
        
        img_idx5 = img_idx_lst[4]
        img_name5 = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx5)
        img5 = Image.open(img_name5).convert('RGB')
        
        img_idx6 = img_idx_lst[5]
        img_name6 = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx6)
        img6 = Image.open(img_name6).convert('RGB')
        
        img_idx7 = img_idx_lst[6]
        img_name7 = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx7)
        img7 = Image.open(img_name7).convert('RGB')
        
        img_idx8 = img_idx_lst[7]
        img_name8 = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx8)
        img8 = Image.open(img_name8).convert('RGB')
        
        img_idx9 = img_idx_lst[8]
        img_name9 = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx9)
        img9 = Image.open(img_name9).convert('RGB')
        
        img_idx10 = img_idx_lst[9]
        img_name10 = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx10)
        img10 = Image.open(img_name10).convert('RGB')
        
        img_idx11 = img_idx_lst[10]
        img_name11 = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx11)
        img11 = Image.open(img_name11).convert('RGB')
        
        img_idx12 = img_idx_lst[11]
        img_name12 = '/scratch1/zhiminc/MVdata/data/%s-Images-180/%s/%s/%s.%d.png' % (self.dataset, names[0], names[1][:-4], names[1][:-4], img_idx12)
        img12 = Image.open(img_name12).convert('RGB')
        


        label = self.label[item]

        pointcloud = self.data[item]
        choice = np.random.choice(len(pointcloud), self.num_points, replace=True)
        pointcloud = pointcloud[choice, :]

        img = self.img_transform(img)
        img2 = self.img_transform(img2)
        img3 = self.img_transform(img3)
        img4 = self.img_transform(img4)
        img5 = self.img_transform(img5)
        img6 = self.img_transform(img6)
        img7 = self.img_transform(img7)
        img8 = self.img_transform(img8)
        img9 = self.img_transform(img9)
        img10 = self.img_transform(img10)
        img11 = self.img_transform(img11)
        img12 = self.img_transform(img12)        
        
        return pointcloud, label, img, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11,img12


    def check_exist(self, item):
        names = self.img_lst[item]
        names = names.split('/')
        # mesh_path = './dataset/ModelNet40_Mesh/' + names[0] + '/train/' + names[1][:-4] + '.npz'
        mesh_path = os.path.join('/scratch1/zhiminc/MVdata/data/ModelNet40_Mesh/', names[0], self.partition, '%s.npz'%(names[1][:-4]))

        return os.path.isfile(mesh_path)


    def get_mesh(self, item):
        names = self.img_lst[item]
        names = names.split('/')
        # mesh_path = './dataset/ModelNet40_Mesh/' + names[0] + '/train/' + names[1][:-4] + '.npz'
        mesh_path = os.path.join('/scratch1/zhiminc/MVdata/data/ModelNet40_Mesh/', names[0], self.partition, '%s.npz'%(names[1][:-4]))

        data = np.load(mesh_path)
        face = data['face']
        neighbor_index = data['neighbor_index']

        # no data augmentation in the testing dataloader
        # if self.partition == 'train':
        #     sigma, clip = 0.01, 0.05
        #     jittered_data = np.clip(sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
        #     face = np.concatenate((face[:, :12] + jittered_data, face[:, 12:]), 1)

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

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index

    def __getitem__(self, item):
        #name of this point cloud

        while not self.check_exist(item):
            idx = random.randint(0, len(self.data)-1)
            item = idx

        pt, target, img, img_v, img3, img4, img5, img6, img7, img8, img9, img10, img11,img12 = self.get_data(item)
        pt = torch.from_numpy(pt)

        centers, corners, normals, neighbor_index = self.get_mesh(item)

        # return pt1, img1, img2, label1
        return pt, img, img_v,img3, img4, img5, img6, img7, img8, img9, img10, img11,img12, centers, corners, normals, neighbor_index, target

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    
    train_set = SingleViewDataloader(num_points = 1024, partition='train')
    
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
