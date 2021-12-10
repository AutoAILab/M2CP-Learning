# from tools.dual_dataloader import SingleViewDataloader, MultiViewDataloader
from tools.test_dataloader import TestDataloader
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
import torchvision.models as models
# from corrnet import SingleViewNet
from models.dgcnn import get_graph_feature
from models.dgcnn import DGCNN
from models.pointnet_part_seg import PointnetPartSeg
# from models.pointnet_part_seg import PointNet_Part
from models.meshnet import MeshNet
from models.SVCNN import Semi3D, SingleViewNet
from tools.triplet_dataloader import TripletDataloader
from tqdm import tqdm
from tools.utils import calculate_accuracy



def extract(dgcnn, num_views, split, exp_name):
    
    dataset = 'ModelNet40'

    # test_data_set = TripletDataloader(dataset = 'ModelNet40', num_points = 1024, partition='test',  perceptange = 10)
    test_data_set = TestDataloader(dataset, num_points = 1024, num_views = 2, partition= 'test')
    test_data_loader = torch.utils.data.DataLoader(test_data_set, batch_size= 1, shuffle = False, num_workers=1)

    print('length of the dataset: ', len(test_data_set))
    
    start_time = time.time()

    img_pred_lst = []
    pt_pred_lst = []
    mesh_pred_lst = []
    fused_pred_lst = []
    target_lst = []

    iteration = 0
    for data in test_data_loader:

        pt, img, imgV, centers, corners, normals, neighbor_index, target = data #the last one is the target
        
        pt = Variable(pt).to('cuda')

        img = Variable(img).to('cuda')
        imgV = Variable(imgV).to('cuda')

        centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
        corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
        normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
        neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))

        target = torch.squeeze(target, dim = 1)
        target = Variable(target).to('cuda')

        pt = pt.permute(0,2,1)

        pt_pred, pt_feat, pt_base = dgcnn(pt)
        # img_pred, img_feat, img_base = img_net(img, imgV)
        # mesh_pred, mesh_feat, mesh_base = mesh_net(centers, corners, normals, neighbor_index)

        # fused_pred =  mesh_pred * pt_pred


        pt_pred_lst.append(pt_pred.data.cpu())
        # mesh_pred_lst.append(mesh_pred.data.cpu())
        # img_pred_lst.append(img_pred.data.cpu())
        # fused_pred_lst.append(fused_pred.data.cpu())
        # print(target.size())
        target_lst.append(target.data.cpu())

        iteration = iteration + 1 
        if iteration % 400 == 0:
            print('iterations: ', iteration)

    pt_pred = torch.cat(pt_pred_lst, dim = 0)
    # mesh_pred = torch.cat(mesh_pred_lst, dim = 0)
    # img_pred = torch.cat(img_pred_lst, dim = 0)
    # fused_pred = torch.cat(fused_pred_lst, dim = 0)
    target = torch.cat(target_lst, dim = 0)

    print('pred size: ', pt_pred.size(), target.size())
    pt_acc = calculate_accuracy(pt_pred, target)

    print('the pt acc: ', pt_acc)



def extract_features(args):

    iterations = 6000
    num_views = 2          # 1 12 80

    # weights_folder = 'ModelNet40-pt1024-5percent-supervised'
    weights_folder = 'ModelNet40-pt1024-5percent-entropy-Icontrast'

    dgcnn = DGCNN(args)
    dgcnn = torch.nn.DataParallel(dgcnn)
    dgcnn_name = './checkpoints/%s/%d-pt_net.pkl'%(weights_folder, iterations)
    dgcnn.load_state_dict(torch.load(dgcnn_name)['state_dict'])

    
    # img_net = img_net.eval()
    dgcnn = dgcnn.eval()
    # mesh_net = mesh_net.eval()
    # img_net = img_net.to('cuda')
    
    dgcnn = dgcnn.to('cuda')
    # mesh_net = mesh_net.to('cuda')


    print('evaluation for the testing split')
    extract(dgcnn, num_views, 'test', exp_name = weights_folder)
    print('------------------ Al the Features are saved ---------------------------')

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='RGB and Point Cloud Correspondence')

    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')

    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    parser.add_argument('--gpu_id', type=str,  default='0',
                        help='GPU used to train the network')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    extract_features(args)