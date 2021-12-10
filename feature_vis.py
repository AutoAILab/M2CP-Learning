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
from models.SVCNN import Semi3D, SingleViewNet, FusionNet
from tools.triplet_dataloader import TripletDataloader
from tqdm import tqdm
from tools.utils import calculate_accuracy



def extract(img_net, dgcnn, mesh_net, fusionnet, num_views, split, exp_name):
    
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
    mean_pred_lst = []
    geometric_mean_pred_lst = []
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
        img_pred, img_feat, img_base = img_net(img, imgV)
        mesh_pred, mesh_feat, mesh_base = mesh_net(centers, corners, normals, neighbor_index)

        fused_pred = fusionnet(pt_base, mesh_base, img_base)

        mean_pred = pt_pred + img_pred + mesh_pred
        geometric_mean_pred = pt_pred * img_pred * mesh_pred


        pt_pred_lst.append(pt_pred.data.cpu())
        mesh_pred_lst.append(mesh_pred.data.cpu())
        img_pred_lst.append(img_pred.data.cpu())
        fused_pred_lst.append(fused_pred.data.cpu())

        mean_pred_lst.append(mean_pred.data.cpu())
        geometric_mean_pred_lst.append(geometric_mean_pred.data.cpu())
        # print(target.size())
        target_lst.append(target.data.cpu())

        iteration = iteration + 1 
        if iteration % 400 == 0:
            print('iterations: ', iteration)

    pt_pred = torch.cat(pt_pred_lst, dim = 0)
    mesh_pred = torch.cat(mesh_pred_lst, dim = 0)
    img_pred = torch.cat(img_pred_lst, dim = 0)
    fused_pred = torch.cat(fused_pred_lst, dim = 0)
    mean_pred = torch.cat(mean_pred_lst, dim = 0)
    geometric_mean_pred = torch.cat(geometric_mean_pred_lst, dim = 0)
    target = torch.cat(target_lst, dim = 0)

    print('pred size: ', img_pred.size(), mesh_pred.size(), target.size())
    img_acc = calculate_accuracy(img_pred, target)
    pt_acc = calculate_accuracy(pt_pred, target)
    mesh_acc = calculate_accuracy(mesh_pred, target)
    fused_acc = calculate_accuracy(fused_pred, target)
    mean_acc = calculate_accuracy(mean_pred, target)
    geometric_mean_acc = calculate_accuracy(geometric_mean_pred, target)

    
    print('the pt acc: %.4f'%(pt_acc))
    print('the img acc: %.4f'%(img_acc))
    print('the mesh acc: %.4f'%(mesh_acc))
    print('the fused acc: %.4f'%(fused_acc))
    print('the mean acc: %.4f'%(mean_acc))
    print('the geometric mean acc: %.4f'%(geometric_mean_acc))

#     img_acc = calculate_accuracy(img_pred, target)
#     pt_acc = calculate_accuracy(pt_pred, target)
#     mesh_acc = calculate_accuracy(mesh_pred, target)
#     fused_acc = calculate_accuracy(fused_pred, target)
#     # print(img_acc, pt_acc, mesh_acc)

#     img_acc_lst.append(img_acc)
#     pt_acc_lst.append(pt_acc)
#     mesh_acc_lst.append(mesh_acc)
#     fused_acc_lst.append(fused_acc)

#     iteration = iteration + 1
#     if iteration % 200 == 0:
#         print('the pt acc: ', sum(pt_acc_lst)/len(pt_acc_lst))
#         print('the img acc: ', sum(img_acc_lst)/len(img_acc_lst))
#         print('the mesh acc: ', sum(mesh_acc_lst)/len(mesh_acc_lst))
#         print('the fused acc: ', sum(fused_acc_lst)/len(fused_acc_lst))    
#         print('-------------------------------------------------')      

# print('the pt acc: ', sum(pt_acc_lst)/len(pt_acc_lst))
# print('the img acc: ', sum(img_acc_lst)/len(img_acc_lst))
# print('the mesh acc: ', sum(mesh_acc_lst)/len(mesh_acc_lst))
# print('the fused acc: ', sum(fused_acc_lst)/len(fused_acc_lst))


def extract_features(args):

    iterations = 4000
    num_views = 2          # 1 12 80

    # weights_folder = 'ModelNet40-pt1024-mesh-img56-Xentropy-Xcontrast-MultiAgreement-T095-Fused-Warmup-2percent'
    # weights_folder = 'ModelNet40-pt1024-mesh-img56-Xentropy-2percent-supervised'
    weights_folder = 'ModelNet40-pt1024-mesh-img56-Xentropy-Xcontrast-PointMultiAgreement-T095-Fused-Warmup-2percent_xcenter_p10_warmup1_0001'

    img_net = SingleViewNet(pre_trained = True)
    # img_net = torch.nn.DataParallel(img_net)

    img_net_name = './checkpoints/%s/%d-img_net.pkl'%(weights_folder, iterations)
    img_net.load_state_dict(torch.load(img_net_name)['state_dict'])

    dgcnn = DGCNN(args)
    dgcnn_name = './checkpoints/%s/%d-pt_net.pkl'%(weights_folder, iterations)
    dgcnn.load_state_dict(torch.load(dgcnn_name)['state_dict'])

    mesh_net = MeshNet()
    mesh_net_name = './checkpoints/%s/%d-mesh_net.pkl'%(weights_folder, iterations)
    mesh_net.load_state_dict(torch.load(mesh_net_name)['state_dict'])
    
    fusion_net = FusionNet()
    fusion_net_name = './checkpoints/%s/%d-fusion_net.pkl'%(weights_folder, iterations)
    fusion_net.load_state_dict(torch.load(fusion_net_name)['state_dict'])


    img_net = img_net.eval()
    dgcnn = dgcnn.eval()
    mesh_net = mesh_net.eval()
    fusion_net = fusion_net.eval()

    img_net = img_net.to('cuda')
    dgcnn = dgcnn.to('cuda')
    mesh_net = mesh_net.to('cuda')
    fusion_net = fusion_net.to('cuda')


    print('evaluation for the testing split')
    extract(img_net, dgcnn, mesh_net, fusion_net, num_views, 'test', exp_name = weights_folder)
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