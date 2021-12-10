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

def extract(img_net, dgcnn, mesh_net, num_views, split, exp_name):
    
    dataset = 'ModelNet40'
    train_set = TestDataloader(dataset, num_points = 1024, num_views = 2, partition= split)

    print('length of the dataset: ', len(train_set))
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False,num_workers=1)
    
    start_time = time.time()

    img_feat = np.zeros((len(train_set), 256*4))
    pt_feat = np.zeros((len(train_set), 256*4))
    mesh_feat = np.zeros((len(train_set), 256*4))
    label = np.zeros((len(train_set)))
    img_pairs = 18

    
    for idx in range(img_pairs):
        print('index of image pair: ', idx)
        iteration = 0
        for data in data_loader: 
            # pt, img, _, target  = data
            pt, img, img_v, centers, corners, normals, neighbor_index, target = data
            
            img = Variable(img).to('cuda')
            img_v = Variable(img_v).to('cuda')

            #universal features
            img_rfeat, img_zfeat, img_spc = img_net(img)
            imgv_rfeat, imgv_zfeat, imgv_spc = img_net(img_v)

            imgF = torch.cat((img_rfeat, img_zfeat, img_spc), dim=1)
            imgvF = torch.cat((imgv_rfeat, imgv_zfeat, imgv_spc), dim=1)

            img_feat[iteration,:] = img_feat[iteration,:] + imgF.data.cpu().numpy() + imgvF.data.cpu().numpy()

            # print(iteration)
            iteration = iteration + 1
            if iteration % 400 == 0:
                print('iteration: ', iteration)


    iteration = 0
    for data in data_loader: 
        # pt, img, _, target  = data
        pt, img, img_v, centers, corners, normals, neighbor_index, target = data

        target = target[:,0]
        
        # img = Variable(img).to('cuda')

        pt = Variable(pt).to('cuda')
        
        centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
        corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
        normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
        neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))

        target = Variable(target).to('cuda')

        pt = pt.permute(0,2,1)
        
        #This part is for domian spcific features
        # cloud_feat, _ = dgcnn(pt)    
        # feat, _ = img_net(img)
        # # feat = img_net(img)
        # # feat = feat[:,:,0,0]
        # # # # print(feat.size())
        # m_feat, _ = mesh_net(centers, corners, normals, neighbor_index)

        #universal features
        cloud_rfeat, cloud_zfeat, cloud_spc = dgcnn(pt)
        # img_rfeat, img_zfeat, img_spc = img_net(img)
        mesh_rfeat, mesh_zfeat, mesh_specific = mesh_net(centers, corners, normals, neighbor_index)

        # print(cloud_rfeat.size(), cloud_zfeat.size(), cloud_spc.size())
        cloudF = torch.cat((cloud_rfeat, cloud_zfeat, cloud_spc), dim=1)
        # imgF = torch.cat((img_rfeat, img_zfeat, img_spc), dim=1)
        meshF = torch.cat((mesh_rfeat, mesh_zfeat, mesh_specific), dim=1)


        # m_feat = m_feat / torch.norm(m_feat)
        # cloud_feat = cloud_feat / torch.norm(cloud_feat)
        # feat = feat / torch.norm(feat)

        # print(cloud_feat.size(), img_feat.size())
        # img_feat[iteration,:] = imgF.data.cpu().numpy()
        pt_feat[iteration,:] = cloudF.data.cpu().numpy() 
        mesh_feat[iteration,:] = meshF.data.cpu().numpy()

        label[iteration] = target.data.cpu().numpy()
        # print(iteration)
        iteration = iteration + 1
        if iteration % 1000 == 0:
            print('iteration: ', iteration)

    #normalize the image features
    img_feat = img_feat/(2.0*img_pairs)

    feature_folder = os.path.join('./extracted_features', exp_name)
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
    
    img_feat_name = os.path.join('./extracted_features', exp_name, '%s-%s-%s_NtXent_img_feat'%(dataset, split, img_pairs))
    pt_feat_name =  os.path.join('./extracted_features', exp_name, '%s-%s_NtXent_cloud1024_feat'%(dataset, split))
    mesh_feat_name = os.path.join('./extracted_features', exp_name, '%s-%s_NtXent_mesh_feat'%(dataset, split))
    label_name =  os.path.join('./extracted_features', exp_name, '%s-%s_NtXent_label'%(dataset, split))

    np.save(img_feat_name, img_feat)
    np.save(pt_feat_name, pt_feat)
    np.save(mesh_feat_name, mesh_feat)
    np.save(label_name, label)


def extract_features(args):

    iterations = 10000
    num_views = 2          # 1 12 80
    # weights_folder = '180view-ModelNet40-pt-mesh-view-pt1024-img112-contrast-specific'
    # weights_folder = '180view-ModelNet40-pt-mesh-view-pt1024-img112-contrast-NoSpecific'
    weights_folder = 'ModelNet40-pt1024-mesh-img56-Xentropy-Xcontrast-PointMultiAgreement-T095-Fused-Warmup-100percent_xcenter_p10_nt_xw2_aw0_cw9_baseshare_newcenter_l2_100'

    img_net = SingleViewNet(pre_trained = None)
    # img_net = torch.nn.DataParallel(img_net)

    img_net_name = './checkpoints/%s/%d-img_net.pkl'%(weights_folder, iterations)

    img_net.load_state_dict(torch.load(img_net_name)['state_dict'])
    # img_net = models.resnet18(pretrained=True)
    # img_net = list(img_net.children())[:-1]
    # img_net = nn.Sequential(*img_net)
    # print(img_net)

    dgcnn = DGCNN(args)
    dgcnn_name = './checkpoints/%s/%d-pt_net.pkl'%(weights_folder, iterations)
    dgcnn.load_state_dict(torch.load(dgcnn_name)['state_dict'])
    # pt_net = PointnetPartSeg()

    mesh_net = MeshNet()
    mesh_net_name = './checkpoints/%s/%d-mesh_net.pkl'%(weights_folder, iterations)
    mesh_net.load_state_dict(torch.load(mesh_net_name)['state_dict'])

    # model = torch.load('./checkpoints/180view-ModelNet40-xentropy-center-pt-img/95000-head_net.pkl')
    
    img_net = img_net.eval()
    dgcnn = dgcnn.eval()
    mesh_net = mesh_net.eval()
    img_net = img_net.to('cuda')
    dgcnn = dgcnn.to('cuda')
    mesh_net = mesh_net.to('cuda')

    # print('extracing features for the training split')
    extract(img_net, dgcnn, mesh_net, num_views, 'train', exp_name = weights_folder)

    print('extracing features for the testing split')
    extract(img_net, dgcnn, mesh_net, num_views, 'test', exp_name = weights_folder)
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