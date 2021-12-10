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
from models.SVCNN import Semi3D, SingleViewNet, FusionNet, FusionHead
from tools.triplet_dataloader import TripletDataloader
from tqdm import tqdm
from tools.utils import calculate_accuracy
from sklearn.preprocessing import normalize
import scipy

def fx_calc_map_label(image, text, label, k = 0, dist_method='euclidean'):
  if dist_method == 'euclidean':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)

def compute_topk(X_train, y_train, X_test, y_test):
    print('X_train range: ', np.min(X_train), np.amax(X_train))
    print('X_test  range: ', np.min(X_test), np.amax(X_test))

    ks = [1, 5, 10, 20, 50]
    topk_correct = {k:0 for k in ks}

    distances = euclidean_distances(X_test, X_train)
    # distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)

    for k in ks:
        # print(k)
        top_k_indices = indices[:, :k]
        # print(top_k_indices.shape, y_test.shape)
        for ind, test_label in zip(top_k_indices, y_test):
            labels = y_train[ind]
            if test_label in labels:
                # print(test_label)
                topk_correct[k] += 1

    for k in ks:
        correct = topk_correct[k]
        total = len(X_test)
        print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct/total))





    max_rank = 100

    qf = X_test
    gf = X_train
    g_pids = y_train
    q_pids = y_test

    # qf = torch.from_numpy(X_test)
   
    # gf = torch.from_numpy(X_train) 
    # g_pids = torch.from_numpy(y_train)  
    # q_pids = torch.from_numpy(y_test)  


    # m, n = qf.size(0), gf.size(0)
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())

    distmat = euclidean_distances(qf, gf)

    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    
    for q_idx in range(num_q):
        # print('XXXXXXXXXXXXXXX')
        # get query pid and camid
        # q_pid = q_pids[q_idx]
        # print('q_pid === ', q_pid)
        # q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        # order = indices[q_idx]
        # remove = (g_pids[order] == q_pid)
        # keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    # print('all_ap ====', len(all_AP))
    mAP = np.mean(all_AP)
    # print(all_cmc, mAP)
    print('the mAP is: ',mAP)

    return all_cmc, mAP

def extract(img_net, dgcnn, mesh_net, fusionnet, fusion_head, num_views, split, exp_name):
    
    dataset = 'ModelNet40'

    # test_data_set = TripletDataloader(dataset = 'ModelNet40', num_points = 1024, partition='test',  perceptange = 10)
    test_data_set = TestDataloader(dataset, num_points = 1024, num_views = 2, partition= 'test')
    test_data_loader = torch.utils.data.DataLoader(test_data_set, batch_size= 1, shuffle = False, num_workers=1)

#     labeled_set = TripletDataloader(dataset = 'ModelNet40', num_points = args.num_points, partition='labeled',  perceptange = 8)
#     labeled_data_loader = torch.utils.data.DataLoader(labeled_set, batch_size= 1, shuffle=True,num_workers=8, drop_last=True) 
    
    print('length of the dataset: ', len(test_data_set))
    
    start_time = time.time()

    img_pred_lst = []
    pt_pred_lst = []
    mesh_pred_lst = []
    fused_pred_lst = []
    mean_pred_lst = []
    geometric_mean_pred_lst = []
    target_lst = []
    img_feat_lst = []
    pt_feat_lst = []
    mesh_feat_lst = []
    fused_feat_lst = []
    pt_base_lst = []
    mesh_base_lst = []
    img_base_lst = []
    img_gfeat_lst = []
    pt_gfeat_lst = []
    mesh_gfeat_lst = []
    
    pt_unmatch_lst = []
    mesh_unmatch_lst = []
    img_unmatch = []
    mesh_gfeat_lst = []
    arg_pred_lst = []
    
    iteration = 0
#     for data in labeled_data_loader:
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

        pt_base = dgcnn(pt)
        img_base = img_net(img, imgV)
        mesh_base = mesh_net(centers, corners, normals, neighbor_index)

        fused_pred = fusionnet(pt_base, mesh_base, img_base)
        
        img_pred, pt_pred, mesh_pred, img_feat, pt_feat, mesh_feat, img_gfeat, pt_gfeat, mesh_gfeat = fusion_head(pt_base, mesh_base, img_base) 

        geometric_mean_pred = pt_pred * img_pred * mesh_pred
        
        mean_pred = pt_pred + img_pred + mesh_pred
        
        arg_pred = torch.max(pt_pred, img_pred)
        arg_pred = torch.max(arg_pred, mesh_pred)


        pt_pred_lst.append(pt_pred.data.cpu())
        mesh_pred_lst.append(mesh_pred.data.cpu())
        img_pred_lst.append(img_pred.data.cpu())
        fused_pred_lst.append(fused_pred.data.cpu())
        
        pt_feat_lst.append(pt_feat.data.cpu())
        mesh_feat_lst.append(mesh_feat.data.cpu())
        img_feat_lst.append(img_feat.data.cpu())
        
        pt_base_lst.append(pt_base.data.cpu())
        mesh_base_lst.append(mesh_base.data.cpu())
        img_base_lst.append(img_base.data.cpu())
        
        pt_gfeat_lst.append(pt_gfeat.data.cpu())
        mesh_gfeat_lst.append(mesh_gfeat.data.cpu())
        img_gfeat_lst.append(img_gfeat.data.cpu())
        
        mean_pred_lst.append(mean_pred.data.cpu())
        arg_pred_lst.append(arg_pred.data.cpu())


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
    
    pt_feat = torch.cat(pt_feat_lst, dim = 0)
    mesh_feat = torch.cat(mesh_feat_lst, dim = 0)
    img_feat = torch.cat(img_feat_lst, dim = 0)
    target_new = target.view(-1,1)
    
    pt_gfeat = torch.cat(pt_gfeat_lst, dim = 0)
    mesh_gfeat = torch.cat(mesh_gfeat_lst, dim = 0)
    img_gfeat = torch.cat(img_gfeat_lst, dim = 0)
    
    pt_base = torch.cat(pt_base_lst, dim = 0)
    mesh_base = torch.cat(mesh_base_lst, dim = 0)
    img_base = torch.cat(img_base_lst, dim = 0)
    arg_pred = torch.cat(arg_pred_lst, dim = 0)

#     output_pt = torch.cat([pt_feat, target_new], dim=1)
#     output_mesh = torch.cat([mesh_feat, target_new], dim=1)
#     output_img = torch.cat([img_feat, target_new], dim=1)

#     output_pt = torch.cat([pt_gfeat, target_new], dim=1)
#     output_mesh = torch.cat([mesh_fgeat, target_new], dim=1)
#     output_img = torch.cat([img_gfeat, target_new], dim=1)
    
#     output_pt = torch.cat([pt_base, target_new], dim=1)
#     output_mesh = torch.cat([mesh_base, target_new], dim=1)
#     output_img = torch.cat([img_base, target_new], dim=1)
    
#     np.save('output_pt_10_l2', output_pt)
#     np.save('output_mesh_10_l2', output_mesh)
#     np.save('output_img_10_l2', output_img)

#     np.save('output_pt_02_cos', output_pt)
#     np.save('output_mesh_02_cos', output_mesh)
#     np.save('output_img_02_cos', output_img)

    
    print('pred size: ', img_pred.size(), mesh_pred.size(), target.size())
    img_acc = calculate_accuracy(img_pred, target)
    pt_acc = calculate_accuracy(pt_pred, target)
    mesh_acc = calculate_accuracy(mesh_pred, target)
    fused_acc = calculate_accuracy(fused_pred, target)
    mean_acc = calculate_accuracy(mean_pred, target)
    geometric_mean_acc = calculate_accuracy(geometric_mean_pred, target)
    arg_acc = calculate_accuracy(arg_pred, target)
    
    print('the pt acc: %.4f'%(pt_acc))
    print('the img acc: %.4f'%(img_acc))
    print('the mesh acc: %.4f'%(mesh_acc))
    print('the fused acc: %.4f'%(fused_acc))
    print('the mean acc: %.4f'%(mean_acc))
    print('the arg acc: %.4f'%(arg_acc))

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
    print('*******************************************************************************')
    print('performance of base feature accuracy')
    
    pt_base_lst = torch.cat(pt_base_lst, dim=0)
    pt_base_lst = pt_base_lst.squeeze(1)

    mesh_base_lst = torch.cat(mesh_base_lst, dim=0)
    mesh_base_lst = mesh_base_lst.squeeze(1)

    img_base_lst = torch.cat(img_base_lst, dim=0)
    img_base_lst = img_base_lst.squeeze(1)
    
    pt_gfeat_lst = torch.cat(pt_gfeat_lst, dim=0)
    pt_gfeat_lst = np.array(pt_gfeat_lst)
    
    mesh_gfeat_lst = torch.cat(mesh_gfeat_lst, dim=0)
    mesh_gfeat_lst = np.array(mesh_gfeat_lst)
    
    img_gfeat_lst = torch.cat(img_gfeat_lst, dim=0)
    img_gfeat_lst = np.array(img_gfeat_lst)


    test_pt_X = normalize(pt_base_lst, norm='l1', axis=1)
    test_img_X = normalize(img_base_lst, norm='l1', axis=1)
    test_mesh_X = normalize(mesh_base_lst, norm='l1', axis=1)
    
#     test_pt_X = normalize(pt_gfeat_lst, norm='l1', axis=1)
#     test_img_X = normalize(img_gfeat_lst, norm='l1', axis=1)
#     test_mesh_X = normalize(mesh_gfeat_lst, norm='l1', axis=1)
    
    
    test_label = target
#     train_pt_X = train_pt_X[:512]
#     test_pt_X = test_pt_X[:512]
#     train_img_X = train_img_X[:512]
#     test_img_X = test_img_X[:512]
#     train_mesh_X = train_mesh_X[:512]
#     test_mesh_X = test_mesh_X[:512]

    print('---------  image -> image --------')
    # compute_topk(train_img_X, label_train, test_img_X, label_test)
    img2img = fx_calc_map_label(test_img_X, test_img_X, test_label, k = 0, dist_method='COS')
    print('%.4f'%img2img)

    print('---------  cloud -> cloud --------')
    # compute_topk(train_pt_X, label_train, test_pt_X, test_label)
    pt2pt = fx_calc_map_label(test_pt_X, test_pt_X, test_label, k = 0, dist_method='COS')
    print('%.4f'%pt2pt)


    print('---------  mesh -> mesh --------')
    # compute_topk(train_mesh_X, label_train, test_mesh_X, test_label)
    mesh2mesh = fx_calc_map_label(test_mesh_X, test_mesh_X, test_label, k = 0, dist_method='COS')
    print('%.4f'%mesh2mesh)

    print('---------  image -> point cloud --------')
    # compute_topk(train_img_X, label_train, test_pt_X, test_label)
    img2pt = fx_calc_map_label(test_img_X, test_pt_X, test_label, k = 0, dist_method='COS')
    print('%.4f'%img2pt)

    print('---------  image -> mesh --------')
    # compute_topk(train_img_X, label_train, test_mesh_X, test_label)
    img2mesh = fx_calc_map_label(test_img_X, test_mesh_X, test_label, k = 0, dist_method='COS')
    print('%.4f'%img2mesh)

    print('---------  point cloud -> image --------')
    # compute_topk(train_pt_X, label_train, test_img_X, test_label)
    pt2img = fx_calc_map_label(test_pt_X, test_img_X, test_label, k = 0, dist_method='COS')
    print('%.4f'%pt2img)

    print('---------  point cloud -> mesh --------')
    # compute_topk(train_pt_X, label_train, test_mesh_X, test_label)
    pt2mesh = fx_calc_map_label(test_pt_X, test_mesh_X, test_label, k = 0, dist_method='COS')
    print('%.4f'%pt2mesh)

    print('---------  mesh -> images --------')
    # compute_topk(train_mesh_X, label_train, test_img_X, test_label)
    mesh2img = fx_calc_map_label(test_mesh_X, test_img_X, test_label, k = 0, dist_method='COS')
    print('%.4f'%mesh2img)

    print('---------  mesh -> point cloud --------')
    # compute_topk(train_mesh_X, label_train, test_pt_X, test_label)
    mesh2pt = fx_calc_map_label(test_mesh_X, test_pt_X, test_label, k = 0, dist_method='COS')
    print('%.4f'%mesh2pt)
def extract_features(args):

    iterations = 100000
    num_views = 12          # 1 12 80

    # weights_folder = 'ModelNet40-pt1024-mesh-img56-Xentropy-Xcontrast-MultiAgreement-T095-Fused-Warmup-2percent'
    # weights_folder = 'ModelNet40-pt1024-mesh-img56-Xentropy-2percent-supervised'
    weights_folder = 'ModelNet40_p10_nt_xw2_aw0_cw9_baseshare_Cos_12views_imgsize224_10w_weight121'

    img_net = SingleViewNet(pre_trained = True)
    # img_net = torch.nn.DataParallel(img_net)

    img_net_name = './checkpoints/%s/%d-img_net.pkl'%(weights_folder, iterations)
    img_net.load_state_dict(torch.load(img_net_name,)['state_dict'],  strict=False)

    dgcnn = DGCNN(args)
    dgcnn_name = './checkpoints/%s/%d-pt_net.pkl'%(weights_folder, iterations)
    dgcnn.load_state_dict(torch.load(dgcnn_name)['state_dict'],  strict=False)

    mesh_net = MeshNet()
    mesh_net_name = './checkpoints/%s/%d-mesh_net.pkl'%(weights_folder, iterations)
    mesh_net.load_state_dict(torch.load(mesh_net_name)['state_dict'],  strict=False)
    
    fusion_net = FusionNet()
    fusion_net_name = './checkpoints/%s/%d-fusion_net.pkl'%(weights_folder, iterations)
    fusion_net.load_state_dict(torch.load(fusion_net_name)['state_dict'],  strict=False)

    fusion_head = FusionHead()
    fusion_head_name = './checkpoints/%s/%d-fusion_head.pkl'%(weights_folder, iterations)
    fusion_head.load_state_dict(torch.load(fusion_head_name)['state_dict'],  strict=False)
    
    
    img_net = img_net.eval()
    dgcnn = dgcnn.eval()
    mesh_net = mesh_net.eval()
    fusion_net = fusion_net.eval()
    fusion_head = fusion_head.eval()

    img_net = img_net.to('cuda')
    dgcnn = dgcnn.to('cuda')
    mesh_net = mesh_net.to('cuda')
    fusion_net = fusion_net.to('cuda')
    fusion_head = fusion_head.to('cuda')
   


    print('evaluation for the testing split')
    extract(img_net, dgcnn, mesh_net, fusion_net, fusion_head, num_views, 'test', exp_name = weights_folder)
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