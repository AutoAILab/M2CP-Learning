
import numpy as np
import copy
from collections import defaultdict
import sys
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
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



# how to calculate mAP
def topk_retrieval():
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """


    img_pairs = 18
    dataset = 'ModelNet40'
    exp_name = 'ModelNet40-pt1024-mesh-img56-Xentropy-Xcontrast-PointMultiAgreement-T095-Fused-Warmup-2percent_task4_p2_w2'

    train_img_X_name = './extracted_features/%s/%s-%s-%s_NtXent_img_feat.npy'%(exp_name, dataset, 'train', img_pairs)
    test_img_X_name = './extracted_features/%s/%s-%s-%s_NtXent_img_feat.npy'%(exp_name, dataset, 'test', img_pairs)

    train_pt_X_name = './extracted_features/%s/%s-%s_NtXent_cloud1024_feat.npy'%(exp_name, dataset, 'train')
    test_pt_X_name = './extracted_features/%s/%s-%s_NtXent_cloud1024_feat.npy'%(exp_name, dataset, 'test')

    train_mesh_X_name = './extracted_features/%s/%s-%s_NtXent_mesh_feat.npy'%(exp_name, dataset, 'train')
    test_mesh_X_name = './extracted_features/%s/%s-%s_NtXent_mesh_feat.npy'%(exp_name, dataset, 'test')

    train_label_name = './extracted_features/%s/%s-%s_NtXent_label.npy'%(exp_name, dataset, 'train')
    test_label_name = './extracted_features/%s/%s-%s_NtXent_label.npy'%(exp_name, dataset, 'test')



    train_pt_X = np.load(train_pt_X_name)
    test_pt_X = np.load(test_pt_X_name)

    train_img_X = np.load(train_img_X_name)
    test_img_X = np.load(test_img_X_name)

    train_mesh_X = np.load(train_mesh_X_name)
    test_mesh_X = np.load(test_mesh_X_name)


    train_label = np.load(train_label_name)
    test_label = np.load(test_label_name)

    print('--------------------------------------------')
    print('The value of features for point cloud should be from -2 to 2')
    print(np.amin(train_pt_X), np.amax(train_pt_X))
    print('--------------------------------------------')
    print('The value of features for mesh should be from -2 to 2')
    print(np.amin(train_mesh_X), np.amax(train_mesh_X))
    print('--------------------------------------------')
    print('The value of features forimage features')
    print(np.amin(train_img_X), np.amax(train_img_X))
    print('--------------------------------------------')


    print('**********************************************************')
    print('performance of base feature accuracy')

    train_pt_X = normalize(train_pt_X[:512], norm='l1', axis=1)
    test_pt_X = normalize(test_pt_X[:512], norm='l1', axis=1)
    train_img_X = normalize(train_img_X[:512], norm='l1', axis=1)
    test_img_X = normalize(test_img_X[:512], norm='l1', axis=1)
    train_mesh_X = normalize(train_mesh_X[:512], norm='l1', axis=1)
    test_mesh_X = normalize(test_mesh_X[:512], norm='l1', axis=1)
    
#     train_pt_X = train_pt_X[:512]
#     test_pt_X = test_pt_X[:512]
#     train_img_X = train_img_X[:512]
#     test_img_X = test_img_X[:512]
#     train_mesh_X = train_mesh_X[:512]
#     test_mesh_X = test_mesh_X[:512]

    print('---------  image -> image --------')
    # compute_topk(train_img_X, label_train, test_img_X, label_test)
    img2img = fx_calc_map_label(test_img_X, test_img_X, test_label, k = 0, dist_method='euclidean')
    print(img2img)

    print('---------  cloud -> cloud --------')
    # compute_topk(train_pt_X, label_train, test_pt_X, test_label)
    pt2pt = fx_calc_map_label(test_pt_X, test_pt_X, test_label, k = 0, dist_method='euclidean')
    print(pt2pt)


    print('---------  mesh -> mesh --------')
    # compute_topk(train_mesh_X, label_train, test_mesh_X, test_label)
    mesh2mesh = fx_calc_map_label(test_mesh_X, test_mesh_X, test_label, k = 0, dist_method='euclidean')
    print(mesh2mesh)

    print('---------  image -> point cloud --------')
    # compute_topk(train_img_X, label_train, test_pt_X, test_label)
    img2pt = fx_calc_map_label(test_img_X, test_pt_X, test_label, k = 0, dist_method='euclidean')
    print(img2pt)

    print('---------  image -> mesh --------')
    # compute_topk(train_img_X, label_train, test_mesh_X, test_label)
    img2mesh = fx_calc_map_label(test_img_X, test_mesh_X, test_label, k = 0, dist_method='euclidean')
    print(img2mesh)

    print('---------  point cloud -> image --------')
    # compute_topk(train_pt_X, label_train, test_img_X, test_label)
    pt2img = fx_calc_map_label(test_pt_X, test_img_X, test_label, k = 0, dist_method='euclidean')
    print(pt2img)

    print('---------  point cloud -> mesh --------')
    # compute_topk(train_pt_X, label_train, test_mesh_X, test_label)
    pt2mesh = fx_calc_map_label(test_pt_X, test_mesh_X, test_label, k = 0, dist_method='euclidean')
    print(pt2mesh)

    print('---------  mesh -> images --------')
    # compute_topk(train_mesh_X, label_train, test_img_X, test_label)
    mesh2img = fx_calc_map_label(test_mesh_X, test_img_X, test_label, k = 0, dist_method='euclidean')
    print(mesh2img)

    print('---------  mesh -> point cloud --------')
    # compute_topk(train_mesh_X, label_train, test_pt_X, test_label)
    mesh2pt = fx_calc_map_label(test_mesh_X, test_pt_X, test_label, k = 0, dist_method='euclidean')
    print(mesh2pt)


    print('*******************************************************************')
    print('performance of invariant feature')

    train_pt_X = normalize(train_pt_X[:, 512:-256], norm='l1', axis=1)
    test_pt_X = normalize(test_pt_X[:, 512:-256], norm='l1', axis=1)
    train_img_X = normalize(train_img_X[:, 512:-256], norm='l1', axis=1)
    test_img_X = normalize(test_img_X[:, 512:-256], norm='l1', axis=1)
    train_mesh_X = normalize(train_mesh_X[:, 512:-256], norm='l1', axis=1)
    test_mesh_X = normalize(test_mesh_X[:, 512:-256], norm='l1', axis=1)
    
#     train_pt_X = train_pt_X[:, 512:-256]
#     test_pt_X = test_pt_X[:, 512:-256]
#     train_img_X = train_img_X[:, 512:-256]
#     test_img_X = test_img_X[:, 512:-256]
#     train_mesh_X = train_mesh_X[:, 512:-256]
#     test_mesh_X = test_mesh_X[:, 512:-256]

    print('---------  image -> image --------')
    # compute_topk(train_img_X, label_train, test_img_X, label_test)
    img2img = fx_calc_map_label(test_img_X, test_img_X, test_label, k = 0, dist_method='euclidean')
    print(img2img)

    print('---------  cloud -> cloud --------')
    # compute_topk(train_pt_X, label_train, test_pt_X, test_label)
    pt2pt = fx_calc_map_label(test_pt_X, test_pt_X, test_label, k = 0, dist_method='euclidean')
    print(pt2pt)


    print('---------  mesh -> mesh --------')
    # compute_topk(train_mesh_X, label_train, test_mesh_X, test_label)
    mesh2mesh = fx_calc_map_label(test_mesh_X, test_mesh_X, test_label, k = 0, dist_method='euclidean')
    print(mesh2mesh)

    print('---------  image -> point cloud --------')
    # compute_topk(train_img_X, label_train, test_pt_X, test_label)
    img2pt = fx_calc_map_label(test_img_X, test_pt_X, test_label, k = 0, dist_method='euclidean')
    print(img2pt)

    print('---------  image -> mesh --------')
    # compute_topk(train_img_X, label_train, test_mesh_X, test_label)
    img2mesh = fx_calc_map_label(test_img_X, test_mesh_X, test_label, k = 0, dist_method='euclidean')
    print(img2mesh)

    print('---------  point cloud -> image --------')
    # compute_topk(train_pt_X, label_train, test_img_X, test_label)
    pt2img = fx_calc_map_label(test_pt_X, test_img_X, test_label, k = 0, dist_method='euclidean')
    print(pt2img)

    print('---------  point cloud -> mesh --------')
    # compute_topk(train_pt_X, label_train, test_mesh_X, test_label)
    pt2mesh = fx_calc_map_label(test_pt_X, test_mesh_X, test_label, k = 0, dist_method='euclidean')
    print(pt2mesh)

    print('---------  mesh -> images --------')
    # compute_topk(train_mesh_X, label_train, test_img_X, test_label)
    mesh2img = fx_calc_map_label(test_mesh_X, test_img_X, test_label, k = 0, dist_method='euclidean')
    print(mesh2img)

    print('---------  mesh -> point cloud --------')
    # compute_topk(train_mesh_X, label_train, test_pt_X, test_label)
    mesh2pt = fx_calc_map_label(test_mesh_X, test_pt_X, test_label, k = 0, dist_method='euclidean')
    print(mesh2pt)


    print('*******************************************************************')
    print('performance of modality-specific feature')

    train_pt_X = normalize(train_pt_X[:, -256:], norm='l1', axis=1)
    test_pt_X = normalize(test_pt_X[:, -256:], norm='l1', axis=1)
    train_img_X = normalize(train_img_X[:, -256:], norm='l1', axis=1)
    test_img_X = normalize(test_img_X[:, -256:], norm='l1', axis=1)
    train_mesh_X = normalize(train_mesh_X[:, -256:], norm='l1', axis=1)
    test_mesh_X = normalize(test_mesh_X[:, -256:], norm='l1', axis=1)
    
#     train_pt_X = train_pt_X[:, -256:]
#     test_pt_X = test_pt_X[:, -256:]
#     train_img_X = train_img_X[:, -256:]
#     test_img_X = test_img_X[:, -256:]
#     train_mesh_X = train_mesh_X[:, -256:]
#     test_mesh_X = test_mesh_X[:, -256:]

    print('---------  image -> image --------')
    # compute_topk(train_img_X, label_train, test_img_X, label_test)
    img2img = fx_calc_map_label(test_img_X, test_img_X, test_label, k = 0, dist_method='euclidean')
    print(img2img)

    print('---------  cloud -> cloud --------')
    # compute_topk(train_pt_X, label_train, test_pt_X, test_label)
    pt2pt = fx_calc_map_label(test_pt_X, test_pt_X, test_label, k = 0, dist_method='euclidean')
    print(pt2pt)


    print('---------  mesh -> mesh --------')
    # compute_topk(train_mesh_X, label_train, test_mesh_X, test_label)
    mesh2mesh = fx_calc_map_label(test_mesh_X, test_mesh_X, test_label, k = 0, dist_method='euclidean')
    print(mesh2mesh)

    print('---------  image -> point cloud --------')
    # compute_topk(train_img_X, label_train, test_pt_X, test_label)
    img2pt = fx_calc_map_label(test_img_X, test_pt_X, test_label, k = 0, dist_method='euclidean')
    print(img2pt)

    print('---------  image -> mesh --------')
    # compute_topk(train_img_X, label_train, test_mesh_X, test_label)
    img2mesh = fx_calc_map_label(test_img_X, test_mesh_X, test_label, k = 0, dist_method='euclidean')
    print(img2mesh)

    print('---------  point cloud -> image --------')
    # compute_topk(train_pt_X, label_train, test_img_X, test_label)
    pt2img = fx_calc_map_label(test_pt_X, test_img_X, test_label, k = 0, dist_method='euclidean')
    print(pt2img)

    print('---------  point cloud -> mesh --------')
    # compute_topk(train_pt_X, label_train, test_mesh_X, test_label)
    pt2mesh = fx_calc_map_label(test_pt_X, test_mesh_X, test_label, k = 0, dist_method='euclidean')
    print(pt2mesh)

    print('---------  mesh -> images --------')
    # compute_topk(train_mesh_X, label_train, test_img_X, test_label)
    mesh2img = fx_calc_map_label(test_mesh_X, test_img_X, test_label, k = 0, dist_method='euclidean')
    print(mesh2img)

    print('---------  mesh -> point cloud --------')
    # compute_topk(train_mesh_X, label_train, test_pt_X, test_label)
    mesh2pt = fx_calc_map_label(test_mesh_X, test_pt_X, test_label, k = 0, dist_method='euclidean')
    print(mesh2pt)


if __name__ == '__main__':
    topk_retrieval()
    # print('mAP ===== ', mAP)
    # print(all_cmc)
    # ks = [1, 5, 10, 20, 50]
    # for i in ks:
    #     print(i, ' = ', all_cmc[i-1])