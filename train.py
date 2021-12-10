from __future__ import division, absolute_import
from models.dgcnn import DGCNN
from models.pointnet_part_seg import PointnetPartSeg
# from models.pointnet_part_seg import PointNet_Part
from models.meshnet import MeshNet
from models.SVCNN_twfview import Semi3D, SingleViewNet, FusionNet, FusionHead
from tools.triplet_dataloader_twfview import TripletDataloader
from tools.utils import calculate_accuracy
import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse
import torch.optim as optim
import time
from centerloss import CenterLoss
from nt_xent import NTXentLoss

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)



def training(args):
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    img_net = SingleViewNet(pre_trained = True)
    pt_net = DGCNN(args)
    # pt_net = PointnetPartSeg()
    meshnet = MeshNet()

    fusionnet = FusionNet()
    fusionhead = FusionHead()
#     iterations = 10000

#     weights_folder = 'ModelNet40-pt1024-mesh-img56-Xentropy-Xcontrast-PointMultiAgreement-T095-Fused-Warmup-2percent_task4_p2_w2'

#     img_net_name = './checkpoints/%s/%d-img_net.pkl' % (weights_folder, iterations)
#     img_net.load_state_dict(torch.load(img_net_name)['state_dict'],  strict=False)

#     dgcnn_name = './checkpoints/%s/%d-pt_net.pkl' % (weights_folder, iterations)
#     pt_net.load_state_dict(torch.load(dgcnn_name)['state_dict'],  strict=False)

#     mesh_net_name = './checkpoints/%s/%d-mesh_net.pkl' % (weights_folder, iterations)
#     meshnet.load_state_dict(torch.load(mesh_net_name)['state_dict'], strict=False)

#     fusionnet_name = './checkpoints/%s/%d-fusion_net.pkl' % (weights_folder, iterations)
#     fusionnet.load_state_dict(torch.load(fusionnet_name)['state_dict'], strict=False)



    model = Semi3D(img_net, pt_net, meshnet, fusionnet, fusionhead)
    
    model = model.to('cuda')

    model = torch.nn.DataParallel(model)

    model.train(True)




#     center_criterion = CenterLoss(num_classes = args.num_classes, feat_dim= 512, use_gpu=True)

    center_criterion = CenterLoss(num_classes = args.num_classes, feat_dim= 512, temperature=0.5, use_gpu=True)
    
    
#     center_criterion = CenterLoss(num_classes = args.num_classes, feat_dim= 256, temperature=0.5, use_gpu=True)
#     center_criterion = CenterLoss(num_classes = args.num_classes, feat_dim= 256, temperature=0.5, use_gpu=True)


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    optimizer_centloss = optim.SGD(center_criterion.parameters(), lr=args.lr_cent)


    writer = SummaryWriter(os.path.join(args.save, 'summary'))

    #data splittted into unlabeled/labeled/test
    labeled_set = TripletDataloader(dataset = 'ModelNet40', num_points = args.num_points, partition='labeled',  perceptange = 10)
    labeled_data_loader = torch.utils.data.DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True,num_workers=8, drop_last=True)


    unlabeled_set = TripletDataloader(dataset = 'ModelNet40', num_points = args.num_points, partition='unlabeled',  perceptange = 10)
    unlabeled_data_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    print('************************************************************')
    print('         check the following important parametes            ')
    print('the number of labeled sample: ', len(labeled_set))
    print('the number of unlabeled sample: ', len(unlabeled_set))
    print('the temperature for the probability: ', args.T)
    print('the threshold for the probability: ', args.threshold)
    print('************************************************************')

    # The loss introduced in Hinton's paper
    nt_xent_criterion = NTXentLoss('cuda', args.batch_size, temperature = 0.5, use_cosine_similarity = True)
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss(reduction='mean')

    iteration = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        for l_data, u_data in zip (labeled_data_loader, unlabeled_data_loader):
            pt1, img1, img1V, img1V3, img1V4, img1V5, img1V6, img1V7, img1V8, img1V9, img1V10, img1V11,img1V12, centers1, corners1, normals1, neighbor_index1, target1 = l_data #the last one is the target
            pt2, img2, img2V,img2V3, img2V4, img2V5, img2V6, img2V7, img2V8, img2V9, img2V10, img2V11,img2V12, centers2, corners2, normals2, neighbor_index2, target2 = u_data #the last one is the target

            pt1 = Variable(pt1).to('cuda')
            pt1 = pt1.permute(0,2,1)

            pt2 = Variable(pt2).to('cuda')
            pt2 = pt2.permute(0,2,1)

            img1 = Variable(img1).to('cuda')
            img1V = Variable(img1V).to('cuda')
            
            img2 = Variable(img2).to('cuda')
            img2V = Variable(img2V).to('cuda')
            
            img1V3 = Variable(img1V3).to('cuda')
            img2V3 = Variable(img2V3).to('cuda')

            img1V4 = Variable(img1V4).to('cuda')
            img2V4 = Variable(img2V4).to('cuda')
            
            img1V5 = Variable(img1V5).to('cuda')
            img2V5 = Variable(img2V5).to('cuda')           
            
            img1V6 = Variable(img1V6).to('cuda')
            img2V6 = Variable(img2V6).to('cuda')
            
            img1V7 = Variable(img1V7).to('cuda')
            img2V7 = Variable(img2V7).to('cuda')           
            
            img1V8 = Variable(img1V8).to('cuda')
            img2V8 = Variable(img2V8).to('cuda')          
            
            img1V9 = Variable(img1V9).to('cuda')
            img2V9 = Variable(img2V9).to('cuda')            
            
            img1V10 = Variable(img1V10).to('cuda')
            img2V10 = Variable(img2V10).to('cuda')
            
            img1V11 = Variable(img1V11).to('cuda')
            img2V11 = Variable(img2V11).to('cuda')            
            
            img1V12 = Variable(img1V12).to('cuda')
            img2V12 = Variable(img2V12).to('cuda')            
                      
            
            target1 = torch.squeeze(target1)
            target1 = Variable(target1).to('cuda')

            target2 = torch.squeeze(target2)
            target2 = Variable(target2).to('cuda')

            centers1 = Variable(torch.cuda.FloatTensor(centers1.cuda()))
            corners1 = Variable(torch.cuda.FloatTensor(corners1.cuda()))
            normals1 = Variable(torch.cuda.FloatTensor(normals1.cuda()))
            neighbor_index1 = Variable(torch.cuda.LongTensor(neighbor_index1.cuda()))

            centers2 = Variable(torch.cuda.FloatTensor(centers2.cuda()))
            corners2 = Variable(torch.cuda.FloatTensor(corners2.cuda()))
            normals2 = Variable(torch.cuda.FloatTensor(normals2.cuda()))
            neighbor_index2 = Variable(torch.cuda.LongTensor(neighbor_index2.cuda()))

            optimizer.zero_grad()

            pt_pred1, mesh_pred1, img_pred1, fused_pred1, pt_feat1, mesh_feat1, img_feat1, pt_base1, mesh_base1, img_base1, pt_gfeat1, mesh_gfeat1, img_gfeat1 = model(pt1, img1, img1V, img1V3, img1V4, img1V5, img1V6, img1V7, img1V8, img1V9, img1V10, img1V11,img1V12,centers1, corners1, normals1, neighbor_index1)

            pt_pred2, mesh_pred2, img_pred2, fused_pred2, pt_feat2, mesh_feat2, img_feat2, pt_base2, mesh_base2, img_base2, pt_gfeat2, mesh_gfeat2, img_gfeat2 = model(pt2, img2, img2V, img2V3, img2V4, img2V5, img2V6, img2V7, img2V8, img2V9, img2V10, img2V11,img2V12, centers2, corners2, normals2, neighbor_index2)

            #cross-entropy loss on the labeled data
            pt_ce_loss = ce_criterion(pt_pred1, target1)
            mesh_ce_loss = ce_criterion(mesh_pred1, target1)
            img_ce_loss = ce_criterion(img_pred1, target1)
            fused_ce_loss = ce_criterion(fused_pred1, target1)
            entropy_loss = pt_ce_loss + mesh_ce_loss + img_ce_loss + fused_ce_loss

            # cross-modal contrastive loss on the unlabeld data
            pt_img_contrast_loss =  nt_xent_criterion(pt_feat2, img_feat2)
            mesh_img_contrast_loss = nt_xent_criterion(mesh_feat2, img_feat2)
            pt_mesh_contrast_loss = nt_xent_criterion(pt_feat2, mesh_feat2)
            Xcontrastive_loss = pt_img_contrast_loss + mesh_img_contrast_loss + pt_mesh_contrast_loss

            #agreement loss on the unlabele data

            pseudo_label = torch.softmax(fused_pred2.detach(), dim=-1)

            max_probs, targets_u = torch.max(pseudo_label, dim=-1)

            mask = max_probs.ge(0.95).float()

            valid_sample_num = torch.sum(mask)
            # print(valid_sample_num.item())
            mask_label = torch.ones(mask.shape[0])
            mask_label = Variable(mask_label).to('cuda')

            pt_pseudo_ce_loss = (F.cross_entropy(pt_pred2, targets_u, reduction='none') * mask).mean()
            mesh_pseudo_ce_loss = (F.cross_entropy(mesh_pred2, targets_u, reduction='none') * mask).mean()
            img_pseudo_ce_loss = (F.cross_entropy(img_pred2, targets_u, reduction='none') * mask).mean()
            fused_pseudo_ce_loss = (F.cross_entropy(fused_pred2, targets_u, reduction='none') * mask).mean()
            agreement_loss = pt_pseudo_ce_loss + mesh_pseudo_ce_loss + img_pseudo_ce_loss + fused_pseudo_ce_loss

            cate_base_feature = torch.cat([pt_base1, mesh_base1, img_base1, pt_base2, mesh_base2, img_base2], dim=0)
#             cate_base_feature = torch.cat([pt_gfeat1, mesh_gfeat1, img_gfeat1, pt_gfeat2, mesh_gfeat2, img_gfeat2], dim=0)
            cate_target = torch.cat([target1, target1, target1, targets_u, targets_u, targets_u], dim=0)
            cate_mask = torch.cat([mask_label, mask_label, mask_label, mask, mask, mask], dim=0)
            
#             cate_base_feature = torch.cat([pt_gfeat1, mesh_gfeat1, img_gfeat1], dim=0)
#             cate_base_feature = torch.cat([pt_base1, mesh_base1, img_base1], dim=0)
#             cate_target = torch.cat([target1, target1, target1], dim=0)
#             cate_mask = torch.cat([mask_label, mask_label, mask_label], dim=0)

            # cate_base_feature = F.relu(cate_base_feature)

            Xcenter_loss, centers = center_criterion(cate_base_feature, cate_target, cate_mask)

            center_loss = Xcenter_loss
            
            pt_img_mse_loss =  mse_criterion(pt_feat2, img_feat2)
            mesh_img_mse_loss = mse_criterion(mesh_feat2, img_feat2)
            pt_mesh_mse_loss = mse_criterion(pt_feat2, mesh_feat2)
            mse_loss = pt_img_mse_loss + mesh_img_mse_loss + pt_mesh_mse_loss
            

            
            #Without MSE LOSS
            Eweight = 1.0
            Xweight = 2.0
#             Aweight = 2.0 * math.exp(-5 * (1 - min(iteration/6000.0, 1))**2)
            Aweight = 0.0 * math.exp(-5 * (1 - min(iteration/60000.0, 1))**2)

            # Aweight = 0.0
            Mweight = 0.0
            Cweight = 9.0 * math.exp(-5 * (1 - min(iteration/60000.0, 1))**2)
        
            loss = Eweight * entropy_loss + Xweight * Xcontrastive_loss + Aweight * agreement_loss +  Mweight * mse_loss + Cweight * center_loss

            loss.backward()

            #update the parameters for the center_loss
            optimizer.step()

            img_acc1 = calculate_accuracy(img_pred1, target1)
            pt_acc1 = calculate_accuracy(pt_pred1, target1)
            mesh_acc1 = calculate_accuracy(mesh_pred1, target1)
            fused_acc1 = calculate_accuracy(fused_pred1, target1)

            img_acc2 = calculate_accuracy(img_pred2, target2)
            pt_acc2 = calculate_accuracy(pt_pred2, target2)
            mesh_acc2 = calculate_accuracy(mesh_pred2, target2)
            fused_acc2 = calculate_accuracy(fused_pred2, target2)

            #classification accuracy on the labeld sample
            writer.add_scalar('Labeled_Acc/img_acc', img_acc1, iteration)
            writer.add_scalar('Labeled_Acc/pt_acc', pt_acc1, iteration)
            writer.add_scalar('Labeled_Acc/mesh_acc', mesh_acc1, iteration)
            writer.add_scalar('Labeled_Acc/fused_acc', fused_acc1, iteration)

            #classification accuracy on the unlabeld sample
            writer.add_scalar('Unlabele_Acc/img_acc' ,img_acc2, iteration)
            writer.add_scalar('Unlabele_Acc/pt_acc', pt_acc2, iteration)
            writer.add_scalar('Unlabele_Acc/mesh_acc', mesh_acc2, iteration)
            writer.add_scalar('Unlabele_Acc/fused_acc', fused_acc2, iteration)
            writer.add_scalar('Unlabele_Acc/valid_sample_num', valid_sample_num.item(), iteration)


            #Xentropy loss on the labeled data
            writer.add_scalar('Xentropy_loss/pt_ce_loss', pt_ce_loss.item(), iteration)
            writer.add_scalar('Xentropy_loss/mesh_ce_loss', mesh_ce_loss.item(), iteration)
            writer.add_scalar('Xentropy_loss/img_ce_loss', img_ce_loss.item(), iteration)
            writer.add_scalar('Xentropy_loss/fused_ce_loss', fused_ce_loss.item(), iteration)

            #agreement loss on the unlabeled data
            writer.add_scalar('Agreement_loss/pt_pseudo_ce_loss', pt_pseudo_ce_loss.item(), iteration)
            writer.add_scalar('Agreement_loss/mesh_pseudo_ce_loss', mesh_pseudo_ce_loss.item(), iteration)
            writer.add_scalar('Agreement_loss/img_pseudo_ce_loss', img_pseudo_ce_loss.item(), iteration)
            writer.add_scalar('Agreement_loss/fused_pseudo_ce_loss', fused_pseudo_ce_loss.item(), iteration)

            #tensorboard visualization
            writer.add_scalar('Xcontrast_loss/cloud_img_contrast_loss', pt_img_contrast_loss.item(), iteration)
            writer.add_scalar('Xcontrast_loss/mesh_img_contrast_loss', mesh_img_contrast_loss.item(), iteration)
            writer.add_scalar('Xcontrast_loss/cloud_mesh_contrast_loss', pt_mesh_contrast_loss.item(), iteration)

            writer.add_scalar('Center_loss/Xcenter_loss', Xcenter_loss.item(), iteration)

            writer.add_scalar('Xmse_loss/cloud_img_mse_loss', pt_img_mse_loss.item(), iteration)
            writer.add_scalar('Xmse_loss/mesh_img_mse_loss', mesh_img_mse_loss.item(), iteration)
            writer.add_scalar('Xmse_loss/cloud_mesh_mse_loss', pt_mesh_mse_loss.item(), iteration)

            writer.add_scalar('loss/Xentropy_loss', entropy_loss.item(), iteration)
            writer.add_scalar('loss/Agreement_loss', agreement_loss.item(), iteration)
            writer.add_scalar('loss/Xcontrastive_loss', Xcontrastive_loss.item(), iteration)
            writer.add_scalar('loss/Center_loss', center_loss.item(), iteration)

            writer.add_scalar('loss/mse_loss', mse_loss.item(), iteration)
            writer.add_scalar('loss/loss', loss.item(), iteration)

            writer.add_scalar('HyperParameters/Aweight', Aweight, iteration)

            if (iteration%args.lr_step) == 0:
                lr = args.lr * (0.1 ** (iteration // args.lr_step))
                print('New  LR:     ' + str(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    
                lr = args.lr_cent * (0.1 ** (iteration // args.lr_step))
                print('New  CenteLR:     ' + str(lr))
                for param_group in optimizer_centloss.param_groups:
                    param_group['lr'] = lr
                       
            if iteration % args.per_print == 0:
                print('[%d][%d]  loss: %.2f Xentropy_loss %.2f Xcontrastive_loss %.2f mse_loss %.2f agreement_loss %.2f Xcenter_loss %.2f time: %.2f  vid: %d valid_sample_num: %d'  % \
                    (epoch, iteration, loss.item(), entropy_loss.item(), Xcontrastive_loss.item(), mse_loss.item(), agreement_loss.item(), Xcenter_loss.item() ,time.time() - start_time, 2 * pt1.size(0),valid_sample_num))
                start_time = time.time()
                # print(max_probs)

            iteration = iteration + 1
            if((iteration+1) % args.per_save) ==0:
                print('----------------- Save The Network ------------------------')
                # with open(args.save + str(iteration+1)+'-head_net.pkl', 'wb') as f:
                #     torch.save(model, f)
                img_net_name = args.save + str(iteration+1)+'-img_net.pkl'
                torch.save({'state_dict': img_net.state_dict()}, img_net_name)

                pt_net_name = args.save + str(iteration+1)+'-pt_net.pkl'
                torch.save({'state_dict': pt_net.state_dict()}, pt_net_name)

                mesh_net_name = args.save + str(iteration+1)+'-mesh_net.pkl'
                torch.save({'state_dict': meshnet.state_dict()}, mesh_net_name)

                fusion_net_name = args.save + str(iteration+1)+'-fusion_net.pkl'
                torch.save({'state_dict': fusionnet.state_dict()}, fusion_net_name)
                
                fusion_head_name = args.save + str(iteration+1)+'-fusion_head.pkl'
                torch.save({'state_dict': fusionhead.state_dict()}, fusion_head_name)

                mesh_net_name = args.save + str(iteration+1)+'-semi3d_model.pkl'
                torch.save({'state_dict': model.state_dict()}, mesh_net_name)
                
                center_name = args.save + str(iteration+1) + 'center'
                np.save(center_name, centers.detach().cpu().numpy())

            iteration = iteration + 1
            if iteration > args.max_step:
                return


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Learning View and Model invariant features for 3D shapes')
    
    parser.add_argument('--num_classes', type=int, default=40, metavar='num_classes',
                        help='Num of Classes)')
    parser.add_argument('--batch_size', type=int, default=48, metavar='batch_size',
                        help='Size of batch)')

    parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                        help='number of episode to train ')
    #optimizer
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--lr_cent', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--lr_step', type=int,  default = 40000,
                        help='how many iterations to decrease the learning rate')

    parser.add_argument('--max_step', type=int,  default = 101000,
                        help='maximum steps to train the network')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--T', type=int,  default = 1,
                        help='temperature for the prediction')

    parser.add_argument('--threshold', type=int,  default = 0.95,
                        help='threshold for the positive samples')

    #image for SVCNN
    parser.add_argument('--num_views', type=int, default=180, metavar='S',
                        help='number of views for training (default: 6)')
    #DGCNN
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

    parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='weight_decay',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--per_save', type=int,  default= 5000,
                        help='how many iterations to save the model')

    parser.add_argument('--per_print', type=int,  default=100,
                        help='how many iterations to print the loss and accuracy')
    parser.add_argument('--save', type=str,  default='./checkpoints/ModelNet40_p10_nt_xw2_aw0_cw9_baseshare_Cos_12views_imgsize224_10w_resnet50/',
                        help='path to save the final model')

    parser.add_argument('--gpu_id', type=str,  default='0,1,2',
                        help='GPU used to train the network')

    parser.add_argument('--log', type=str,  default='log/',
                        help='path to the log information')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = False
    training(args)
