from __future__ import division, absolute_import
from models.dgcnn import DGCNN
from models.pointnet_part_seg import PointnetPartSeg
# from models.pointnet_part_seg import PointNet_Part
from models.meshnet import MeshNet
from models.SVCNN import Semi3D, SingleViewNet
from tools.triplet_dataloader import TripletDataloader
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

    model = Semi3D(img_net, pt_net, meshnet)
    
    model = model.to('cuda')

    model = torch.nn.DataParallel(model)

    model.train(True)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    writer = SummaryWriter(os.path.join(args.save, 'summary'))

    #data splittted into unlabeled/labeled/test
    labeled_set = TripletDataloader(dataset = 'ModelNet40', num_points = args.num_points, partition='labeled',  perceptange = 2)
    labeled_data_loader = torch.utils.data.DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True,num_workers=8, drop_last=True)


    # unlabeled_set = TripletDataloader(dataset = 'ModelNet40', num_points = args.num_points, partition='unlabeled',  perceptange = 10)
    # unlabeled_data_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    print('************************************************************')
    print('         check the following important parametes            ')
    print('the number of labeled sample: ', len(labeled_set))
    # print('the number of unlabeled sample: ', len(unlabeled_set))
    print('the temperature for the probability: ', args.T)
    print('the threshold for the probability: ', args.threshold)
    print('************************************************************')

    # The loss introduced in Hinton's paper
    # nt_xent_criterion = NTXentLoss('cuda', args.batch_size, temperature = 0.5, use_cosine_similarity = True)
    # mse_criterion = nn.MSELoss() 
    ce_criterion = nn.CrossEntropyLoss(reduction='mean')

    iteration = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        for l_data in labeled_data_loader:
            pt1, img1, img1V, centers1, corners1, normals1, neighbor_index1, target1 = l_data #the last one is the target
            # pt2, img2, img2V, centers2, corners2, normals2, neighbor_index2, target2 = u_data #the last one is the target

            pt1 = Variable(pt1).to('cuda')
            pt1 = pt1.permute(0,2,1)

            # pt2 = Variable(pt2).to('cuda')
            # pt2 = pt2.permute(0,2,1)

            img1 = Variable(img1).to('cuda')
            img1V = Variable(img1V).to('cuda')
            # img2 = Variable(img2).to('cuda')
            # img2V = Variable(img2V).to('cuda')

            target1 = torch.squeeze(target1)
            target1 = Variable(target1).to('cuda')

            # target2 = torch.squeeze(target2)
            # target2 = Variable(target2).to('cuda')

            centers1 = Variable(torch.cuda.FloatTensor(centers1.cuda()))
            corners1 = Variable(torch.cuda.FloatTensor(corners1.cuda()))
            normals1 = Variable(torch.cuda.FloatTensor(normals1.cuda()))
            neighbor_index1 = Variable(torch.cuda.LongTensor(neighbor_index1.cuda()))

            # centers2 = Variable(torch.cuda.FloatTensor(centers2.cuda()))
            # corners2 = Variable(torch.cuda.FloatTensor(corners2.cuda()))
            # normals2 = Variable(torch.cuda.FloatTensor(normals2.cuda()))
            # neighbor_index2 = Variable(torch.cuda.LongTensor(neighbor_index2.cuda()))

            optimizer.zero_grad()

            pt_pred1, mesh_pred1, img_pred1, fused_pred1, pt_feat1, mesh_feat1, img_feat1 = model(pt1, img1, img1V, centers1, corners1, normals1, neighbor_index1)

            # pt_pred2, mesh_pred2, img_pred2, fused_pred2, pt_feat2, mesh_feat2, img_feat2 = model(pt2, img2, img2V, centers2, corners2, normals2, neighbor_index2)

            #cross-entropy loss on the labeled data
            pt_ce_loss = ce_criterion(pt_pred1, target1)
            mesh_ce_loss = ce_criterion(mesh_pred1, target1)
            img_ce_loss = ce_criterion(img_pred1, target1)
            # print(fused_pred1.size(), target1.size())
            fused_ce_loss = ce_criterion(fused_pred1, target1)
            entropy_loss = pt_ce_loss + mesh_ce_loss + img_ce_loss + fused_ce_loss


            Eweight = 1.0
            loss = Eweight * entropy_loss

            loss.backward()

            #update the parameters for the center_loss
            optimizer.step()

            img_acc1 = calculate_accuracy(img_pred1, target1)
            pt_acc1 = calculate_accuracy(pt_pred1, target1)
            mesh_acc1 = calculate_accuracy(mesh_pred1, target1)
            fused_acc1 = calculate_accuracy(fused_pred1, target1)

            #classification accuracy on the labeld sample
            writer.add_scalar('Labeled_Acc/img_acc', img_acc1, iteration)
            writer.add_scalar('Labeled_Acc/pt_acc', pt_acc1, iteration)
            writer.add_scalar('Labeled_Acc/mesh_acc', mesh_acc1, iteration)
            writer.add_scalar('Labeled_Acc/fused_acc', fused_acc1, iteration)

            #Xentropy loss on the labeled data
            writer.add_scalar('Xentropy_loss/pt_ce_loss', pt_ce_loss.item(), iteration)
            writer.add_scalar('Xentropy_loss/mesh_ce_loss', mesh_ce_loss.item(), iteration)
            writer.add_scalar('Xentropy_loss/img_ce_loss', img_ce_loss.item(), iteration)
            writer.add_scalar('Xentropy_loss/fused_ce_loss', fused_ce_loss.item(), iteration)

            writer.add_scalar('loss/Xentropy_loss', entropy_loss.item(), iteration)

            writer.add_scalar('loss/loss', loss.item(), iteration)


            if (iteration%args.lr_step) == 0:
                lr = args.lr * (0.1 ** (iteration // args.lr_step))
                print('New  LR:     ' + str(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if iteration % args.per_print == 0:
                print('[%d][%d]  loss: %.2f Xentropy_loss %.2f time: %.2f  vid: %d' % \
                    (epoch, iteration, loss.item(), entropy_loss.item(), time.time() - start_time, 2 * pt1.size(0)))
                start_time = time.time()

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

                mesh_net_name = args.save + str(iteration+1)+'-semi3d_model.pkl'
                torch.save({'state_dict': model.state_dict()}, mesh_net_name)

            iteration = iteration + 1
            if iteration > args.max_step:
                return


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Learning View and Model invariant features for 3D shapes')

    parser.add_argument('--batch_size', type=int, default=48, metavar='batch_size',
                        help='Size of batch)')

    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of episode to train ')
    #optimizer
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')

    parser.add_argument('--lr_step', type=int,  default = 4000,
                        help='how many iterations to decrease the learning rate')

    parser.add_argument('--max_step', type=int,  default = 12100,
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

    parser.add_argument('--per_save', type=int,  default=2500,
                        help='how many iterations to save the model')

    parser.add_argument('--per_print', type=int,  default=100,
                        help='how many iterations to print the loss and accuracy')
    parser.add_argument('--save', type=str,  default='./checkpoints/ModelNet40-pt1024-mesh-img56-Xentropy-2percent-supervised/',
                        help='path to save the final model')

    parser.add_argument('--gpu_id', type=str,  default='0,1,2',
                        help='GPU used to train the network')

    parser.add_argument('--log', type=str,  default='log/',
                        help='path to the log information')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = False
    training(args)
