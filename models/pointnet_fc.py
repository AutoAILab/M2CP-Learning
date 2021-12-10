
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.pointnet import STN3d, STNkd, feature_transform_reguliarzer

class PartSeg(nn.Module):
    def __init__(self, part_num=50, normal_channel=False):
        super(PartSeg, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)

    def forward(self, point_cloud):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        # out_max = out_max.view(-1, 2048)

        # # # out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048, 1).repeat(1, 1, N)

        return torch.cat([expand, out1, out2, out3, out4, out5], 1), out_max


class get_model(nn.Module):

    def __init__(self, part_num=50, num_classes=16):
        super(get_model, self).__init__()
        self.part_num = part_num
        self.num_classes = num_classes
        self.convs_label = torch.nn.Conv1d(self.num_classes, 64, 1)
        self.convs1 = torch.nn.Conv1d(4928+64, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, self.part_num, 1)
        self.bns_label = nn.BatchNorm1d(64)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, pt_feat, pt_cat, label):
        B, D, N = pt_cat.size()
        # print('size of cat ====', pt_cat.size())
        # print('size of feat ====', pt_feat.size())
        # print('size of label ====', label.size())
        label = label.view(B, self.num_classes, 1)
        # print('label ====', label.size())
        label = F.relu(self.bns_label(self.convs_label(label)))
        # print('label ====', label.size())

        # feat_label = torch.cat([pt_feat,label.squeeze()],1)
        # print('feat_label ====', feat_label.size())
        # feat_label = feat_label.view(-1, 512+16, 1).repeat(1, 1, N)

        label = label.view(-1, 64, 1).repeat(1, 1, N)
        # print('feat_label ====', feat_label.size())
        concat = torch.cat([pt_cat, label], 1)
        # print('concat ====', concat.size())

        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num) # [B, N, 50]

        return net


class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        # self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        # mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        # total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return loss
