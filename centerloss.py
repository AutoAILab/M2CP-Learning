import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes = 40, feat_dim = 256,temperature = 0.07, use_gpu = True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.temperature=temperature
        self.base_temperature=0.07
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, valid_mask):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        
        x = F.normalize(x, dim=1)
        centers = F.normalize(self.centers, dim=1)
        
        anchor_dot_contrast = torch.div(
            torch.matmul(x, centers.T),
            self.temperature)


#         x = F.normalize(x, dim=1)
#         centers = F.normalize(self.centers, dim=1)

#         anchor_dot_contrast = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                   torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#         anchor_dot_contrast.addmm_(1, -2, x, centers.t())
        
#         anchor_dot_contrast = torch.div(anchor_dot_contrast, self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        mask = mask.float()
        valid_mask = valid_mask.float()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - (0.5/ self.base_temperature) * mean_log_prob_pos

        loss = loss * valid_mask
        loss = loss.mean()

        return loss, centers
 



# import torch
# import torch.nn as nn

# class CenterLoss(nn.Module):
#     """Center loss.
#
#     Reference:
#     Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
#
#     Args:
#         num_classes (int): number of classes.
#         feat_dim (int): feature dimension.
#     """
#     def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.use_gpu = use_gpu
#
#         if self.use_gpu:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
#         else:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
#
#     def forward(self, x, labels):
#         """
#         Args:
#             x: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (batch_size).
#         """
#         batch_size = x.size(0)
#         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#         distmat.addmm_(1, -2, x, self.centers.t())
#
#         classes = torch.arange(self.num_classes).long()
#         if self.use_gpu: classes = classes.cuda()
#         labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
#         mask = labels.eq(classes.expand(batch_size, self.num_classes))
#
#         dist = distmat * mask.float()
#         loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
#
#         return loss