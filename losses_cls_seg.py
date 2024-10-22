import numpy as np
import torch
from torch import nn
import math
import sys
def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)

    return neg_output, neg_labels

def _neg_loss(pos, neg):

    eps = torch.tensor(1.e-8).cuda()
    pos_pred, neg_pred = pos, neg

    #neg
    neg_pred = torch.clamp(neg_pred, min=eps, max=1-eps)
    neg_loss = (torch.log(1 - neg_pred) * torch.pow(neg_pred, 2)).mean()

    #pos
    if len(pos_pred)>0:
        
        pos_pred = torch.clamp(pos_pred, min=eps, max=1-eps)
        index = (pos_pred<0.9).nonzero().flatten()
        pos_pred_inds1 = torch.index_select(pos_pred, 0, index)
        if not len(index)==0: 
            pos_loss1 = (4 * torch.log(pos_pred_inds1) * torch.pow(1 - pos_pred_inds1, 2)).mean()
        else:
            pos_loss1 = torch.tensor(0)
            pos_pred_inds1 = []

        index2 = (pos_pred>=0.9).nonzero().flatten()
        if not len(index2)==0:
            pos_pred_inds2 = torch.index_select(pos_pred, 0, index2)
            pos_loss2 = (torch.log(pos_pred_inds2) * torch.pow(1 - pos_pred_inds2, 2)).mean()
        else:
            pos_loss2 = torch.tensor(0)
            pos_pred_inds2 = []
        
        pos_loss = pos_loss1 + pos_loss2
       
        loss = - (0.25 * pos_loss + 0.75 * neg_loss) 
    else:    
        loss = - neg_loss 

    return loss

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss
    def forward(self, pos, neg):
        return self.neg_loss(pos, neg)

class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = FocalLoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.offset_loss = nn.SmoothL1Loss() 
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.pos_num = int(config['pos_num']) 
    def forward(self, output, segout, labels, lunamask, flag=0, train = True):
        
        batch_size, D, H, W, C = labels.size() 

        if flag==0:
            seg_bceloss_data = 0.0
            seg_diceloss_data = 0.0
            for i in range(len(segout)):
                out = segout[i].squeeze(1)
                lunamask = lunamask.squeeze(1).float()
                seg_bceloss_data += self.bce_loss(out, lunamask)
                out = self.sigmoid(out)
                for j in range(batch_size):
                    if lunamask[j].max()==1:
                        seg_diceloss_data += dice_loss(out[j], lunamask[j])
            seg_loss_data = 0.5*seg_bceloss_data + 0.5*seg_diceloss_data
        else:
            seg_loss_data = 0.0

        output = output.view(-1, 5)
        labels = labels.view(-1, 5)

        pos_idcs = (labels[:, 0] == 1)
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5)
        pos_labels = labels[pos_idcs].view(-1, 5)

        neg_idcs = (labels[:, 0] == -1)
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs] 

        if train:
            if len(pos_labels)>0:
                num_hard = self.pos_num * 100
            else:
                num_hard = 100
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, num_hard * batch_size) #OHEM

        neg_prob = self.sigmoid(neg_output)


        if len(pos_output)>0:
            pos_prob = self.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            offset_loss_losses = [
                self.offset_loss(pz, lz),
                self.offset_loss(ph, lh),
                self.offset_loss(pw, lw)]
            offset_losses_data = [l for l in offset_loss_losses]

            regress_loss_data = self.regress_loss(pd, ld)
            classify_loss_data = self.classify_loss(pos_prob, neg_prob)
            pos_correct = (pos_prob.data >= 0.5).sum()
            pos_total = len(pos_prob)
            

        else:
            offset_losses_data = [0,0,0]
            classify_loss_data = self.classify_loss(pos_output, neg_prob)
            regress_loss_data = 0
            pos_correct = 0
            pos_total = 0

            
        offset_loss_data_total = 0
        for offset_loss_data in offset_losses_data:
            offset_loss_data_total += offset_loss_data

        loss = classify_loss_data + regress_loss_data + offset_loss_data_total + seg_loss_data

        neg_correct = (neg_prob.data < 0.5).sum()
        neg_total = len(neg_prob)

        return [loss, classify_loss_data, regress_loss_data] + offset_losses_data + [seg_loss_data, pos_correct, pos_total, neg_correct, neg_total]

class GetPBB(object): 
    def __init__(self, config):
        self.stride = config['stride'] 

    def __call__(self, output, thresh = -3, n = 100, ismask=False): 
        stride = self.stride
        output_size = output.shape 
        print(output_size) 
        zz,yy,xx = np.meshgrid(np.linspace(0,output_size[0]-1,output_size[0]),
                           np.linspace(0,output_size[1]-1,output_size[1]),
                           np.linspace(0,output_size[2]-1,output_size[2]),indexing ='ij')
        coord = np.concatenate([zz[np.newaxis,...], yy[np.newaxis,...], xx[np.newaxis,:]],0).astype('float32')
        output[:, :, :, 1] = coord[0] + output[:, :, :, 1] #z
        output[:, :, :, 2] = coord[1] + output[:, :, :, 2] #y
        output[:, :, :, 3] = coord[2] + output[:, :, :, 3] #x
        output[:, :, :, 4] = np.exp(output[:, :, :, 4]) #d
        output = torch.from_numpy(output).reshape(-1, 5)
        pos_idcs = (output[:, 0] > thresh) 
        pos_output = output[:, 0][pos_idcs] 
        pos_inds = torch.nonzero(output[:, 0] > thresh) 
        _, idcs = torch.topk(pos_output, min(n, len(pos_output))) 
        pos_output = np.asarray(output[pos_inds[idcs],:].view(-1, 5))
        # if len(pos_output)>0:
        #     pos_prob = self.sigmoid(pos_output[:, 0])
        #     pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
        return pos_output
        
def nms(output, nms_th):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def iou(box0, box1):
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union


