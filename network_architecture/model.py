import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from network_architecture.DeTrans.DeformableTrans import DeformableTransformer
from network_architecture.DeTrans.position_encoding import build_position_encoding
from losses_cls_seg import nms

def Conv_Block(in_planes, out_planes, stride=1):
    """3x3x3 convolution with batchnorm and relu"""
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False),
                         nn.BatchNorm3d(out_planes),
                         nn.ReLU(inplace=True))


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Norm(nn.Module):
    def __init__(self, N):
        super(Norm, self).__init__()
        self.normal = nn.BatchNorm3d(N)

    def forward(self, x):
        return self.normal(x)

class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = Norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = Norm(planes)
        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                Norm(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = Norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = Norm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm(planes.cuda(non_blocking=True))
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = Norm(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = Norm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SAC(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(SAC, self).__init__()

        self.conv_1 = nn.Conv3d(
            input_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv3d(
            input_channel, out_channel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_5 = nn.Conv3d(
            input_channel, out_channel, kernel_size=3, stride=1, padding=5, dilation=5)
        self.weights = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(0)

    def forward(self, inputs):
        feat_1 = self.conv_1(inputs)
        feat_3 = self.conv_3(inputs)
        feat_5 = self.conv_5(inputs)
        weights = self.softmax(self.weights)
        feat = feat_1 * weights[0] + feat_3 * weights[1] + feat_5 * weights[2]
        return feat


class Pyramid_3D(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256, using_sac=False):
        super(Pyramid_3D, self).__init__()

        self.P5_1 = nn.Conv3d(C5_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='trilinear')
        self.P5_2 = nn.Conv3d(feature_size, 64, kernel_size=3, stride=1,
                              padding=1) if not using_sac else SAC(feature_size, 128)

        self.P4_1 = nn.Conv3d(C4_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='trilinear')
        self.P4_2 = nn.Conv3d(feature_size, 64, kernel_size=3, stride=1,
                              padding=1) if not using_sac else SAC(feature_size, 128)

        self.P3_1 = nn.Conv3d(C3_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='trilinear')
        self.P3_2 = nn.Conv3d(feature_size, 64, kernel_size=3, stride=1,
                              padding=1) if not using_sac else SAC(feature_size, 64)

        self.P2_1 = nn.Conv3d(C2_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv3d(feature_size, 32, kernel_size=3, stride=1,
                              padding=1) if not using_sac else SAC(feature_size, 32)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        return [P2_x, P3_x, P4_x, P5_x]


class Attention_SE_CA(nn.Module):
    def __init__(self, channel):
        super(Attention_SE_CA, self).__init__()
        self.Global_Pool = nn.AdaptiveAvgPool3d(1)
        self.FC1 = nn.Sequential(nn.Linear(channel, channel),
                                 nn.ReLU(), )
        self.FC2 = nn.Sequential(nn.Linear(channel, channel),
                                 nn.Sigmoid(), )

    def forward(self, x):
        G = self.Global_Pool(x)
        G = G.view(G.size(0), -1)
        fc1 = self.FC1(G)
        fc2 = self.FC2(fc1)
        fc2 = torch.unsqueeze(fc2, 2)
        fc2 = torch.unsqueeze(fc2, 3)
        fc2 = torch.unsqueeze(fc2, 4)
        return fc2*x


class Net(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 32
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = Norm(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = Norm(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.atttion1 = Attention_SE_CA(32)
        self.atttion2 = Attention_SE_CA(32)
        self.atttion3 = Attention_SE_CA(64)
        self.atttion4 = Attention_SE_CA(64)
        
        self.conv_8x = ResBlock(128, 64)
        self.conv_4x = ResBlock(64, 32)
        self.conv_2x = ResBlock(32, 32)

        self.convc = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.convr = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.convo = nn.Conv3d(32, 3, kernel_size=1, stride=1)

        self.convm11 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        
        self.position_embed = build_position_encoding(mode='v2', hidden_dim=128)
        self.encoder_Detrans = DeformableTransformer(d_model=128, dim_feedforward=512, dropout=0.1, activation='gelu', num_feature_levels=2, nhead=8, num_encoder_layers=6, enc_n_points=4) 
        total = sum([param.nelement() for param in self.encoder_Detrans.parameters()])
        print('  + Number of Transformer Params: %.2f(e6)' % (total / 1e6))
        
        self.fpn = Pyramid_3D(32, 64, 64, 64, feature_size=64, using_sac=True) 
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def posi_mask(self, x):
    
        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            if lvl > 1:
                x_fea.append(fea) 
                x_posemb.append(self.position_embed(fea))
                masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda()) #B H W D

        return x_fea, masks, x_posemb

    def crop_and_pad(self, features, bbox):

        result = torch.empty_like(features)
        bbox_z = torch.zeros(features.size(0))
        img_scale = torch.zeros(features.size(0))
        for i in range(features.size(0)):
            image = features[i]
            z = bbox[i][0]
            scale = 1
            target_cropsize = features.size(-1)
            start = []
            pad = [0,0,0,0,0,0]
            if not torch.isnan(bbox[i][0]):
                target_cropsize = int(bbox[i][3]+6)
                k = features.size(-1)//target_cropsize
                while features.size(-1)%k != 0:
                    k -= 1
                scale = int(k)
                target_cropsize = int(features.size(-1)//scale)
                start = []
                for j in range(3):
                    start.append(int(bbox[i][j]-target_cropsize/2))
                pad = []
                for j in range(1,4):
                    leftpad = max(0,-start[-j])
                    rightpad = max(0,start[-j]+target_cropsize-image.shape[1])
                    pad.append(leftpad)
                    pad.append(rightpad)
                image = image[:,
                    max(start[0],0):min(start[0] + target_cropsize,image.shape[1]),
                    max(start[1],0):min(start[1] + target_cropsize,image.shape[2]),
                    max(start[2],0):min(start[2] + target_cropsize,image.shape[3])]
                
                pad = tuple(pad)
                image = F.pad(image, pad,'constant',value=0) 
                z = bbox[i][0] - start[0]
                image = F.interpolate(image.unsqueeze(0), scale_factor=scale, mode='trilinear').squeeze(0)
            
            result[i] = image
            bbox_z[i] = z
            img_scale[i] = scale
            
        return result, [bbox_z, img_scale]
    

    def forward(self, x, bbox, test_image=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.atttion1(x)
        x1 = self.layer1(x)
        x1 = self.atttion2(x1)
        x2 = self.layer2(x1)
        x2 = self.atttion3(x2)
        x3 = self.layer3(x2)
        x3 = self.atttion4(x3)
        x4 = self.layer4(x3)
        feats = self.fpn([x1, x2, x3, x4])
        x_fea, masks, x_posemb = self.posi_mask(feats) 
        x_trans = self.encoder_Detrans(x_fea, masks, x_posemb)

        # trans0 = x_trans[:, x_fea[2].shape[-3]*x_fea[2].shape[-2]*x_fea[2].shape[-1]::].transpose(-1, -2).view(feats[-1].shape)
        # trans1 = x_trans[:, x_fea[1].shape[-3]*x_fea[1].shape[-2]*x_fea[1].shape[-1]:x_fea[2].shape[-3]*x_fea[2].shape[-2]*x_fea[2].shape[-1]].transpose(-1, -2).view(feats[-2].shape) 
        # trans2 = x_trans[:, x_fea[0].shape[-3]*x_fea[0].shape[-2]*x_fea[0].shape[-1]:x_fea[1].shape[-3]*x_fea[1].shape[-2]*x_fea[1].shape[-1]].transpose(-1, -2).view(feats[-3].shape) 
        # trans3 = x_trans[:, 0:x_fea[0].shape[-3]*x_fea[0].shape[-2]*x_fea[0].shape[-1]].transpose(-1, -2).view(feats[-4].shape) 
        trans0 = x_trans[:, x_fea[0].shape[-3]*x_fea[0].shape[-2]*x_fea[0].shape[-1]::].transpose(-1, -2).view(feats[-1].shape) 
        trans1 = x_trans[:, 0:x_fea[0].shape[-3]*x_fea[0].shape[-2]*x_fea[0].shape[-1]].transpose(-1, -2).view(feats[-2].shape) 
        
        feat_8x = F.upsample(
            trans0, scale_factor=2, mode='trilinear') + trans1
        feat_8x = self.conv_8x(feat_8x)
        feat_4x = F.upsample(
            feat_8x, scale_factor=2, mode='trilinear') + feats[1]
        feat_4x = self.conv_4x(feat_4x)
        feat_2x = F.upsample(feat_4x, scale_factor=2, mode='trilinear') + feats[0]
        feat_2x = self.conv_2x(feat_2x)

        feat_1x = F.upsample(feat_2x, scale_factor=2, mode='trilinear')
        feats[0] = F.upsample(feats[0], scale_factor=2, mode='trilinear') 

        Cls1 = self.convc(feats[0])
        Cls2 = self.convc(feat_1x)
        Reg1 = self.convr(feats[0])
        Reg2 = self.convr(feat_1x)
        Off1 = self.convo(feats[0])
        Off2 = self.convo(feat_1x)

        out1 = torch.cat((Cls1, Off1, Reg1),1)
        size = out1.size()
        out1 = out1.view(out1.size(0), out1.size(1), -1)
        out1 = out1.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], 5)

        out = torch.cat((Cls2, Off2, Reg2),1)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], 5)

        if test_image:
            return [out, out1], feat_1x
        
        feat_1x, img_values = self.crop_and_pad(feat_1x, bbox)
        mask = self.convm11(feat_1x)
        segout = [mask]
        
        return [out, out1], segout, img_values 


def Layers(**kwargs):
    
    model = Net(BasicBlock, [2, 2, 2, 2], **kwargs)  # [2,2,3,3]
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of Backbone Params: %.2f(e6)' % (total / 1e6))
    return model

if __name__ == '__main__':
    device = torch.device("cuda")
    input = torch.ones(1, 1, 96, 96, 96).to(device)
    label = torch.ones(1, 5, 96, 96, 96).to(device)
    net = Layers().to(device)
    net.eval()
    out = net(input)
    print(out)
    print('finish')
