import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
from scipy.spatial import distance
from config_data import config
import matplotlib.pyplot as plt
import math

class Data3Detector(Dataset):
    def __init__(self, data_dir, split_path, config, phase='train', split_comber=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']   
        self.stride = config['stride']       
        sizelim = config['sizelim']/config['reso']
        sizelim2 = config['sizelim2']/config['reso']
        sizelim3 = config['sizelim3']/config['reso']
        sizelim4 = config['sizelim4']/config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale'] 
        self.r_rand = config['r_rand_crop'] 
        self.augtype = config['augtype'] 
        self.pad_value = config['pad_value'] 
        self.split_comber = split_comber
        idcs = split_path 

        if phase!='test':
            idcs = [f for f in idcs if (f not in self.blacklist)]

        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]
        self.lunamasknames = [os.path.join(data_dir[:-1]+'MASK', '%s_MASK.npy' % idx) for idx in idcs]
        labels = []
        print (len(idcs))
        for idx in idcs:
            l = np.load(data_dir+idx+'_label.npy', allow_pickle=True)
            if np.all(l==0): #without nodule
                l=np.array([])
            labels.append(l)
        self.sample_bboxes = labels 
        if self.phase != 'test':
            self.bboxes = []
            self.valbboxes = []
            for i, l in enumerate(labels): 
                if len(l) > 0 : #overcome the imbalance of the number of different nodule sizes #resample
                    for t in l: 
                        if self.phase == 'train':
                            if t[3]>sizelim: 
                                self.bboxes.append([np.concatenate([[i],t])])
                            if t[3]>sizelim2:
                                self.bboxes+=[[np.concatenate([[i],t])]]*2
                            if t[3]>sizelim3:
                                self.bboxes+=[[np.concatenate([[i],t])]]*4
                            if t[3] >sizelim4:
                                self.bboxes += [[np.concatenate([[i], t])]]*8
                        else:
                            self.valbboxes.append([np.concatenate([[i],t])])
            
            if self.phase == 'train':
                self.bboxes = np.concatenate(self.bboxes,axis = 0) 
            else:
                self.valbboxes = np.concatenate(self.valbboxes,axis = 0)

        self.crop = Crop(config) 
        self.label_mapping = LabelMapping(config, self.phase) # make lable 

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time

        isRandomImg  = False
        if self.phase =='train':
            if idx>=len(self.bboxes):
                isRandom = True
                idx = np.random.randint(0, len(self.bboxes))
                isRandomImg = np.random.randint(2)
                flag = True
            else: 
                isRandom = False
        else:  
            isRandom = False
        
        if self.phase != 'test':
            if self.phase == 'train':
                if not isRandomImg:
                    bbox = self.bboxes[idx] 
                    filename = self.filenames[int(bbox[0])] 
                    imgs = np.load(filename, allow_pickle=True)
                    lunamaskname = self.lunamasknames[int(bbox[0])]
                    lunamasks = np.load(lunamaskname, allow_pickle=True)
                    bboxes = self.sample_bboxes[int(bbox[0])] 
                    isScale = self.augtype['scale'] and (self.phase=='train')
                    if isRandom:
                        while flag:
                            sample, target, crop_bboxes, samplemask, isbg = self.crop(imgs, lunamasks, bbox[1:].copy(), bboxes.copy(), isScale, isRandom)
                            flag = not isbg
                    else:
                        sample, target, crop_bboxes, samplemask, _ = self.crop(imgs, lunamasks, bbox[1:].copy(), bboxes.copy(), isScale, isRandom)
                    if self.phase=='train' and not isRandom:
                        sample, target, crop_bboxes, samplemask = augment(sample, samplemask, target, crop_bboxes,
                            ifflip = self.augtype['flip'], ifrotate=self.augtype['rotate'], ifswap = self.augtype['swap'])
                else:
                    randimid = np.random.randint(len(self.filenames))
                    filename = self.filenames[randimid]
                    imgs = np.load(filename, allow_pickle=True)
                    lunamaskname = self.lunamasknames[randimid]
                    lunamasks = np.load(lunamaskname, allow_pickle=True)
                    bboxes = self.sample_bboxes[randimid]
                    isScale = self.augtype['scale'] and (self.phase=='train')
                    while flag:
                        sample, target, crop_bboxes, samplemask, isbg = self.crop(imgs, lunamasks, [], bboxes.copy(), isScale=False,isRand=True)
                        flag = not isbg
            #val
            else:
                bbox = self.valbboxes[idx] 
                filename = self.filenames[int(bbox[0])] 
                imgs = np.load(filename, allow_pickle=True)
                lunamaskname = self.lunamasknames[int(bbox[0])]
                lunamasks = np.load(lunamaskname, allow_pickle=True)
                bboxes = self.sample_bboxes[int(bbox[0])] 
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, crop_bboxes, samplemask, _ = self.crop(imgs, lunamasks, bbox[1:].copy(), bboxes.copy(), isScale, isRandom, self.phase)#coord->position information
                    
            label = self.label_mapping(sample.shape[1:], target, crop_bboxes, samplemask) 
            sample = (sample.astype(np.float32)-128)/128 

            #samplemask crop_rescale
            if samplemask.max() == 1:
                target_cropsize = int(target[3]+6) 
                sample_size = sample.shape[-1]
                k = sample_size//target_cropsize
                while sample_size%k != 0:
                    k -= 1
                scale = int(k)
                target_cropsize = int(sample_size//scale)
                start = []
                for i in range(3):
                    start.append(int(target[i]-target_cropsize/2))
                pad = []
                pad.append([0,0])
                for i in range(3):
                    leftpad = max(0,-start[i])
                    rightpad = max(0,start[i]+target_cropsize-samplemask.shape[i+1])
                    pad.append([leftpad,rightpad])

                crop_lunamask = samplemask[:,
                    max(start[0],0):min(start[0] + target_cropsize,samplemask.shape[1]),
                    max(start[1],0):min(start[1] + target_cropsize,samplemask.shape[2]),
                    max(start[2],0):min(start[2] + target_cropsize,samplemask.shape[3])]
                
                crop_lunamask = np.pad(crop_lunamask,pad,'constant',constant_values=0)

                if self.phase == 'train':
                    samplemask = zoom(crop_lunamask,[1,scale,scale,scale],order=1) 
                else:
                    samplemask = crop_lunamask

            target_point = np.array([float(target[0]),float(target[1]),float(target[2]),float(target[3])])

            return torch.from_numpy(sample), torch.from_numpy(label), torch.from_numpy(samplemask), target_point
        
        #test
        else:
            imgs = np.load(self.filenames[idx], allow_pickle=True)
            originimg = imgs.copy()
            bboxes = self.sample_bboxes[idx]
            imgs, nzhw = self.split_comber.split(imgs) 
            imgs = (imgs.astype(np.float32)-128)/128
            return torch.from_numpy(imgs), bboxes.copy(), np.array(nzhw), originimg

    def __len__(self):
        if self.phase == 'train':
            return int(len(self.bboxes)//(1-self.r_rand))
        elif self.phase =='val':
            return len(self.valbboxes)
        else:
            return len(self.sample_bboxes)
        
        
def augment(sample, samplemask, target, bboxes, ifflip = True, ifrotate=True, ifswap = True):
                    
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand()*180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3]>target[3]) and np.all(newtarget[:3]< np.array(sample.shape[1:4])-newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
                samplemask = rotate(samplemask,angle1,axes=(2,3),reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat,box[1:3]-size/2)+size/2
            else:
                counter += 1
                if counter ==3:
                    break
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            samplemask = np.transpose(samplemask,np.concatenate([[0],axisorder+1]))
            target[:3] = target[:3][axisorder]
            bboxes[:,:3] = bboxes[:,:3][:,axisorder]
            
    if ifflip:
        flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        samplemask = np.ascontiguousarray(samplemask[:,::flipid[0],::flipid[1],::flipid[2]])
        for ax in range(3):
            if flipid[ax]==-1:
                target[ax] = np.array(sample.shape[ax+1])-target[ax]
                bboxes[:,ax]= np.array(sample.shape[ax+1])-bboxes[:,ax]
    return sample, target, bboxes, samplemask

class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size'] 
        self.bound_size = config['bound_size'] 
        self.stride = config['stride']
        self.pad_value = config['pad_value'] 
    def __call__(self, imgs, lunamasks, target, bboxes,isScale=False,isRand=False, phase='train'):

        
        if isScale: 
            radiusLim = [8.,120.]
            scaleLim = [0.75,1.25]
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]),1])
                         ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]),1])] 
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0] 
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')
        else:
            crop_size=self.crop_size

        bound_size = self.bound_size 
        
        start = []
        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r)+ 1 - bound_size
                e = np.ceil (target[i] + r)+ 1 + bound_size - crop_size[i] 
            else:
                s = np.max([imgs.shape[i+1]-crop_size[i]/2,imgs.shape[i+1]/2+bound_size])
                e = np.min([crop_size[i]/2,imgs.shape[i+1]/2-bound_size])
                target = np.array([np.nan,np.nan,np.nan,np.nan])
            if s>e:
                if phase == 'train':
                    start.append(np.random.randint(e,s))
                else:
                    start.append(np.int((e+s)/2))
            else:
                start.append(int(target[i])-crop_size[i]/2+np.random.randint(-bound_size/2,bound_size/2))

        pad = []
        pad.append([0,0])
        for i in range(3):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        crop = imgs[:,
            max(start[0],0):min(start[0] + crop_size[0],imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1],imgs.shape[2]),
            max(start[2],0):min(start[2] + crop_size[2],imgs.shape[3])]
        crop = np.pad(crop,pad,'constant',constant_values =self.pad_value)
        #mask
        crop_lunamask = lunamasks[:,
            max(start[0],0):min(start[0] + crop_size[0],imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1],imgs.shape[2]),
            max(start[2],0):min(start[2] + crop_size[2],imgs.shape[3])]
        crop_lunamask = np.pad(crop_lunamask,pad,'constant',constant_values=0)

        isbg = True
        if isRand:
            if crop_lunamask.max()==1:
                isbg = False
        if isbg:
            #target
            for i in range(3):
                target[i] = target[i] - start[i]  
                
            for i in range(len(bboxes)):
                for j in range(3):
                    bboxes[i][j] = bboxes[i][j] - start[j] 
                    
            if isScale:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    crop = zoom(crop,[1,scale,scale,scale],order=1)
                    crop_lunamask = zoom(crop_lunamask,[1,scale,scale,scale],order=1)
                newpad = self.crop_size[0]-crop.shape[1:][0]
                if newpad<0:
                    crop = crop[:,:-newpad,:-newpad,:-newpad]
                    crop_lunamask = crop_lunamask[:,:-newpad,:-newpad,:-newpad]
                elif newpad>0:
                    pad2 = [[0,0],[0,newpad],[0,newpad],[0,newpad]]
                    crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
                    crop_lunamask = np.pad(crop_lunamask, pad2, 'constant', constant_values=0)
                for i in range(4):
                    target[i] = target[i]*scale
                for i in range(len(bboxes)):
                    for j in range(4):
                        bboxes[i][j] = bboxes[i][j]*scale

        return crop, target, bboxes, crop_lunamask, isbg 

def select_samples(target, D, num, map, state, points):

    closet = np.argsort(D, axis=0).flatten()
    #positive 1
    if state == 1:
        sample_index = closet[0:num]
    #ignore 0
    if state == 0:
        sample_index = closet[0:num]
    sample_points = points[sample_index]
    print(sample_points)
    d, h, w = sample_points[:,0].astype('int'), sample_points[:,1].astype('int'), sample_points[:,2].astype('int')
    map[d, h, w, 0] = state
    if state == 1:
            #offset target-pred
            map[d, h, w, 1] = target[0] - sample_points[:,0]
            map[d, h, w, 2] = target[1] - sample_points[:,1]
            map[d, h, w, 3] = target[2] - sample_points[:,2]
            #d
            map[d, h, w, 4] = np.log(target[3])
    
    return map

class LabelMapping(object):
    def __init__(self, config, phase):

        self.phase = phase
        self.input_size = config['crop_size'] 
        self.pos_num = int(config['pos_num']) 
        zz,yy,xx = np.meshgrid(np.linspace(0, self.input_size[0]-1, self.input_size[0]),
                           np.linspace(0, self.input_size[1]-1, self.input_size[1]),
                           np.linspace(0, self.input_size[2]-1, self.input_size[2]),indexing ='ij')
        coord = np.concatenate([zz[np.newaxis,...], yy[np.newaxis,...],xx[np.newaxis,:]],0).astype('float32')
        coord = coord.reshape(coord.shape[0], -1)
        self.points = np.array(coord.T)

    def __call__(self, input_size, target, bboxes, samplemask):


        label = -1 * np.ones([input_size[0],input_size[1],input_size[2], 5], np.float32)  
        #pos 1 neg -1 ignore 0
        #exclude other nodule into ignore
        n_z, n_h, n_w = np.where(samplemask.squeeze(0)== 1)
        label[n_z, n_h, n_w, 0] = 0
        #target nan
        if np.isnan(target[0]) and self.phase == 'train':
                # neg_z, neg_h, neg_w = np.where(label[:, :, :, 0] == -1)
                # neg_idcs = random.sample(range(len(neg_z)), min(self.neg_num, len(neg_z)))
                # neg_z, neg_h, neg_w = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs]
                # label[:, :, :, 0] = 0
                # label[neg_z, neg_h, neg_w, 0] = -1
                return label

        if np.isnan(target[0]):
            return label

        target_point = np.array([float(target[0]),float(target[1]),float(target[2])])        
        D = distance.cdist(self.points, [target_point])
        label = select_samples(target, D, self.pos_num, label, 1, self.points)
        
        return label 

def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.abc.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

