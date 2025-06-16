import argparse
import os
import time
import numpy as np
import dataloading
from importlib import import_module
import shutil
from utils import *
import sys
sys.path.append('../')
from split_combine import SplitComb
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from config_data import config
from losses_cls_seg import Loss, GetPBB, nms
from network_architecture.model import Layers
from colorama import Fore, Back, Style
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=DeprecationWarning)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Detector')
parser.add_argument('--config', '-c', default='config_training', type=str)
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--save_freq', default=1, type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test_save_dir', default=None, type=str, metavar='SAVE', 
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=1, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not') 
parser.add_argument('--testthresh', default=-3, type=float,
                    help='threshod for get pbb')
parser.add_argument('--testpos', default=800, type=int,
                    help='number of pos test')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')#
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--num_hard', default=100, type=int, metavar='N',
                    help='number of gpu for test')

def main():

    global args
    args = parser.parse_args()

    writer = SummaryWriter()
    config_training = import_module(args.config)
    config_training = config_training.config
 
    torch.manual_seed(0)

    net = Layers()
    net = net.to(device)
    loss = Loss(config)
    get_pbb = GetPBB(config)
    
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    
    testdatadir = config_training['test_preprocess_result_path']

    #test 
    if args.test == 1: 
        testfilelist = []
        for folder in config_training['test_data_path']:
            for f in os.listdir(folder):
                if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:
                    testfilelist.append(folder.split('/')[-2]+'/'+f[:-4])
        
        margin = 16 
        sidelen = 96 
        split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
        test_dataset = dataloading.Data3Detector(
            testdatadir,
            testfilelist,
            config,
            phase='test',
            split_comber=split_comber)
        test_loader = DataLoader(
            test_dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = args.workers,
            collate_fn = dataloading.collate,
            pin_memory=False)

        # check data consistency
        for i, (dataset, target, nzhw, originimg) in enumerate(test_loader): 
            if i >= len(testfilelist)/args.batch_size:
                break
            
        #load trained model    
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['state_dict'])
        test_image(test_loader, net, get_pbb, args.test_save_dir)


def test_image(data_loader, net, get_pbb, save_dir):
    
    start_time = time.time()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber

    with torch.no_grad(): 
        for i_name, (data, target, nzhw, originimg) in enumerate(data_loader): 

            nzhw = nzhw[0]
            target = [np.asarray(t, np.float32) for t in target]
            lbb = target[0]
            name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_clean')[0]
            namelist.append(name)
            data = data[0][0]
            originimg = originimg[0][0]
            #test_process
            n_per_run = args.n_test
            splitlist = range(0,len(data)+1,n_per_run)
            outputlist = []
            outfeatlist = []
            bbox = torch.tensor([[np.nan,np.nan,np.nan,np.nan]])
            for i in range(len(splitlist)-1):
                input = data[splitlist[i]:splitlist[i+1]].cuda()
                output, outfeat = net(input,bbox, test_image=True) 
                outputlist.append(output[0].data.cpu().numpy())
                size = outfeat.size()
                outfeat = outfeat.view(outfeat.size(0), outfeat.size(1), -1)
                outfeat = outfeat.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], size[1])
                outfeatlist.append(outfeat.data.cpu().numpy())
            
            output = np.concatenate(outputlist, 0)
            output = split_comber.combine(output,nzhw=nzhw)
            outfeat = np.concatenate(outfeatlist, 0)
            outfeat = split_comber.combine(outfeat,nzhw=nzhw)
            outfeat = torch.tensor(outfeat).cuda()
            outfeat = outfeat.transpose(-2,-1).transpose(-3, -2).transpose(-4, -3)

            thresh = args.testthresh 
            n = args.testpos
            pbb = get_pbb(output,thresh,n,ismask=True)  
            pbbold = np.array(pbb[pbb[:,0] > 0]) 
            pbbold = np.array(pbbold[pbbold[:,-1] > 3])  
            pbb = nms(pbbold, 0.1)
            bboxes = np.array(pbb[:, 1:]) #predited bboxes
            
            segouts = []
            img_size = outfeat.size(1)
            segout = torch.zeros([1,1,outfeat.size(1),outfeat.size(2), outfeat.size(3)],dtype=torch.float)
            for bbox in bboxes:
                target_cropsize = int(bbox[3]+6)
                k = img_size//target_cropsize
                while img_size%k != 0:
                    k -= 1
                scale = int(k)
                target_cropsize = int(img_size//scale)
                start = []
                for j in range(3):
                    start.append(int(bbox[j]-target_cropsize/2))
                pad = []
                for j in range(1,4):
                    leftpad = max(0,-start[-j])
                    rightpad = max(0,start[-j]+target_cropsize-originimg.shape[-j])
                    pad.append(leftpad)
                    pad.append(rightpad)
                image = outfeat[:,
                    max(start[0],0):min(start[0] + target_cropsize,originimg.shape[0]),
                    max(start[1],0):min(start[1] + target_cropsize,originimg.shape[1]),
                    max(start[2],0):min(start[2] + target_cropsize,originimg.shape[2])]
                
                pad = tuple(pad)
                image = F.pad(image, pad,'constant',value=0) 
                image = F.interpolate(image.unsqueeze(0), scale_factor=scale, mode='trilinear').float().cuda()
                seg_out = net.convm11(image)
                pad_d, pad_h, pad_w = [pad[4], pad[5]], [pad[2], pad[3]], [pad[0], pad[1]] 
                seg_out = F.interpolate(seg_out, scale_factor=1/scale, mode='trilinear')
                segout[:,:,
                        max(start[0],0):min(start[0] + target_cropsize,segout.shape[2]),
                        max(start[1],0):min(start[1] + target_cropsize,segout.shape[3]),
                        max(start[2],0):min(start[2] + target_cropsize,segout.shape[4])] = \
                seg_out[:, :, pad_d[0]:target_cropsize-pad_d[1], 
                        pad_h[0]:target_cropsize-pad_h[1],
                        pad_w[0]:target_cropsize-pad_w[1]]
                segouts.append(segout)
                
            if segout.max()==0:
                segouts.append(net.convm11(outfeat.unsqueeze(0).float())) 
            
            # np.save(os.path.join(save_dir, name+'_pbb.npy'), pbb) 
            # np.save(os.path.join(save_dir, name+'_lbb.npy'), lbb)
            
        # np.save(os.path.join(save_dir, 'namelist.npy'), namelist)    
        end_time = time.time()
        print('all test time is %3.2f seconds' % (end_time - start_time))
        
if __name__ == '__main__':
    main()
