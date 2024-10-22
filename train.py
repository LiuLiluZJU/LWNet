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
from losses_cls_seg import acc, Loss, GetPBB
from network_architecture.model import Net
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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save_freq', default=1, type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--save_dir', default=None, type=str, metavar='SAVE', 
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST', 
                    help='1 do test evaluation, 0 not') 


def main():

    global args
    args = parser.parse_args()

    writer = SummaryWriter()
    config_training = import_module(args.config)
    config_training = config_training.config
 
    torch.manual_seed(0)

    net = Net()
    net = net.to(device)
    loss = Loss(config)
    
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # start_epoch = args.start_epoch
    
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    
    traindatadir = config_training['train_preprocess_result_path']
    valdatadir = config_training['val_preprocess_result_path']

    #train
    print (len(trainfilelist))
    trainfilelist = []
    for folder in config_training['train_data_path']:
        print (folder)
        for f in os.listdir(folder):
            if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:
                trainfilelist.append(folder.split('/')[-2]+'/'+f[:-4])
    valfilelist = []
    for folder in config_training['val_data_path']:
        print (folder)
        for f in os.listdir(folder):
            if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:
                valfilelist.append(folder.split('/')[-2]+'/'+f[:-4])

    
    #train
    train_dataset = dataloading.Data3Detector(
        traindatadir,
        trainfilelist,
        config,
        phase = 'train')
    train_loader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory=True)
    
    #val
    val_dataset = dataloading.Data3Detector(
        valdatadir,
        valfilelist,
        config,
        phase = 'val')
    val_loader = DataLoader(
        val_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = args.workers,
        pin_memory=True)
    
    # check data consistency
    for i, (data, target, coord) in enumerate(train_loader):
        if i >= len(trainfilelist)/args.batch_size:
            break

    for i, (data, target, coord) in enumerate(val_loader):
        if i >= len(valfilelist)/args.batch_size:
            break

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum = 0.9,
        weight_decay = args.weight_decay)
    
    def get_lr(epoch):
        
        if epoch <= 20:
            lr = 0.0001
        elif epoch<=150:
            lr = 0.1 * args.lr
        elif epoch<=args.epochs:
            lr = 0.01 * args.lr

        return lr 
    
    for epoch in range(args.epochs):
        train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir, writer)   
        validate(val_loader, net, loss, epoch, writer)

def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir, writer):
    start_time = time.time()
    net.train()

    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics0 = []
    metrics1 = []

    for i, (data, target, lunamask, bbox) in enumerate(data_loader):
        data = data.to('cuda', non_blocking=True)
        target = target.to('cuda', non_blocking=True)
        lunamask = lunamask.to('cuda', non_blocking=True)
        output,segout,_ = net(data, bbox)
        loss_output0 = loss(output[0], segout, target, lunamask, flag=0) 
        loss_output1 = loss(output[1], segout, target, lunamask, flag=1) 
        loss_output = loss_output0[0] + loss_output1[0]
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step() 
        metrics0.append(loss_output0)
        metrics1.append(loss_output1)
        torch.cuda.empty_cache()
        del data, target, output, lunamask

    if epoch % save_freq==0:            
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
            
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()
    metrics0 = np.asarray(torch.tensor(metrics0, device = 'cpu'))
    print(Fore.RED+'Train Epoch %03d (lr %.5f)' % (epoch, lr)+Style.RESET_ALL)
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics0[:, 7]) / np.sum(metrics0[:, 8]),
        100.0 * np.sum(metrics0[:, 9]) / np.sum(metrics0[:, 10]),
        np.sum(metrics0[:, 8]),
        np.sum(metrics0[:, 10]),
        end_time - start_time))
    print('train loss %2.4f, classify loss %2.4f, d loss %2.4f, offset loss %2.4f, %2.4f, %2.4f, seg loss %2.4f' % (
        np.mean(metrics0[:, 0]),
        np.mean(metrics0[:, 1]),
        np.mean(metrics0[:, 2]),
        np.mean(metrics0[:, 3]),
        np.mean(metrics0[:, 4]),
        np.mean(metrics0[:, 5]),
        np.mean(metrics0[:, 6]))) # p d z y x
    
    writer.add_scalars('train loss', {'train_loss': np.mean(metrics0[:, 0])}, global_step=epoch)
    writer.add_scalars('train classify_loss', {'classify_loss': np.mean(metrics0[:, 1])}, global_step=epoch)
    writer.add_scalars('train d_loss', {'d_loss': np.mean(metrics0[:, 2])}, global_step=epoch)
    writer.add_scalars('train offset_loss/offset loss Z', {'offset loss Z': np.mean(metrics0[:, 3])}, global_step=epoch)
    writer.add_scalars('train offset_loss/offset loss Y', {'offset loss Y': np.mean(metrics0[:, 4])}, global_step=epoch)
    writer.add_scalars('train offset_loss/offset loss x', {'offset loss X': np.mean(metrics0[:, 5])}, global_step=epoch)
    writer.add_scalars('train seg_loss', {'seg_loss': np.mean(metrics0[:, 6])}, global_step=epoch)
    writer.add_scalars('train tpr', {'tpr': 100.0 * np.sum(metrics0[:, 7]) / np.sum(metrics0[:, 8])}, global_step=epoch)
    
    metrics1 = np.asarray(torch.tensor(metrics1, device = 'cpu'))
    print(Fore.RED+'Train Epoch %03d (lr %.5f)' % (epoch, lr)+Style.RESET_ALL)
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics1[:, 7]) / np.sum(metrics1[:, 8]),
        100.0 * np.sum(metrics1[:, 9]) / np.sum(metrics1[:, 10]),
        np.sum(metrics1[:, 8]),
        np.sum(metrics1[:, 10]),
        end_time - start_time))
    print('train loss %2.4f, classify loss %2.4f, d loss %2.4f, offset loss %2.4f, %2.4f, %2.4f, seg loss %2.4f' % (
        np.mean(metrics1[:, 0]),
        np.mean(metrics1[:, 1]),
        np.mean(metrics1[:, 2]),
        np.mean(metrics1[:, 3]),
        np.mean(metrics1[:, 4]),
        np.mean(metrics1[:, 5]),
        np.mean(metrics1[:, 6]))) # p d z y x
    
    writer.add_scalars('train loss2', {'train_loss': np.mean(metrics1[:, 0])}, global_step=epoch)
    writer.add_scalars('train classify_loss2', {'classify_loss': np.mean(metrics1[:, 1])}, global_step=epoch)
    writer.add_scalars('train d_loss2', {'d_loss': np.mean(metrics1[:, 2])}, global_step=epoch)
    writer.add_scalars('train offset_loss2/offset loss Z', {'offset loss Z': np.mean(metrics1[:, 3])}, global_step=epoch)
    writer.add_scalars('train offset_loss2/offset loss Y', {'offset loss Y': np.mean(metrics1[:, 4])}, global_step=epoch)
    writer.add_scalars('train offset_loss2/offset loss x', {'offset loss X': np.mean(metrics1[:, 5])}, global_step=epoch)
    writer.add_scalars('train seg_loss2', {'seg_loss': np.mean(metrics1[:, 6])}, global_step=epoch)
    writer.add_scalars('train tpr2', {'tpr': 100.0 * np.sum(metrics1[:, 7]) / np.sum(metrics1[:, 8])}, global_step=epoch)

def validate(data_loader, net, loss, epoch, writer):
    start_time = time.time()
    
    net.eval()

    metrics0 = []
    metrics1 = []
    meanDice_sum, meaniou_sum = [], []

    with torch.no_grad():
        for i, (data, target, lunamask, bbox) in enumerate(data_loader):
            data = data.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)
            lunamask = lunamask.to('cuda', non_blocking=True)
            output, segout, img_values = net(data, bbox) 
            scale = img_values[1][0].item()
            segout = F.interpolate(segout[0], scale_factor=1/scale, mode='trilinear')
            loss_output0 = loss(output[0], segout, target, lunamask, flag=0, train = False) 
            loss_output1 = loss(output[1], segout, target, lunamask, flag=1, train = False) 
            metrics0.append(loss_output0) 
            metrics1.append(loss_output1) 

            if lunamask.max() == 1:
                #------------3d-----------------#
                pred_mask = segout[0].squeeze()
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask[pred_mask >= 0.5] = 1
                pred_mask[pred_mask < 0.5] = 0
                pred_mask = pred_mask.cpu()
                lunamask = lunamask.squeeze().cpu()
                #-----------2d-----------------#
                # bbox_z = img_values[0]
                # z = int(bbox_z)
                # pred_mask = segout[0].squeeze()
                # pred_mask = torch.sigmoid(pred_mask[z,:,:])
                # pred_mask[pred_mask >= 0.5] = 1
                # pred_mask[pred_mask < 0.5] = 0
                # pred_mask = pred_mask.cpu()
                # lunamask = (lunamask.squeeze().cpu())[z,:,:]
                #------------------------------#
                TP = np.sum(np.logical_and(pred_mask.numpy().astype(np.int), lunamask.numpy().astype(np.int)))
                FP = np.sum(np.logical_and(pred_mask.numpy().astype(np.int), np.logical_not(lunamask.numpy().astype(np.int))))
                FN = np.sum(np.logical_and(np.logical_not(pred_mask.numpy().astype(np.int)), lunamask.numpy().astype(np.int)))
            
                meanDice = 2 * TP / (2 * TP + FP + FN)
                meaniou = TP / (TP + FP + FN)

                meanDice_sum.append(meanDice)
                meaniou_sum.append(meaniou)
                #--------------------------#
            torch.cuda.empty_cache()
            del data, target, output,lunamask

        end_time = time.time()
        meanDice = np.mean(np.asarray(meanDice_sum))
        meaniou = np.mean(np.asarray(meaniou_sum))
        
        metrics0 = np.asarray(torch.tensor(metrics0, device = 'cpu'))
        print(Fore.GREEN+ 'Val Epoch %03d ' % epoch + Style.RESET_ALL)
        print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
            100.0 * np.sum(metrics0[:, 7]) / np.sum(metrics0[:, 8]),
            100.0 * np.sum(metrics0[:, 9]) / np.sum(metrics0[:, 10]),
            np.sum(metrics0[:, 8]),
            np.sum(metrics0[:, 10]),
            end_time - start_time))
        print('val loss %2.4f, classify loss %2.4f, d loss %2.4f, offset loss %2.4f, %2.4f, %2.4f, seg loss %2.4f' % (
            np.mean(metrics0[:, 0]),
            np.mean(metrics0[:, 1]),
            np.mean(metrics0[:, 2]),
            np.mean(metrics0[:, 3]),
            np.mean(metrics0[:, 4]),
            np.mean(metrics0[:, 5]),
            np.mean(metrics0[:, 6]))) # p d z y x
        print("Dice: {} Iou: {}".format(meanDice, meaniou))

        writer.add_scalars('val loss', {'val loss': np.mean(metrics0[:, 0])}, global_step=epoch)
        writer.add_scalars('val classify_loss', {'classify_loss': np.mean(metrics0[:, 1])}, global_step=epoch)
        writer.add_scalars('val d_loss', {'d_loss': np.mean(metrics0[:, 2])}, global_step=epoch)
        writer.add_scalars('val offset_loss/offset loss Z', {'offset loss Z': np.mean(metrics0[:, 3])}, global_step=epoch)
        writer.add_scalars('val offset_loss/offset loss Y', {'offset loss Y': np.mean(metrics0[:, 4])}, global_step=epoch)
        writer.add_scalars('val offset_loss/offset loss x', {'offset loss X': np.mean(metrics0[:, 5])}, global_step=epoch)
        writer.add_scalars('val seg_loss', {'d_loss': np.mean(metrics0[:, 6])}, global_step=epoch)
        writer.add_scalars('val/tpr', {'tpr': 100.0 * np.sum(metrics0[:, 7]) / np.sum(metrics0[:, 8])}, global_step=epoch)
        writer.add_scalars('val seg/avg_dice', {'globalDice': meanDice}, global_step=epoch)
        writer.add_scalars('val seg/avg_iou', {'globalDice': meaniou}, global_step=epoch)


if __name__ == '__main__':
    main()
