import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
from noduleCADEvaluationLUNA16 import noduleCADEvaluation
import os 
import csv 
from multiprocessing import Pool
import functools
import SimpleITK as sitk
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=DeprecationWarning)
import os, sys
os.chdir(sys.path[0])
sys.path.append(os.getcwd())


fold = 0
annotations_filename = './10FoldCsvFiles/annotations0.csv'# path for ground truth annotations for the fold
annotations_excluded_filename = './10FoldCsvFiles/annotations_excluded0.csv'# path for excluded annotations for the fold
seriesuids_filename = './10FoldCsvFiles/seriesuids0.csv'# path for seriesuid for the fold
results_path = '../../results/test0/' #path for predicted bboxes (lbb.npy pbb.npy)
sideinfopath = '../../LUNA16/LUNA16PROPOCESSPATH/subset0/'
datapath = '../../LUNA16/DOWNLOADLUNA16PATH/subset0/'

maxeps = 170 #设置要评估的epoch范围
eps = range(0, maxeps+1, 1)#
detp = [0] 
isvis = False 
nmsthresh = 0.1 
use_softnms = False
frocarr = np.zeros((maxeps, len(detp)))
firstline = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']

def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing, isflip

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

def convertcsv(bboxfname, bboxpath, detp): 
    sliceim,origin,spacing,isflip = load_itk_image(datapath+bboxfname[:-8]+'.mhd')
    origin = np.load(sideinfopath+bboxfname[:-8]+'_origin.npy', mmap_mode='r')
    spacing = np.load(sideinfopath+bboxfname[:-8]+'_spacing.npy', mmap_mode='r')
    resolution = np.array([1, 1, 1])
    extendbox = np.load(sideinfopath+bboxfname[:-8]+'_extendbox.npy', mmap_mode='r')
    pbb = np.load(bboxpath+bboxfname, mmap_mode='r')# p z y x d
    pbbold = np.array(pbb[pbb[:,0] > detp]) 
    pbbold = np.array(pbbold[pbbold[:,-1] > 3]) 
    pbbold = pbbold[np.argsort(-pbbold[:,0])][:1000]
    print(pbbold.shape)
    pbb = nms(pbbold, nmsthresh)
    pbb = np.array(pbb[:, :-1]) 
    pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:,0], 1).T) 
    pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T) 
    if isflip: 
        Mask = np.load(sideinfopath+bboxfname[:-8]+'_mask.npy', mmap_mode='r')
        pbb[:, 2] = Mask.shape[1] - pbb[:, 2]
        pbb[:, 3] = Mask.shape[2] - pbb[:, 3]
    pos = VoxelToWorldCoord(pbb[:, 1:], origin, spacing)
    rowlist = []
    for nk in range(pos.shape[0]): 
        rowlist.append([bboxfname[:-8], pos[nk, 2], pos[nk, 1], pos[nk, 0], 1/(1+np.exp(-pbb[nk,0]))])
    return rowlist

def getfrocvalue(results_filename):
    return noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,'./')#,vis=True compare

def getcsv(detp, eps): 
    for ep in eps:
        bboxpath = results_path + str(ep) + '/' 
        for detpthresh in detp: 
            print ('ep', ep, 'detp', detpthresh)
            f = open(bboxpath + 'predanno'+ str(detpthresh) + '.csv', 'w') 
            fwriter = csv.writer(f)
            fwriter.writerow(firstline) 
            fnamelist = []
            for fname in os.listdir(bboxpath): 
                if fname.endswith('_pbb.npy'):
                    fnamelist.append(fname)
            print(len(fnamelist))
            convert = functools.partial(convertcsv, bboxpath = bboxpath, detp = detpthresh) 
            for fname in fnamelist:
                print(fname)
                rowlist = convert(fname) 
                for row in rowlist:
                    print(row)
                    fwriter.writerow(row)
            f.close()

def getfroc(detp, eps):
    maxfroc = 0
    maxep = 0
    for ep in eps:
        bboxpath = results_path + str(ep) + '/'
        predannofnamalist = []
        for detpthresh in detp: 
            predannofnamalist.append(bboxpath + 'predanno'+ str(detpthresh) + '.csv')
        
        froclist = [getfrocvalue(predanno) for predanno in predannofnamalist] 
        #(fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up)
        if maxfroc < max(froclist):
            maxep = ep
            maxfroc = max(froclist)
    print(froclist)
    print(maxfroc, maxep)

getcsv(detp, eps)
getfroc(detp, eps)

