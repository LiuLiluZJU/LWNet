import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high
import sqlalchemy as sq
from utils import is_dir_path,segment_lung
from pylidc.utils import consensus
from PIL import Image
from scipy.spatial.distance import pdist,squareform
import cv2
warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset','MASK_PATH'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_PATH'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_MASK'))
META_DIR = is_dir_path(parser.get('prepare_dataset','META_PATH'))

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.meta = pd.DataFrame(index=[],columns=['patient_id','series_instance_uid','study_instance_uid','total_slice','nodule_no','slice_no','original_image','mask_image',
                                                   'subtlety',
                                                    'internalStructure',
                                                    'calcification',
                                                    'sphericity',
                                                    'margin',
                                                    'lobulation',
                                                    'spiculation',
                                                    'texture','malignancy','is_cancer','is_clean','nodIDs'])
        self.meta_nodule = pd.DataFrame(index=[],columns=['patient_id','series_instance_uid','study_instance_uid','total_slice','nodule_no','x_loc','y_loc','slice_c',
                                                          'subtlety',
                                                    'internalStructure',
                                                    'calcification',
                                                    'sphericity',
                                                    'margin',
                                                    'lobulation',
                                                    'spiculation',
                                                    'texture','malignancy','is_cancer','is_clean','nodIDs'])

    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[]
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)

        malignancy = median_high(list_of_malignancy)
        if  malignancy > 3:
            return malignancy,True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'
        
    def calculate_features(self,nodule,feature_names):
        res = []
        for feature in feature_names:
            list_of_feature =[]
            for annotation in nodule:
                list_of_feature.append(getattr(annotation, feature))
            feature = median_high(list_of_feature)
            res.append(feature)
        return res
        

    def save_meta(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['patient_id','series_instance_uid','study_instance_uid','total_slice','nodule_no','x_loc','y_loc','greatest_diameter','slice_c',\
                                         'slice_no','original_image','mask_image',
                                         'subtlety',
                                        'internalStructure',
                                        'calcification',
                                        'sphericity',
                                        'margin',
                                        'lobulation',
                                        'spiculation',
                                        'texture',
                                        'malignancy','is_cancer','is_clean','nodIDs'])
        self.meta = self.meta.append(tmp,ignore_index=True)
    
    def save_meta_nodule(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['patient_id','series_instance_uid','study_instance_uid','total_slice','nodule_no','x_loc','y_loc','greatest_diameter','slice_c',
                                         'subtlety',
                                        'internalStructure',
                                        'calcification',
                                        'sphericity',
                                        'margin',
                                        'lobulation',
                                        'spiculation',
                                        'texture',
                                        'malignancy','is_cancer','is_clean','nodIDs'])
        self.meta_nodule = self.meta_nodule.append(tmp,ignore_index=True)

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)


        
        for patient in tqdm(self.IDRI_list):
            pid = patient #LIDC-IDRI-0001~
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            nodules_annotation = scan.cluster_annotations()
            study_instance_uid = scan.study_instance_uid
            series_instance_uid = scan.series_instance_uid
            vol = scan.to_volume() 
            
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))

            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

            # nodule_img_name = "{}_ONI".format(pid[-4:])
            # np.save(patient_image_dir / nodule_img_name,vol)
            nodule_mask = np.zeros(vol.shape,dtype= np.int)

            # segment lung and save original image
            # vol_slice_list = []
            # for slice_num in range(vol.shape[2]):
            #     lung_segmented = segment_lung(vol[:,:,slice_num])
            #     lung_segmented[lung_segmented==-0] = 0
            #     vol_slice_list.append(lung_segmented)
            # vol_slice_list = np.array(vol_slice_list)
            # vol_process = np.transpose(np.concatenate([vol_slice_list],axis=0),(1,2,0))
            # nodule_origin_name = "{}_PNI".format(pid[-4:])
            # np.save(patient_image_dir / nodule_origin_name,vol_process)

            if len(nodules_annotation) > 0: 
                count = 0
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    nodule_ID, nodule_x, nodule_y, nodule_cslice = list(), list(), list(), list()
                    for i in range(len(nodule)):
                        nodule_ID.append(getattr(nodule[i], '_nodule_id'))
                        nodule_x.append(getattr(nodule[i], 'centroid')[1])
                        nodule_y.append(getattr(nodule[i], 'centroid')[0])
                        nodule_cslice.append(getattr(nodule[i], 'centroid')[2])
                # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                # This current for loop iterates over total number of nodules in a single patient 
                    mask, cbbox, masks = consensus(nodule,self.c_level,self.padding)
                    slice_range = cbbox[2]
                    lung_np_array = vol[cbbox]

                    # We calculate the malignancy information
                    # malignancy, cancer_label = self.calculate_malignancy(nodule)
                    feature_names = \
                            ('subtlety',
                                'internalStructure',
                                'calcification',
                                'sphericity',
                                'margin',
                                'lobulation',
                                'spiculation',
                                'texture',
                                'malignancy')
                    subtlety,internalStructure,calcification,sphericity,margin,\
                        lobulation,spiculation,texture,malignancy = self.calculate_features(nodule, feature_names)
                    if  malignancy > 3:
                        cancer_label = True
                    elif malignancy < 3:
                        cancer_label = False
                    else:
                        cancer_label = 'Ambiguous'
                    
                    greatest_diameter = -np.inf
                    for nodule_slice in range(mask.shape[2]):
                        # This second for loop iterates over each single nodule.
                        # There are some mask sizes that are too small. These may hinder training.
                        # if np.sum(mask[:,:,nodule_slice]) <= self.mask_threshold:
                        #     continue
                        # Segment Lung part only
                        lung_segmented_np_array = segment_lung(lung_np_array[:,:,nodule_slice])
                        
                        lung_segmented_np_array[lung_segmented_np_array==-0] = 0
                        # This itereates through the slices of a single nodule
                        # Naming of each file: NI= Nodule Image, MA= Mask Original
                        nodule_slice_revise = nodule_slice + slice_range.start
                        nodule_name = "{}_NI{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice_revise])
                        mask_name = "{}_MA{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice_revise])
                        meta_list = [pid[-4:],series_instance_uid,study_instance_uid,vol.shape[2],nodule_idx,np.mean(nodule_x).round(2),np.mean(nodule_y).round(2),np.mean(nodule_cslice).round(2),greatest_diameter,\
                                      prefix[nodule_slice_revise],nodule_name,mask_name,\
                                     subtlety,internalStructure,calcification,sphericity,margin,lobulation,spiculation,texture,malignancy,cancer_label,False,nodule_ID]
                        self.save_meta(meta_list)

                        contour_array, _ = cv2.findContours(255*(mask[:,:,nodule_slice].copy()).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                        diameters = squareform(pdist(np.array(contour_array[0].squeeze(1), dtype=float)* scan.pixel_spacing))
                        diameter  = diameters.max()
                        if diameter > greatest_diameter:
                            greatest_diameter = diameter
                        # nodule_mask[:,:,nodule_slice_revise] = mask[:,:,nodule_slice]

                        if len(nodule) >= 3 and scan.slice_thickness <= 2.5:
                            nodule_mask[:,:,nodule_slice_revise] += mask[:,:,nodule_slice]
                    
                        np.save(patient_image_dir / nodule_name,lung_segmented_np_array)
                        np.save(patient_mask_dir / mask_name,mask[:,:,nodule_slice])
                    #luna16
                    if len(nodule) >= 3 and scan.slice_thickness <= 2.5:
                        x_loc = np.mean(nodule_x).round(2)
                        y_loc = np.mean(nodule_y).round(2)
                        slice_c = np.mean(nodule_cslice).round(2) 
                        meta_nodule_list = [pid[-4:],series_instance_uid,study_instance_uid,vol.shape[2],count,x_loc,y_loc,greatest_diameter, slice_c,\
                                    subtlety,internalStructure,calcification,sphericity,margin,lobulation,spiculation,texture,malignancy,cancer_label,False,nodule_ID]
                        self.save_meta_nodule(meta_nodule_list)
                        count += 1

            else:
                print("Clean Dataset",pid)
                patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
                #There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
                for slice in range(vol.shape[2]):
                    if slice >50:
                        break
                    lung_segmented_np_array = segment_lung(vol[:,:,slice])
                    lung_segmented_np_array[lung_segmented_np_array==-0] =0
                    lung_mask = np.zeros_like(lung_segmented_np_array)

                    #CN= CleanNodule, CM = CleanMask
                    nodule_name = "{}_CN001_slice{}".format(pid[-4:],prefix[slice])
                    mask_name = "{}_CM001_slice{}".format(pid[-4:],prefix[slice])
                    meta_list = [pid[-4:],series_instance_uid,study_instance_uid,vol.shape[2],slice,0,0,0,0,prefix[slice],nodule_name,mask_name,0,0,0,0,0,0,0,0,0,False,True,0]
                    self.save_meta(meta_list)
                    np.save(patient_clean_dir_image / nodule_name, lung_segmented_np_array)
                    np.save(patient_clean_dir_mask / mask_name, lung_mask)

            # self.meta_nodule.to_csv(self.meta_path+pid+'_meta_nodule_info.csv',index=False) ###
            # self.meta.to_csv(self.meta_path+pid+'_meta_info.csv',index=False)
            nodule_mask_name = "{}_OMA".format(pid[-4:])
            np.save(patient_image_dir / nodule_mask_name,nodule_mask)

        print("Saved Meta data")
        self.meta_nodule.to_csv(self.meta_path+'meta_nodule_info.csv',index=False)
        self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)

if __name__ == '__main__':
    
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()

    test= MakeDataSet(LIDC_IDRI_list,IMAGE_DIR,MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR,mask_threshold,padding,confidence_level)
    test.prepare_dataset()
