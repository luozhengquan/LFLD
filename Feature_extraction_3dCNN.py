from __future__ import absolute_import
import torch
from p3d_model import P3D199 as P3D
from LFLD_datagen_off_shelf import DataGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import cv2
from os.path import join, isdir, isfile
from os import listdir
# import matplotlib.pyplot as plt
import random
import time
from skimage.io import ImageCollection, imread, concatenate_images
import scipy.misc as scm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
#import utilities
from sklearn.metrics import det_curve
import pdb 
import yaml

import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

roi_size = int(480/2.0*0.9)
fs_size = int(96*0.9)
#fs_len = int(145 - 1) // 8 + 1
fs_len=145

model = P3D(pretrained=True,num_classes=400)
model = model.cuda().to(device)


data_gen = DataGenerator(root_dir='/home/zhengquan.luo/LFLD_datasets')
data_gen.generateSet(rand=False)
data_dict,val_name=data_gen.getname()
features_p3d=[]

batchsize=10
i=0
border_ratio=0.1
p=0
while i<len(val_name):
    lenb=0
    b_fs=[]
    b_roi=[]
    labels_now=[]
    while lenb<batchsize and i<len(val_name):
        name=val_name[i]
        print(i,name)
        start_p = data_dict[name]['start_p']
        assert start_p is not None
        fs_c = data_dict[name]['fs']
        fs = concatenate_images(ImageCollection(fs_c))
        if len(fs.shape) > 3:
            nrof_fs, height_fs, width_fs, channel_fs = fs.shape
        else:
            nrof_fs, height_fs, width_fs = fs.shape
        fs_cropped_size = int(min([height_fs, width_fs]) * (1 - border_ratio))
        fs_h = int(start_p[0] * height_fs)
        fs_w = int(start_p[1] * width_fs)
        assert fs is not None
        roi_in = data_dict[name]['roi']
        roi_im = imread(roi_in)
        #roi_im = imresize(roi_im, size=0.5)
        height, width = roi_im.shape[:2]
        size = (int(width*0.5), int(height*0.5))
        roi_im=cv2.resize(roi_im, size, interpolation=cv2.INTER_AREA) 
        if len(roi_im.shape) == 3:
            height_roi, width_roi, channel_roi = roi_im.shape
        else:
            height_roi, width_roi = roi_im.shape
        roi_cropped_size = int(min([height_roi, width_roi]) * (1 -border_ratio))
        roi_h = int(start_p[0] * height_roi)
        roi_w = int(start_p[1] * width_roi)
        assert roi_im is not None
        label = data_dict[name]['label']
        #train_label.append(label)
        interval = 1
        norm_fs = fs[::interval, fs_h:fs_h+fs_cropped_size, fs_w:fs_w+fs_cropped_size]
        norm_fs = (norm_fs.astype(np.float32) - 127.5) / 128.0
        stacked_fs = np.stack([norm_fs, norm_fs, norm_fs], axis=0)
        stacked_fs=np.transpose(stacked_fs,(1,2,3,0))
        #train_fs.append(stacked_fs)
        norm_roi = roi_im[roi_h:roi_h+roi_cropped_size, roi_w:roi_w+roi_cropped_size]
        norm_roi = (norm_roi.astype(np.float32) - 127.5) / 128.0
        stacked_roi = np.stack([norm_roi, norm_roi, norm_roi], axis=0)
        stacked_roi=np.transpose(stacked_roi,(1,2,0))
        #train_roi.append(stacked_roi)
        b_fs.append(np.array(stacked_fs))
        b_roi.append(np.array(stacked_roi))
        labels_now.append(label)
        i=i+1
        lenb=lenb+1
    b_fs=b_fs.cuda()
    feature = model(b_fs)
    print(feature.size)
    for j in range(lenb):
        #loc=int((p%25)/5)
        #print(loc)
        features_p3d.append(feature[j])
        labels_l.append(labels_now[j])
        p=p+1
