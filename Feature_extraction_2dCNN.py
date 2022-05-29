from __future__ import absolute_import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from tensorflow.keras.models import Model
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from classification_models_3D.tfkeras import Classifiers
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Concatenate 
from LFLD_datagen_off_shelf import DataGenerator

from sklearn.decomposition import PCA
from build_c3d import build_c3d_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
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
import tensorflow as tf
#config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
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
def sigmoid_new(X,useStatus):
    if useStatus:
        return 1.0 / (1 + np.exp(-float(X)));
    else:
        return float(X)

def BPCER(y_true, y_pred):
    N1 = K.sum(y_true)
    FP1 = K.sum(y_true - y_pred * y_true)
    recall = FP1/N1
    return recall

def APCER(y_true, y_pred):
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
    
def make_meshgrid(x, y, h=1.0):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
    
roi_size = int(480/2.0*0.9)
fs_size = int(96*0.9)
#fs_len = int(145 - 1) // 8 + 1
fs_len=145
#resnet_model=Sequential()
#resnet_model=ResNet50(include_top=False, weights='imagenet', input_shape=(roi_size, roi_size, 3))
#resnet_model=InceptionV3(include_top=False, weights='imagenet', input_shape=(roi_size, roi_size, 3))
resnet_model=MobileNetV2(include_top=False, alpha= 1.0, weights='imagenet', input_shape=(roi_size, roi_size, 3))
#resnet_model.add(Flatten())
for i, layer in enumerate(resnet_model.layers):
    print(i, layer.name, layer.input_shape, layer.output_shape)

roi_fea = resnet_model.output
print(roi_fea.shape)
roi_fea = GlobalAveragePooling2D()(roi_fea)

#roi_fea = Dense(512, activation='relu')(roi_fea)

ResNet50_3D, preprocess_input = Classifiers.get('resnet50')
c3d_model=Sequential()
c3d_model.add(ResNet50_3D(
   input_shape=(145, 86, 86, 3),
   stride_size=4,
   kernel_size=3, 
   weights='imagenet'
))
c3d_model.add(Flatten())
#c3d_model = build_c3d_model(weights=True, summary=True, l_size=fs_size, l_len=fs_len)

fs_fea = c3d_model.output
fs_fea=fs_fea
fused_fea = Concatenate(axis=1)([roi_fea, fs_fea])

out_fea = Dense(512, activation='relu')(fused_fea)

predictions = Dense(1, activation='sigmoid')(out_fea)

model = Model(inputs=[c3d_model.input, resnet_model.input], outputs=fused_fea)
model_c3d=Model(inputs=c3d_model.input, outputs=fs_fea)
model_res=Model(inputs=resnet_model.input,outputs=roi_fea)
#plot_model(model, to_file='./model_fs_roi.png', show_shapes=True)

# for layer in base_model.layers:
#     layer.trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy',APCER,BPCER])

data_gen = DataGenerator(root_dir='/home/zhengquan.luo/LFLD_datasets')
data_gen.generateSet(rand=False)

#fs, roi_im, labels=data_gen.getdata(normalize=True,  debug=False)
data_dict,val_name=data_gen.getname()
#print(labels)
#print(a,b,c)
features_l=[]
features_c3d_l=[]
features_res_l=[]
#features_test=[]
#features_c3d_test=[]
#features_res_test=[]
labels_l=[]
#labels_test=[]

batchsize=8
i=0
border_ratio=0.1
p=0
#file_2dcnn_resnet50 = open('./features/2dcnn_MobileNetV2.txt',mode='w')
file_3dcnn_c3d = open('./features/3dcnn_Resnet50_3D.txt',mode='w')
#file_label = open('./features/label.txt',mode='w')
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
    feature = model.predict([np.array(b_fs), np.array(b_roi)])
    feature_c3d = model_c3d.predict(np.array(b_fs))
    feature_res = model_res.predict(np.array(b_roi))
    for j in range(lenb):
        #loc=int((p%25)/5)
        #print(loc)
        features_l.append(feature[j])
        features_c3d_l.append(feature_c3d[j])
        print(len(feature[j]))
        #print(feature_c3d[j])
        features_res_l.append(feature_res[j])
        #for pp in range(len(feature_res[j])):
        #    print(feature_res[j][pp])
        labels_l.append(labels_now[j])
        #for k in range(len(feature_res[j])):
        #    file_2dcnn_resnet50.write(str(feature_res[j][k]))
        #    file_2dcnn_resnet50.write(' ')
        
        for k in range(len(feature_c3d[j])):
            file_3dcnn_c3d.write(str(feature_c3d[j][k]))
            file_3dcnn_c3d.write(' ')
        #file_label.write(str(labels_now[j]))
        #file_2dcnn_resnet50.write('\n')
        file_3dcnn_c3d.write('\n')
        #file_label.write('\n')
        p=p+1
#file_2dcnn_resnet50.close()
file_3dcnn_c3d.close()
#file_label.close()