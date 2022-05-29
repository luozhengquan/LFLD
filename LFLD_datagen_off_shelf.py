# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
"""
Light Field Liveliness Detection (LFLD) Dataset Generator
 
@author: Yunlong Wang
@mail : yunlong.wang(at)cripac.ia.ac.cn

"""
import numpy as np
import cv2
import os
from os.path import join, isdir, isfile
from os import listdir
# import matplotlib.pyplot as plt
import random
import time
from skimage.io import ImageCollection, imread, concatenate_images
import scipy.misc as scm
#from scipy.misc import imresize

def crop_border(mb, border_ratio):

    if mb is 'LT':
        s_h = 0
        s_w = 0
    elif mb is 'RT':
        s_h = 0
        s_w = border_ratio
    elif mb is 'LB':
        s_h = border_ratio
        s_w = 0
    elif mb is 'RB':
        s_h = border_ratio
        s_w = border_ratio
    elif mb is 'C':
        s_h = border_ratio // 2
        s_w = border_ratio // 2
    else:
        raise IOError('No such options.')

    return [s_h, s_w]



class DataGenerator():
    """
    DataGenerator Class : To generate Train, Validation and Test sets for LFLD datasets
    Formalized DATA:
        Inputs (two modalities):
            1) Focal stacks:
            Inputs have a shape of (Batchsize) X (Number of slices: 145) X (Height: 88) X (Width: 120) X (Channels: 1)
            Notice original height 96, width 128, i.e., cropped border 8 pixels
            2) Crop ROI
            Inputs have a shape of (Batchsize) X (Height: 440) X (Width: 600) X (Channels: 1)
            Notice orignial height 480, width 640, i.e., cropped border 40 pixels
        Outputs:
            Labels have a shape (Batchsize) X (Classes: 2)
            Notice Fake 0 and Real 1
    """

    def __init__(self, root_dir=None, border_ratio=0.1):
        """ Initializer
        Args:
            root_dir            : Directory containing everything
            train_data_file        : Text file with training set data
            remove_joints        : Joints List to keep (See documentation)
        """
        self.root_dir = root_dir
        assert isdir(self.root_dir)
        self.fs_dir = join(self.root_dir, 'Focal_Stack')
        assert isdir(self.fs_dir)
        self.real_fs_dir = join(self.fs_dir, 'Real')
        assert isdir(self.real_fs_dir)
        self.fake_fs_dir = join(self.fs_dir, 'Fake')
        assert isdir(self.fake_fs_dir)
        self.roi_dir = join(root_dir, 'Crop_ROI')
        assert isdir(self.roi_dir)
        self.l_roi_dir = join(self.roi_dir, 'ROI_L')
        assert isdir(self.l_roi_dir)
        self.r_roi_dir = join(self.roi_dir, 'ROI_R')
        assert isdir(self.r_roi_dir)

        self.border_ratio = border_ratio

    # --------------------Generator Initialization Methods ---------------------

    def _create_train_table(self):
        """ Create Table of samples from TEXT file
        """
        self.train_table = []
        self.data_dict = {}
        print('=' * 50)
        print('READING TRAIN DATA OF REAL SAMPLES.')
        print('>>> Real Samples, %d in total.' % len(listdir(self.real_fs_dir)))
        for sub_dir in listdir(self.real_fs_dir):
            label = 1
            print('=' * 50)
            print(sub_dir)
            if isdir(join(self.real_fs_dir, sub_dir)):
                l_fs_dir = join(self.real_fs_dir, sub_dir, 'LEye')
                assert isdir(l_fs_dir)
                l_fs_c = l_fs_dir + '/*.jpg'
                l_roi_imname = join(self.l_roi_dir, str(sub_dir) + '_Lcrop.bmp')
                print(l_roi_imname)
                assert isfile(l_roi_imname)
                # Five cropping mode: Left-Top (LT), Right-Top (RT), Left-Bottom (LB), Right-Bottom (RB), Central (C)
                for mb in ['LT', 'RT', 'LB', 'RB', 'C']:
                    name = str(sub_dir) + '_LEye_' + mb
                    start_p = crop_border(mb, self.border_ratio)
                    print('Sample name: ', name, ' Start Point: ', str(start_p), ' Label:', str(label))
                    self.data_dict[name] = {'fs': l_fs_c, 'roi': l_roi_imname, 'start_p': start_p, 'label': label}
                    self.train_table.append(name)

                r_fs_dir = join(self.real_fs_dir, sub_dir, 'REye')
                assert isdir(r_fs_dir)
                r_fs_c = r_fs_dir + '/*.jpg'
                r_roi_imname = join(self.r_roi_dir, str(sub_dir) + '_Rcrop.bmp')
                assert isfile(r_roi_imname)
                # Five cropping mode: Left-Top (LT), Right-Top (RT), Left-Bottom (LB), Right-Bottom (RB), Central (C)
                for mb in ['LT', 'RT', 'LB', 'RB', 'C']:
                    name = str(sub_dir) + '_REye_' + mb
                    start_p = crop_border(mb, self.border_ratio)
                    print('Sample name: ', name, ' Start Point: ', str(start_p), ' Label:', str(label))
                    self.data_dict[name] = {'fs': r_fs_c, 'roi': r_roi_imname, 'start_p': start_p, 'label': label}
                    self.train_table.append(name)

        print('>>> Fake Samples, %d in total.' % len(listdir(self.fake_fs_dir)))
        for sub_dir in listdir(self.fake_fs_dir):
            label = 0
            print('=' * 50)
            print(sub_dir)
            if isdir(join(self.fake_fs_dir, sub_dir)):
                l_fs_dir = join(self.fake_fs_dir, sub_dir, 'LEye')
                assert isdir(l_fs_dir)
                l_fs_c = l_fs_dir + '/*.jpg'
                l_roi_imname = join(self.l_roi_dir, str(sub_dir) + '_Lcrop.bmp')
                assert isfile(l_roi_imname)
                # Five cropping mode: Left-Top (LT), Right-Top (RT), Left-Bottom (LB), Right-Bottom (RB), Central (C)
                for mb in ['LT', 'RT', 'LB', 'RB', 'C']:
                    name = str(sub_dir) + '_LEye_' + mb
                    start_p = crop_border(mb, self.border_ratio)
                    print('Sample name: ', name, ' Start Point: ', str(start_p), ' Label:', str(label))
                    self.data_dict[name] = {'fs': l_fs_c, 'roi': l_roi_imname, 'start_p': start_p, 'label': label}
                    self.train_table.append(name)

                r_fs_dir = join(self.fake_fs_dir, sub_dir, 'REye')
                assert isdir(r_fs_dir)
                r_fs_c = r_fs_dir + '/*.jpg'
                r_roi_imname = join(self.r_roi_dir, str(sub_dir) + '_Rcrop.bmp')
                assert isfile(r_roi_imname)
                # Five cropping mode: Left-Top (LT), Right-Top (RT), Left-Bottom (LB), Right-Bottom (RB), Central (C)
                for mb in ['LT', 'RT', 'LB', 'RB', 'C']:
                    name = str(sub_dir) + '_REye_' + mb
                    start_p = crop_border(mb, self.border_ratio)
                    print('Sample name: ', name, ' Start Point: ', str(start_p), ' Label:', str(label))
                    self.data_dict[name] = {'fs': r_fs_c, 'roi': r_roi_imname, 'start_p': start_p, 'label': label}
                    self.train_table.append(name)
        print('=*'*50)
        print('%d samples in total.' % len(self.train_table))

    def _randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.train_table)


    def _give_batch_name(self, batch_size=16, set='train'):
        """ Returns a List of Samples
        Args:
            batch_size    : Number of sample wanted
            set                : Set to use (valid/train)
        """
        list_file = []
        for i in range(batch_size):
            if set == 'train':
                list_file.append(random.choice(self.train_set))
            elif set == 'valid':
                list_file.append(random.choice(self.valid_set))
            else:
                print('Set must be : train/valid')
                break
        return list_file

    def _create_sets(self, validation_rate=1.0):
        """ Select Elements to feed training and validation set
        Args:
            validation_rate        : Percentage of validation data (in ]0,1[, don't waste time use 0.1)
        """
        sample = len(self.train_table)
        valid_sample = int(sample * validation_rate)
        self.train_set = self.train_table[:sample - valid_sample]
        self.valid_set = self.train_table[sample - valid_sample:]
        print('SET CREATED')
        np.save('Dataset-Validation-Set', self.valid_set)
        np.save('Dataset-Training-Set', self.train_set)
        print('--Training set :', len(self.train_set), ' samples.')
        print('--Validation set :', len(self.valid_set), ' samples.')

    def generateSet(self, rand=False):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._create_train_table()
        if rand:
            self._randomize()
        self._create_sets()

    # ----------------------- Batch Generator ----------------------------------
    def _aux_generator(self, batch_size=16, normalize=True, sample_set='train', debug=False):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        while True:
            train_fs = []
            train_roi = []
            train_label = []
            i = 0
            while i < batch_size:
                if sample_set == 'train':
                    name = random.choice(self.train_set)
                elif sample_set == 'valid':
                    name = random.choice(self.valid_set)
                start_p = self.data_dict[name]['start_p']
                assert start_p is not None
                fs_c = self.data_dict[name]['fs']
                fs = concatenate_images(ImageCollection(fs_c))
                if len(fs.shape) > 3:
                    nrof_fs, height_fs, width_fs, channel_fs = fs.shape
                else:
                    nrof_fs, height_fs, width_fs = fs.shape
                fs_cropped_size = int(min([height_fs, width_fs]) * (1 - self.border_ratio))
                fs_h = int(start_p[0] * height_fs)
                fs_w = int(start_p[1] * width_fs)
                assert fs is not None
                roi_in = self.data_dict[name]['roi']
                roi_im = imread(roi_in)
                #roi_im = imresize(roi_im, size=0.5)
                height, width = roi_im.shape[:2]
                size = (int(width*0.5), int(height*0.5))
                roi_im=cv2.resize(roi_im, size, interpolation=cv2.INTER_AREA) 
                if len(roi_im.shape) == 3:
                    height_roi, width_roi, channel_roi = roi_im.shape
                else:
                    height_roi, width_roi = roi_im.shape

                roi_cropped_size = int(min([height_roi, width_roi]) * (1 -self.border_ratio))
                roi_h = int(start_p[0] * height_roi)
                roi_w = int(start_p[1] * width_roi)
                assert roi_im is not None
                label = self.data_dict[name]['label']
                train_label.append(label)

                interval = 1

                if normalize:
                    norm_fs = fs[::interval, fs_h:fs_h+fs_cropped_size, fs_w:fs_w+fs_cropped_size]
                    norm_fs = (norm_fs.astype(np.float32) - 127.5) / 128.0
                    stacked_fs = np.stack([norm_fs, norm_fs, norm_fs], axis=0)
                    stacked_fs=np.transpose(stacked_fs,(1,2,3,0))
                    train_fs.append(stacked_fs)
                    norm_roi = roi_im[roi_h:roi_h+roi_cropped_size, roi_w:roi_w+roi_cropped_size]
                    norm_roi = (norm_roi.astype(np.float32) - 127.5) / 128.0
                    stacked_roi = np.stack([norm_roi, norm_roi, norm_roi], axis=0)
                    stacked_roi=np.transpose(stacked_roi,(1,2,0))
                    train_roi.append(stacked_roi)
                else:
                    cropped_fs = fs[::interval, fs_h:fs_h+fs_cropped_size, fs_w:fs_w+fs_cropped_size]
                    stacked_fs = np.stack([cropped_fs, cropped_fs, cropped_fs], axis=0)
                    stacked_fs=np.transpose(stacked_fs,(1,2,3,0))
                    train_fs.append(stacked_fs.astype(np.float32))
                    cropped_roi = roi_im[roi_h:roi_h+roi_cropped_size, roi_w:roi_w+roi_cropped_size]
                    stacked_roi = np.stack([cropped_roi, cropped_roi, cropped_roi], axis=0)
                    stacked_roi=np.transpose(stacked_roi,(1,2,0))
                    train_roi.append(stacked_roi.astype(np.float32))

                i = i + 1


            if debug:
                print('='*50)
                print('Mini-batch shape: ',
                      np.array(train_fs).shape, ' ',
                      np.array(train_roi).shape, ' ',
                      np.array(train_label).shape)

            yield ([np.array(train_fs), np.array(train_roi)], np.array(train_label))
    
    def getname(self,):
        return self.data_dict,self.valid_set
    def generator(self, batchSize=16, norm=True, sample='train'):
        """ Create a Sample Generator
        Args:
            batchSize     : Number of image per batch
            norm               : (bool) True to normalize the batch
            sample          : 'train'/'valid' Default: 'train'
        """
        return self._aux_generator(batch_size=batchSize, normalize=norm, sample_set=sample)


    def getdata(self, normalize=True, debug=False):
        train_fs = []
        train_roi = []
        train_label = []
        #i = 0
        #while i < batch_size:
        for name in self.valid_set:
            print(name)
            #if sample_set == 'train':
            #    name = random.choice(self.train_set)
            #elif sample_set == 'valid':
            #    name = random.choice(self.valid_set)
            start_p = self.data_dict[name]['start_p']
            assert start_p is not None
            fs_c = self.data_dict[name]['fs']
            fs = concatenate_images(ImageCollection(fs_c))
            if len(fs.shape) > 3:
                nrof_fs, height_fs, width_fs, channel_fs = fs.shape
            else:
                nrof_fs, height_fs, width_fs = fs.shape
            fs_cropped_size = int(min([height_fs, width_fs]) * (1 - self.border_ratio))
            fs_h = int(start_p[0] * height_fs)
            fs_w = int(start_p[1] * width_fs)
            assert fs is not None
            roi_in = self.data_dict[name]['roi']
            roi_im = imread(roi_in)
            #roi_im = imresize(roi_im, size=0.5)
            height, width = roi_im.shape[:2]
            size = (int(width*0.5), int(height*0.5))
            roi_im=cv2.resize(roi_im, size, interpolation=cv2.INTER_AREA) 
            if len(roi_im.shape) == 3:
                height_roi, width_roi, channel_roi = roi_im.shape
            else:
                height_roi, width_roi = roi_im.shape
            roi_cropped_size = int(min([height_roi, width_roi]) * (1 -self.border_ratio))
            roi_h = int(start_p[0] * height_roi)
            roi_w = int(start_p[1] * width_roi)
            assert roi_im is not None
            label = self.data_dict[name]['label']
            train_label.append(label)
              
            interval = 1
            if normalize:
                norm_fs = fs[::interval, fs_h:fs_h+fs_cropped_size, fs_w:fs_w+fs_cropped_size]
                norm_fs = (norm_fs.astype(np.float32) - 127.5) / 128.0
                stacked_fs = np.stack([norm_fs, norm_fs, norm_fs], axis=0)
                stacked_fs=np.transpose(stacked_fs,(1,2,3,0))
                train_fs.append(stacked_fs)
                norm_roi = roi_im[roi_h:roi_h+roi_cropped_size, roi_w:roi_w+roi_cropped_size]
                norm_roi = (norm_roi.astype(np.float32) - 127.5) / 128.0
                stacked_roi = np.stack([norm_roi, norm_roi, norm_roi], axis=0)
                stacked_roi=np.transpose(stacked_roi,(1,2,0))
                train_roi.append(stacked_roi)
            else:
                cropped_fs = fs[::interval, fs_h:fs_h+fs_cropped_size, fs_w:fs_w+fs_cropped_size]
                stacked_fs = np.stack([cropped_fs, cropped_fs, cropped_fs], axis=0)
                stacked_fs=np.transpose(stacked_fs,(1,2,3,0))
                train_fs.append(stacked_fs.astype(np.float32))
                cropped_roi = roi_im[roi_h:roi_h+roi_cropped_size, roi_w:roi_w+roi_cropped_size]
                stacked_roi = np.stack([cropped_roi, cropped_roi, cropped_roi], axis=0)
                stacked_roi=np.transpose(stacked_roi,(1,2,0))
                train_roi.append(stacked_roi.astype(np.float32))
                
        return np.array(train_fs), np.array(train_roi), np.array(train_label)
            #yield ([np.array(train_fs), np.array(train_roi)], np.array(train_label))
      
    def getSample(self, sample=None):
        """
        Returns information of a sample

        """
        if sample != None:
            try:
                start_p = self.data_dict[name]['start_p']
                # assert start_p is not None
                fs_c = self.data_dict[name]['fs']
                fs = np.array(ImageCollection(fs_c))
                # assert fs is not None
                roi_in = self.data_dict[name]['roi']
                roi_im = imread(roi_in)
                # assert roi_im is not None
                label = self.data_dict[name]['label']
                return start_p, fs, roi_im, label
            except:
                return False
        else:
            print('Specify a sample name')
