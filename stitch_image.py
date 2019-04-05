#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:20:19 2019

@author: owen
"""
import pdb
from skimage.io import imsave
import argparse
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
#from torch.utils import data
from model import Generator
import pdb
from torch.nn import functional as F


from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import os
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import glob
import pdb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

gpu_id = 0
CROP_SIZE = 128
UPSCALE_FACTOR = 4
TEST_MODE = True 
MODEL_NAME = 'G_180.pt'
model = Generator(UPSCALE_FACTOR).eval()
model.to(gpu_id)
model = torch.load('models/' + MODEL_NAME)

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

high_file = 'results_high_train'
low_file = 'results_low_train'
SR_file = 'results_SR_train'
if not os.path.exists(high_file): os.mkdir(high_file)
if not os.path.exists(low_file): os.mkdir(low_file)
if not os.path.exists(SR_file): os.mkdir(SR_file)

cnt = 0
for img_file in glob.glob('../2_train/*.jpg'):
    # img_file = 'high_big_picture/All_Hail_King_Julien_80018987_boxshot_USA_en_1_571x800_50_100.jpg'
    save_name = img_file.split('/')[2]
    # pdb.set_trace()
    img = np.array(Image.open(img_file))
    low = np.array(Image.open(img_file).resize((int(img.shape[1]/4), int(img.shape[0]/4))))
    
    s = 128
    h, w = img.shape[0], img.shape[1]
    nh, nw = int(np.ceil(h/s)), int(np.ceil(w/s))
    delta_h = s*nh - h; delta_w = s*nw - w
    img_pad = np.zeros((img.shape[0]+delta_h, img.shape[1]+delta_w, 3))
    img_pad[:h, :w] = img

    hr_transform = train_hr_transform(128)
    lr_transform = train_lr_transform(128, 4)

    ########## save image patches ##########
    c = 0
    img_pad_2x = np.zeros((img_pad.shape[0], img_pad.shape[1], 3))
    for i in range(nh): 
        for j in range(nw): 
            patch = img_pad[i*s:(i+1)*s, j*s:(j+1)*s]
            
            # super resolve
            data = hr_transform(Image.fromarray(patch.astype(np.uint8)))
            data = lr_transform(data).unsqueeze(0)
            z = Variable(data).to(gpu_id)
            
            fake_img = model(z)
            fake_img = (F.tanh(fake_img) + 1) / 2        
            patch_2x = fake_img.cpu().data.transpose(1,3).transpose(1,2).numpy()[0]
            
            # stitch upsampled patches into canvas
            img_pad_2x[i*s:(i+1)*s, j*s:(j+1)*s] = patch_2x

            c += 1

    img_2x = np.zeros((img.shape[0], img.shape[1],3))
    img_2x= img_pad_2x[:img_2x.shape[0], :img_2x.shape[1]]

    imsave(str(SR_file)+'/'+str(save_name)+'.jpg', img_2x)
    imsave(str(low_file)+'/'+str(save_name)+'.jpg', low)
    imsave(str(high_file)+'/'+str(save_name)+'.jpg', img)
    
    cnt += 1
    print(cnt, save_name)



'''
if not os.path.exists('test_results'): os.mkdir('test_results')
if not os.path.exists('test_results/SR'): os.mkdir('test_results/SR')
if not os.path.exists('test_results/low'): os.mkdir('test_results/low')
if not os.path.exists('test_results/high'): os.mkdir('test_results/high')

cnt = 0
for img_file in glob.glob('../test_icon_imgs/*.jpg'):
    # img_file = 'high_big_picture/All_Hail_King_Julien_80018987_boxshot_USA_en_1_571x800_50_100.jpg'

    img = np.array(Image.open(img_file))
    low = np.array(Image.open(img_file).resize((int(img.shape[1]/2), int(img.shape[0]/2))))

    s = 128
    h, w = img.shape[0], img.shape[1]
    nh, nw = int(np.ceil(h/s)), int(np.ceil(w/s))
    delta_h = s*nh - h; delta_w = s*nw - w
    img_pad = np.zeros((img.shape[0]+delta_h, img.shape[1]+delta_w, 3))
    img_pad[:h, :w] = img

    hr_transform = train_hr_transform(128)
    lr_transform = train_lr_transform(128, 2)

    ########## save image patches ##########
    c = 0
    img_pad_2x = np.zeros((img_pad.shape[0], img_pad.shape[1], 3))
    for i in range(nh): 
        for j in range(nw): 
            patch = img_pad[i*s:(i+1)*s, j*s:(j+1)*s]
            
            # super resolve
            data = hr_transform(Image.fromarray(patch.astype(np.uint8)))
            data = lr_transform(data).unsqueeze(0)
            z = Variable(data).cuda()

            pdb.set_trace()
            fake_img = model(z)
            fake_img = (F.tanh(fake_img) + 1) / 2        
            patch_2x = fake_img.cpu().data.transpose(1,3).transpose(1,2).numpy()[0]
            
            # stitch upsampled patches into canvas
            img_pad_2x[i*s:(i+1)*s, j*s:(j+1)*s] = patch_2x

            c += 1
            # print(c)
    
    img_2x = np.zeros((img.shape[0], img.shape[1],3))
    img_2x= img_pad_2x[:img_2x.shape[0], :img_2x.shape[1]]

    imsave('test_results/SR/'+str(cnt)+'.jpg', img_2x)
    imsave('test_results/low/'+str(cnt)+'.jpg', low)
    imsave('test_results/high/'+str(cnt)+'.jpg', img)
    
    cnt += 1
    print(cnt)
'''


