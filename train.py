import argparse
import os
from math import log10
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from vis_tools import *
from data_utils import *
from loss import GeneratorLoss
from model import Generator, Discriminator
import architecture as arch
import pdb 
import torch.nn.functional as F
from torchvision.utils import save_image
from skimage.io import imsave
# import pytorch_ssim

gpu_id = 1
display = visualizer(port=8091)
report_feq = 5
NUM_EPOCHS = 20

train_set = MyDataLoader(hr_dir='../data/train/HR/', lr2x_dir='../data/train/LR_2x/', lr4x_dir='../data/train/LR_4x/')
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=2, shuffle=True)

# First Stage 2x Network
UPSCALE_FACTOR = 2
first_G = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=2, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
first_D = Discriminator()

# Second Stage 2x Network
UPSCALE_FACTOR = 2
second_G = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=2, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
second_D = Discriminator()

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    first_G.to(gpu_id)
    second_G.to(gpu_id)
    first_D.to(gpu_id)
    second_D.to(gpu_id)
    generator_criterion.to(gpu_id)

first_optimizerG = optim.Adam(first_G.parameters(), lr=0.0002)
first_optimizerD = optim.Adam(first_D.parameters(), lr=0.0002)
second_optimizerG = optim.Adam(second_G.parameters(), lr=0.0002)
second_optimizerD = optim.Adam(second_D.parameters(), lr=0.0002)

step = 0
for epoch in range(1, NUM_EPOCHS + 1):
    first_G.train(), first_D.train(), second_G.train(), second_D.train()

    for idx, (lr4x, lr2x, hr) in enumerate(train_loader):
        lr4x, lr2x, hr = lr4x.to(gpu_id), lr2x.to(gpu_id), hr.to(gpu_id)

        ############# train first D ###############
        lr2x_hat = first_G(lr4x)
        lr2x_hat = (F.tanh(lr2x_hat) + 1) / 2
        
        first_D.zero_grad()
        real_out = first_D(lr2x).mean()
        fake_out = first_D(lr2x_hat).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        first_optimizerD.step()

        ############# train first G ###############
        first_G.zero_grad()
        g_loss = generator_criterion(fake_out, lr2x_hat, lr2x)
        g_loss.backward()
        first_optimizerG.step()
        lr2x_hat = first_G(lr4x)
        lr2x_hat = (F.tanh(lr2x_hat) + 1) / 2
        
        hr_hat = second_G(lr2x_hat)
        hr_hat = (F.tanh(hr_hat) + 1) / 2
        
        if step > 10000:
            ############# train second D ###############
            second_D.zero_grad()
            real_out = second_D(hr).mean()
            fake_out = second_D(hr_hat).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            second_optimizerD.step()

            ############# train second G ###############
            second_G.zero_grad()
            g_loss = generator_criterion(fake_out, hr_hat, hr)
            g_loss.backward()
            second_optimizerG.step()
            hr_hat = second_G(lr2x_hat)
            hr_hat = (F.tanh(hr_hat) + 1) / 2
            
            if step % report_feq == 0:
                # _ssim2x = pytorch_ssim.ssim(lr2x, lr2x_hat)
                # ssim_4x = pytorch_ssim.ssim(hr, hr_hat)
                # ssim_2x_np = ssim_2x.cpu().data.numpy()
                # ssim_4x_np = ssim_4x.cpu().data.numpy()
                # err_dict = {'ssim_2x': ssim_2x_np, 
                #             'ssim_4x': ssim_4x_np}   
                # display.plot_error(err_dict)

                vis_2x_gt = lr2x[0].detach().cpu().data.numpy()
                vis_4x_gt = hr[0].detach().cpu().data.numpy()
                vis_2x_hat = lr2x_hat[0].detach().cpu().data.numpy()
                vis_4x_hat = hr_hat[0].detach().cpu().data.numpy()
                vis_low = lr4x[0].detach().cpu().data.numpy()

                display.plot_img_255(vis_2x_gt, win=1, caption='vis_2x_gt')
                display.plot_img_255(vis_4x_gt,  win=2, caption='vis_4x_gt')
                display.plot_img_255(vis_2x_hat,  win=3, caption='vis_2x_hat')
                display.plot_img_255(vis_4x_hat,  win=4, caption='vis_4x_hat')
                display.plot_img_255(vis_low,  win=5, caption='vis_low')

            ########## Save Models ##########
            if step % 5000 == 0:
                if not os.path.exists('models'): os.mkdir('models')
                torch.save(first_G, 'models/first_G_'+str(step)+'.pt')
                torch.save(first_D, 'models/first_D_'+str(step)+'.pt')
                torch.save(second_G, 'models/second_G_'+str(step)+'.pt')
                torch.save(second_D, 'models/second_D_'+str(step)+'.pt')

        print(epoch, step)

        step += 1
        
