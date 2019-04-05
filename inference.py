import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils import data
from model import Generator
import pdb
from torch.nn import functional as F
from skimage.io import imsave
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, MyDataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data.dataset import Dataset
import os

if not os.path.exists('../2k_celeba_hr'): os.mkdir('../2k_celeba_hr')

gpu_id = 0
CROP_SIZE = 128
UPSCALE_FACTOR = 4
TEST_MODE = True 

first_G = Generator(UPSCALE_FACTOR).eval().to(gpu_id)
first_G = torch.load('models/first_G_2.pt')
second_G = Generator(UPSCALE_FACTOR).eval().to(gpu_id)
second_G = torch.load('models/second_G_2.pt')

train_set = MyDataLoader(hr_dir='../data/train/HR/', lr2x_dir='../data/train/LR_2x/', lr4x_dir='../data/train/LR_4x/')
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=False)

cnt = 0
# train_bar = tqdm(train_loader)
for low, high in train_loader:

    high = high.to(gpu_id)
    low = low.to(gpu_id)
    
    high_hat_2x = first_G(low)
    high_hat_2x = (F.tanh(high_hat_2x) + 1) / 2
    high_hat_4x = second_G(high_hat_2x)
    high_hat_4x = (F.tanh(high_hat_4x) + 1) / 2

    # pdb.set_trace()

    # imsave('SR_results/'+str(cnt)+'.jpg', fake_img_np)
    save_image(high, '../2k_celeba_hr/'+str(cnt)+'.jpg', nrow=1, padding=0)
    save_image(high_hat_4x, '../2k_celeba_sr/'+str(cnt)+'.jpg', nrow=1, padding=0)

    cnt += 1

    print(cnt)



