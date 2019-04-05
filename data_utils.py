from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import glob
import pdb

def toTensor_transform():
    return Compose([
        ToTensor()
    ])

class MyDataLoader(Dataset):
    def __init__(self, hr_dir='../data/train/HR/', lr2x_dir='../data/train/LR_2x/', lr4x_dir='../data/train/LR_4x/'):
        super(MyDataLoader, self).__init__()

        n_imgs = 0
        for file in glob.glob(str(hr_dir)+'*.jpg'):
            n_imgs += 1

        hr_list = []; lr2x_list = []; lr4x_list = []
        for i in range(n_imgs):
            hr_list.append(str(hr_dir)+str(i)+'.jpg')
            lr2x_list.append(str(lr2x_dir)+str(i)+'.jpg')
            lr4x_list.append(str(lr4x_dir)+str(i)+'.jpg')

        self.transform = toTensor_transform()
        self.hr_list = hr_list
        self.lr2x_list = lr2x_list
        self.lr4x_list = lr4x_list

    def __getitem__(self, idx):
        hr = self.transform(Image.open(self.hr_list[idx]))
        lr2x = self.transform(Image.open(self.lr2x_list[idx]))
        lr4x = self.transform(Image.open(self.lr4x_list[idx]))
        return lr4x, lr2x, hr

    def __len__(self):
        return len(self.hr_list)



'''
train_set = MyDataLoader()
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=100, shuffle=True)

for idx, (lr4x, lr2x, hr) in enumerate(train_loader):
    print(idx, lr4x.shape, lr2x.shape, hr.shape)
'''















