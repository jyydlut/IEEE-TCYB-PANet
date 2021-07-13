import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import numpy as np
import scipy.io as sio
class SalObjDataset(data.Dataset):
    mean_rgb = np.array([0.485, 0.456, 0.406])
    std_rgb = np.array([0.229, 0.224, 0.225])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)
    def __init__(self, image_root, gt_root, focal_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.mat')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.focals = sorted(self.focals) # 排序
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
    #   return image, mask
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        focal = self.focal_loader(self.focals[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        focal = np.array(focal, dtype=np.float)/255.0
        focal -= self.mean_focal
        focal /= self.std_focal
        focal = focal.transpose(2, 0, 1)
        focal = torch.from_numpy(focal).float()
        return image, gt, focal
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def focal_loader(self, path):
        with open(path, 'rb') as f:
            focal = sio.loadmat(f)
            focal = focal['img']
            return focal

    def resize(self, img, gt, focal):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), focal
        else:
            return img, gt, focal

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, focal_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, focal_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory, drop_last=True)
    return data_loader

class test_dataset:
    mean_rgb = np.array([0.485, 0.456, 0.406])
    std_rgb = np.array([0.229, 0.224, 0.225])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)
    def __init__(self, image_root, gt_root, focal_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.mat')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.focals = sorted(self.focals)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0
    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        focal = self.focal_loader(self.focals[self.index])
        focal = np.array(focal, dtype=np.float) / 255.0
        focal -= self.mean_focal
        focal /= self.std_focal
        focal = focal.transpose(2, 0, 1)
        focal = torch.from_numpy(focal).float()


        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, focal, gt, name
    def focal_loader(self, path):
        with open(path, 'rb') as f:
            focal = sio.loadmat(f)
            focal = focal['img']
            return focal

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


