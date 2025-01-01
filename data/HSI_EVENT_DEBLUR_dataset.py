from io import BytesIO
import numpy as np
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import os
import scipy.io as sio
import torchvision

totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(triplet, split='train'):    
    triplet = [totensor(img) for img in triplet]
    mat = triplet[0]
    blur = triplet[1]
    gt = triplet[2]
    blur = blur[0:1,:,:] # rgb -> gray
    gt = gt[0:1,:,:]
    # horizontal flip
    if split == 'train' and random.random() < 0.5:
        mat = torch.flip(mat, [2,])
        blur = torch.flip(blur, [2,])
        gt = torch.flip(gt, [2,])
    # verticle flip
    if split == 'train' and random.random() < 0.5:
        mat = torch.flip(mat, [1,])
        blur = torch.flip(blur, [1,])
        gt = torch.flip(gt, [1,])

    ret_triplet = {'event_bin': mat, 'hsi_blur': blur, 'hsi_gt': gt}
    return ret_triplet

# This class can also be used for the real-captured CelexV + XIMEA SMHC data
class GOPRO_HSI_EventDeblurDataset(Dataset):
    # data root: contains train and test folder
    # split: train or test
    def __init__(self, dataroot, split='train', data_len=-1):
        self.dataroot = dataroot
        self.split = split
        self.split_root = os.path.join(dataroot, split)
        self.data_len = data_len
        # get paths of all event mat files, blur XIMEA frame and gt sharp frame
        all_files = [ [x[0],x[2]] for x in os.walk(self.split_root)]
        triplets = {}
        for [x,y] in all_files:
            if x not in triplets:
                triplets[x] = {}
            for z in y:
                if 'event_bin.mat' in z or 'blur.png' in z or 'gt.png' in z:
                    video_name = z.split('_')[0]
                    if video_name not in triplets[x]:
                        triplets[x][video_name] = [None, None, None]
                    if 'event_bin.mat' in z:
                        triplets[x][video_name][0] = os.path.join(x, z)
                    if 'blur.png' in z:
                        triplets[x][video_name][1] = os.path.join(x, z)
                    if 'gt.png' in z:
                        triplets[x][video_name][2] = os.path.join(x, z)
        self.all_triplets = [] # list of [event_bin.mat, blur.png, gt.png] path triplets
        for k1 in triplets.keys():
            for k2 in triplets[k1].keys():
                self.all_triplets.append(triplets[k1][k2])

        self.dataset_len = len(self.all_triplets)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        mat_path = self.all_triplets[index][0]
        blur_path = self.all_triplets[index][1]
        gt_path = self.all_triplets[index][2]
        event_bin = sio.loadmat(mat_path)['event_bin']
        # switch channel order to H X W X C
        event_bin = np.transpose(event_bin, (1, 2, 0))
        hsi_blur = Image.open(blur_path).convert("RGB")
        hsi_gt = Image.open(gt_path).convert("RGB")
        ret_triplet = transform_augment([event_bin, hsi_blur, hsi_gt], split=self.split)
        return ret_triplet


if __name__ == '__main__':
    droot = '/ssd_datasets/multispectral/gopro_dataset/GOPRO_XIMEA_blur_DAVIS346GRAY_EVENT/'
    dataset = GOPRO_HSI_EventDeblurDataset(droot)
    loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                num_workers=8,
                drop_last=True,
                pin_memory=True)
    for _, data in enumerate(loader):
        import ipdb
        ipdb.set_trace()
        print(data.keys())
