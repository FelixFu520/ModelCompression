import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ['segDataset']

torch.manual_seed(0)


class segDataset(Dataset):
    """
    This is a wrapper for the seg dataset
    """

    def __init__(self, dataset_path, transforms=None):
        super(segDataset, self).__init__()
        if not os.path.exists(dataset_path):
            print("cannot find {}".format(dataset_path))
            exit(-1)
        self.dataset_path = os.path.abspath(dataset_path)
        self.imgList = []
        self.labelList = []
        all_images = [os.path.join(self.dataset_path, p) for p in os.listdir(self.dataset_path) if p.endswith("bmp")]
        for img_p in all_images:
            self.imgList.append(img_p)
            self.labelList.append(img_p[:-4]+".png")
        self.length = len(self.imgList)
        self.transforms = transforms   

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % self.length
        imgPath = self.imgList[idx]
        labelPath = self.labelList[idx]
        img = cv2.imread(imgPath,0)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        label = cv2.imread(labelPath, 0)
        label = np.where(label!=0, 1, 0)
        label = label.astype(np.int64)
        return img,label