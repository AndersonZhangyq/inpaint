import os
import torch
import torchvision
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
import random
import cv2
from PIL import Image


class Noise2NoiseDataset(Dataset):
    def __init__(self, image_dir, source_noise_model, target_noise_model, clean_targets = False, batch_size=32, image_size=64):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_num = len(self.image_paths)
        self.batch_size = batch_size
        self.image_size = image_size

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))
        
        # Read all images
        batch_size = self.batch_size
        batch_size = len(os.listdir(image_dir))
        image_size = self.image_size
        # self.x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        # self.y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        self.x = []
        self.y = []

        sample_id = 0

        while True:
            image_path = random.choice(self.image_paths)
            image = cv2.imread(str(image_path))
            h, w, _ = image.shape

            if h >= image_size and w >= image_size:
                h, w, _ = image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                clean_patch = image[i:i + image_size, j:j + image_size]
                # self.x[sample_id] = Image.fromarray(self.source_noise_model(clean_patch))
                # self.y[sample_id] = Image.fromarray(self.target_noise_model(clean_patch))
                self.x.append(Image.fromarray(self.source_noise_model(clean_patch)))
                if clean_targets:
                    self.y.append(Image.fromarray(clean_patch))
                else:
                    self.y.append(Image.fromarray(self.target_noise_model(clean_patch)))
                sample_id += 1
                
                if sample_id == self.image_num:
                    break
        
        # print(torch.tensor(self.x[0]).size())
    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return transforms.ToTensor()(self.x[idx]), transforms.ToTensor()(self.y[idx])

class ValDataset(Dataset):
    def __init__(self, image_dir, val_noise_model):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num = len(image_paths)
        self.data = []

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        for image_path in image_paths:
            y = cv2.imread(str(image_path))
            h, w, _ = y.shape
            y = y[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            x = val_noise_model(y)
            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return transforms.ToTensor()(self.data[idx])