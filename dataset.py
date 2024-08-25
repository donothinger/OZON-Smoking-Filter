from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


class SigDataset(Dataset):
    def __init__(self, image_names, labels, feature_extractor, dataroot='data', transforms=None):
        self.image_names = image_names
        self.labels = labels

        if(len(image_names) != len(labels)):
            raise ValueError("image_names and labels must have the same length")
        
        self.feature_extractor = feature_extractor 

        self.dataroot = dataroot
        self.transforms = transforms


    def __getitem__(self, idx):
        imagename = self.image_names[idx]
        label = self.labels[idx]

        image = Image.open(os.path.join(self.dataroot,  imagename))
        image = image.convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        input = self.feature_extractor(images=image,return_tensors="pt")['pixel_values'].squeeze()

        return input, label

    def __len__(self):
        return len(self.image_names)

    def _preprocess(self, image):
        image = image.transpose(2, 0, 1)
        image = (image - 127.5) / 127.5

        return image