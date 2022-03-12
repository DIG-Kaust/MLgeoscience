import os
import numpy as np
import torch

from skimage import io
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset


class SaltDataset(Dataset):
    """Salt Dataset

    Parameters
    ----------
    imagedir : :obj:`str`
        Directory containing images
    maskdir : :obj:`str`
        Directory containing masks
    files : :obj:`list`
        File names (common for images and masks)
    transform : :obj:`torchvision.transforms`
        Transformation to be applied to images
    transformmask : :obj:`torchvision.transforms`
        Transformation to be applied to masks

    """
    def __init__(self, imagedir, maskdir, files, transform=None, transformmask=None):
        self.imagedir = imagedir
        self.maskdir = maskdir
        self.transform = transform
        self.transformmask = transformmask
        self.total_imgs = files

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.imagedir, self.total_imgs[idx])
        mask_loc = os.path.join(self.maskdir, self.total_imgs[idx])
        image = io.imread(img_loc, as_gray=True).squeeze().astype(np.float32)
        mask = io.imread(mask_loc, as_gray=True).astype(np.float32).squeeze() / 65535.
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['mask'] = self.transformmask(sample['mask'])
        return sample


class DeMean(object):
    """Center mean around 0 and entire range between -1 and 1
    """
    def __init__(self):
        pass

    def __call__(self, image):
        image_mean = image.mean()
        image -= image_mean
        image_max = torch.abs(image).max()
        if image_max != 0:
            image = image / image_max
        return image


class Binarize(object):
    """Binarize mask (0 or 1)
    """
    def __init__(self):
        pass

    def __call__(self, image):
        image[image > 0.5] = 1.
        image[image <= 0.5] = 0.
        return image