import cv2
import random 
import numpy as np

# ref https://github.com/ternaus/robot-surgery-segmentation
class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask
    
# Augmentation 1: flipping an image vertically
class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask

# Augmentation 2: flipping an image horizontally
class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask
    
# Augmentation 3: randomly crop an image
class RandomCrop:
    def __init__(self, size):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, img, mask=None):
        # Get the shape of an image.
        height, width, _ = img.shape
        
        # Randomly reduce the shapes.
        h_start = np.random.randint(0, height - self.h)
        w_start = np.random.randint(0, width - self.w)
        
        img = img[h_start: h_start + self.h, w_start: w_start + self.w,:]

        assert img.shape[0] == self.h
        assert img.shape[1] == self.w
        
        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)
            mask = mask[h_start: h_start + self.h, w_start: w_start + self.w,:]

        return img, mask
