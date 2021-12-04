from fastai.vision.all import *
import pandas as pd
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import torch

# Converts run-length encoding of the target variable into 768x786 matrix of 0 and 1.
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# Encodes  768x786 matrix of 0 and 1 back to run-lengths format.
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Decodes an rle-label.
def label_func(fn):
    rle = masks[fn.name]
    if not isinstance(rle, str):
        return np.zeros((768, 768), dtype=np.uint8)
    return rle_decode(rle)

# Predicts the location of a ship, and returns it in run-length format.
def predict(path, learn):
    img = cv2.imread(str(path))
    predicted_masks = learn.predict(img)
    mask = cv2.resize(predicted_masks[1].numpy().astype('uint8'), (768, 768))
    return rle_encode(mask)

# Training procedures.
def main():
    input_path = './train_v2'
    test_input_path = './test_v2'
    images = [img for img in Path(input_path).ls() if img.name[-4:] == '.jpg']
    test_images = [img for img in Path(test_input_path).ls() if img.name[-4:] == '.jpg']
    sample = [image for image in images if isinstance(masks[image.name], str)]

    # Initialize the model of batch_size = 32
    dls = SegmentationDataLoaders.from_label_func(
        input_path,
        bs=32,
        fnames=sample,
        label_func=label_func,
        item_tfms=RandomResizedCrop(256, min_scale=0.3),
        batch_tfms=aug_transforms(),
    )
    # Initialize the model of batch_size = 16
    dls2 = SegmentationDataLoaders.from_label_func(
        input_path,
        bs=16,
        fnames=sample,
        label_func=label_func,
        item_tfms=RandomResizedCrop(348, min_scale=0.3),
        batch_tfms=aug_transforms(),
    )
    
    # Initialize the model of batch_size = 8
    dls3 = SegmentationDataLoaders.from_label_func(
        input_path,
        bs=8,
        fnames=sample,
        label_func=label_func,
        batch_tfms=aug_transforms(),
    )
    
    # Training the models.
    learn = unet_learner(dls, resnet34, n_out=2, metrics=JaccardCoeff(), cbs=CSVLogger('resnet34_1.csv'))
    learn.fine_tune(10)

    learn.save('ResNet_1')

    learn = unet_learner(dls2, resnet34, n_out=2, metrics=JaccardCoeff(), cbs=CSVLogger('resnet34_2.csv'))
    learn = learn.load('ResNet_1')
    learn.fine_tune(5)

    learn.save('ResNet_2')

    learn = unet_learner(dls3, resnet34, n_out=2, metrics=JaccardCoeff(), cbs=CSVLogger('resnet34_3.csv'))
    learn = learn.load('ResNet_2')
    learn.fine_tune(5)

    learn.save('ResNet_3')
    
    # Predicting images from test set and saving the result.
    preds = []
    with learn.no_bar():
        for img in test_images:
            preds.append(predict(img.name, learn))
        
    sub = pd.DataFrame()
    sub['ImageId'] = [img.name for img in test_images] 
    sub['EncodedPixels'] = preds
    sub.to_csv('fastai_resnet34.csv', index=False)

masks = pd.read_csv(Path('./train_ship_segmentations_v2.csv'), index_col='ImageId').EncodedPixels
if __name__ == '__main__':
    main()
