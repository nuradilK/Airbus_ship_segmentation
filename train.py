import os
import numpy as np 
import pandas as pd
from skimage.io import imread
import cv2
import random
from datetime import datetime
import json
from tqdm import tqdm
from pathlib import Path
import argparse

import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.model_selection import train_test_split
from skimage.morphology import binary_opening, disk
from skimage.morphology import label
from autoencoders import ResNet18, UNet
from augmentation import DualCompose, VerticalFlip, HorizontalFlip, RandomCrop
from utils import * 


def validation(model, criterion, valid_loader, batch_size):
    print("Validation of model")
    model.eval()
    losses, jaccard = [], []
    tq = tqdm(total=len(valid_loader) *  batch_size // 2, position=0, leave=True) # show progress bar
    tq.set_description('validation')
    # Iteration almost same as for the training epoch, a single difference is that the gradiennts are not calculated and the weights not updated
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = variable(inputs, volatile=True)
            targets = variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            jaccard += [get_jaccard(targets, (outputs > 0).float()).item()]
            tq.update(batch_size)
            tq.set_postfix(loss='{:.5f}'.format(loss.item()))

    tq.close()
    valid_loss = np.mean(losses)
    valid_jaccard = np.mean(jaccard)

    print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
    metrics = {'valid_loss': valid_loss, 'jaccard_score': valid_jaccard}
    return metrics
    
# ref  https://github.com/ternaus/robot-surgery-segmentation
def train(lr, model, criterion, train_loader, valid_loader, validation, init_optimizer, batch_size, model_path, log_path, n_epochs=1):
    optimizer = init_optimizer(lr)
    if torch.cuda.is_available():
        model.cuda()
       
    # Load the model's weights if it exists
    model_path = Path(model_path)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path)) # prevents accidental model loss

    report_each = 50
    log = open(log_path, 'at', encoding='utf8')
    best_valid_loss = 1e9
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        # Visual representation of training process (creates a progress bar)
        tq = tqdm(total=len(train_loader) *  batch_size, position=0, leave=True)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                # Gets prediction
                outputs = model(inputs)
                # Calculates loss
                loss = criterion(outputs, targets)
                # Clears gradients
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                # Calculates gradients
                loss.backward()
                # Updates weights
                optimizer.step()
                # Updates the progress bar
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                # This is needed to see batch loss online
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
            tq.close()
            # Loggings
            write_event(log, epoch, loss=np.mean(losses))
            valid_metrics = validation(model, criterion, valid_loader, batch_size)
            write_event(log, epoch, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            # Update the best model according to the valid loss
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                save(epoch + 1)
        except KeyboardInterrupt:
            # We can quit from the training safely by saving the model
            tq.close()
            print('done.')
            return
        
def make_loader(in_df, batch_size, train_image_dir, test_image_dir, shuffle=False, transform=None):
    return DataLoader(
        dataset=SegmentationDataset(in_df, train_image_dir, test_image_dir, transform=transform),
        shuffle=shuffle,
        num_workers = 0,
        batch_size = batch_size,
        pin_memory=torch.cuda.is_available()
    ) # data loader for batch training
    
def make_submission(model, submission_path, train_image_dir, test_image_dir, batch_size): # function used for Kaggle submissions
    test_paths = os.listdir(test_image_dir)
    print(len(test_paths), 'test images found')

    test_df = pd.DataFrame({'ImageId': test_paths, 'EncodedPixels':None})

    loader = DataLoader(
        dataset=SegmentationDataset(test_df, train_image_dir, test_image_dir, transform=None, mode='predict'),
        shuffle=False,
        batch_size=batch_size // 2,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    ) 

    out_pred_rows = []
    # Almost same as the validation function
    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = variable(inputs, volatile=True)
            outputs = model(inputs)
            for i, image_name in enumerate(paths):
                mask = F.sigmoid(outputs[i,0]).data.cpu().numpy()
                # This function makes outputing mask more smooth, so there will not be some random positive single pixels
                cur_seg = binary_opening(np.squeeze(mask>0.5), disk(2))
                cur_seg.reshape(mask.shape)
                # We have to encode the mask to be able to calculate the score 
                cur_rles = multi_rle_encode(cur_seg)
                if len(cur_rles)>0:
                    for c_rle in cur_rles:
                        out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': c_rle}]
                else:
                    out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': None}]

    # Make the submission file
    submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
    submission_df.to_csv(submission_path, index=False)

def load_model(model_name, model_path):
    model = None
    if model_name == 'resnet':
        model = ResNet18()
    else:
        model = UNet()
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='AutoEncoder trainer')
    parser.add_argument('--epoch', help='an epoch num', default=1)
    parser.add_argument('--batch-size', help='a batch size', default=36)
    parser.add_argument('--lr', help='learning rate', default=0.001)
    parser.add_argument('--model-name', help='model name (resnet or unet)', default='resnet')
    parser.add_argument('--model-path', help='model path to read or write', default='model.pt')
    parser.add_argument('--log-path', help='log path to read or write', default='logs.log')
    parser.add_argument('--train-dataset-dir', help='a train dataset dir', default='train_v2')
    parser.add_argument('--test-dataset-dir', help='a test dataset dir', default='test_v2')
    parser.add_argument('--df-path', help='a train data-frame path', default='train_ship_segmentations_v2.csv')
    parser.add_argument('--submission-path', help='a csv submission path', default='submission.csv')
    args = parser.parse_args()

    masks = pd.read_csv(args.df_path)
    # Drop exessive data which do not contain ships
    masks = masks.drop(masks[masks.EncodedPixels.isnull()].sample(70000,random_state=42).index)

    # Slitting the dataset into train and valid
    unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
    train_ids, valid_ids = train_test_split(unique_img_ids, 
                     test_size = 0.05, 
                     # This is important to make distribution as same as possible 
                     stratify = unique_img_ids['counts'],
                     random_state=42
                    )
    
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)
    
    train_df['counts'] = train_df.apply(lambda c_row: c_row['counts'] if 
                                        isinstance(c_row['EncodedPixels'], str) else
                                        0, 1)
    valid_df['counts'] = valid_df.apply(lambda c_row: c_row['counts'] if 
                                        isinstance(c_row['EncodedPixels'], str) else
                                        0, 1)
    
    # Augmentation techniques
    train_transform = DualCompose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomCrop((256,256,3)),
    ])
    
    # Dropping the Corrupted Data
    train_df = train_df[train_df['ImageId'] != '6384c3e78.jpg']
    train_loader = make_loader(train_df, batch_size=args.batch_size, 
                               train_image_dir=args.train_dataset_dir, test_image_dir=args.test_dataset_dir, shuffle=True, transform=train_transform)
    valid_loader = make_loader(valid_df, batch_size=args.batch_size // 2, 
                               train_image_dir=args.train_dataset_dir, test_image_dir=args.test_dataset_dir, transform=None)

    model = None
    if args.model_name == 'resnet':
        model = ResNet18()
    else:
        model = UNet()
    train(init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        lr = 1e-3,
        n_epochs = args.epoch,
        model=model,
        criterion=LossBinary(jaccard_weight=5),
        train_loader=train_loader,
        valid_loader=valid_loader,
        batch_size=args.batch_size,
        validation=validation,
        model_path=args.model_path,
        log_path=args.log_path,
    )

    model = load_model(args.model_name, args.model_path)
    make_submission(model, args.submission_path, args.train_dataset_dir, args.test_dataset_dir, args.batch_size)

if __name__ == '__main__':
    main()

