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


def validation(model: nn.Module, criterion, valid_loader, batch_size):
    print("Validation of model")
    model.eval()
    losses, jaccard = [], []
    tq = tqdm(total=len(valid_loader) *  batch_size // 2, position=0, leave=True)
    tq.set_description('validation')
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
def train(lr, model, criterion, train_loader, valid_loader, validation, init_optimizer, batch_size, n_epochs=1):
    optimizer = init_optimizer(lr)
    #model = nn.DataParallel(model, device_ids=None)
    if torch.cuda.is_available():
        model.cuda()
       
    model_path = Path('model_1_UNet.pt')
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
    }, str(model_path))

    report_each = 50
    log = open('train_1_UNet.log', 'at', encoding='utf8')
    best_valid_loss = 1e9
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_loader) *  batch_size, position=0, leave=True)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
            tq.close()
            write_event(log, epoch, loss=np.mean(losses))
            valid_metrics = validation(model, criterion, valid_loader, batch_size)
            write_event(log, epoch, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                save(epoch + 1)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
        
def make_loader(in_df, batch_size, train_image_dir, test_image_dir, shuffle=False, transform=None):
    return DataLoader(
        dataset=SegmentationDataset(in_df, train_image_dir, test_image_dir, transform=transform),
        shuffle=shuffle,
        num_workers = 0,
        batch_size = batch_size,
        pin_memory=torch.cuda.is_available()
    )
    
def make_submission(model, train_image_dir, test_image_dir, batch_size):
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
    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = variable(inputs, volatile=True)
            outputs = model(inputs)
            for i, image_name in enumerate(paths):
                mask = F.sigmoid(outputs[i,0]).data.cpu().numpy()
                cur_seg = binary_opening(np.squeeze(mask>0.5), disk(2))
                cur_seg.reshape(mask.shape)
                cur_rles = multi_rle_encode(cur_seg)
                if len(cur_rles)>0:
                    for c_rle in cur_rles:
                        out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': c_rle}]
                else:
                    out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': None}]

    submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
    submission_df.to_csv('submission_UNet.csv', index=False)

def main():
    ship_dir = ''
    train_image_dir = os.path.join(ship_dir, 'train_v2')
    test_image_dir = os.path.join(ship_dir, 'test_v2')
    
    masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))
    # Drop exessive data which do not contain ships
    masks = masks.drop(masks[masks.EncodedPixels.isnull()].sample(70000,random_state=42).index)

    unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
    train_ids, valid_ids = train_test_split(unique_img_ids, 
                     test_size = 0.05, 
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
    
    train_transform = DualCompose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomCrop((256,256,3)),
    ])
    
    batch_size = 64 
    # Corrupted Data
    train_df = train_df[train_df['ImageId'] != '6384c3e78.jpg']
    train_loader = make_loader(train_df, batch_size =  batch_size, 
                               train_image_dir=train_image_dir, test_image_dir=test_image_dir, shuffle=True, transform=train_transform)
    valid_loader = make_loader(valid_df, batch_size = batch_size // 2, 
                               train_image_dir=train_image_dir, test_image_dir=test_image_dir, transform=None)

    model = UNet()
    train(init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        lr = 1e-3,
        n_epochs = 10,
        model=model,
        criterion=LossBinary(jaccard_weight=5),
        train_loader=train_loader,
        valid_loader=valid_loader,
        batch_size=batch_size,
        validation=validation,
    )

    model = UNet()
    batch_size = 64
    model_path ='model_1_UNet.pt'
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    make_submission(model, train_image_dir, test_image_dir, batch_size)

if __name__ == '__main__':
    main()
    
