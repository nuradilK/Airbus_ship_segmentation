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


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
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

def multi_rle_encode(img):
    labels = label(img)
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

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
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

class SegmentationDataset(Dataset):
    def __init__(self, in_df, train_image_dir, test_image_dir, transform=None, mode='train'):
        grp = list(in_df.groupby('ImageId'))
        self.image_ids =  [_id for _id, _ in grp] 
        self.image_masks = [m['EncodedPixels'].values for _,m in grp]
        self.transform = transform
        self.mode = mode
        self.train_image_dir = train_image_dir
        self.test_image_dir = test_image_dir
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_ids)
               
    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx]
        if self.mode == 'train':
            rgb_path = os.path.join(self.train_image_dir, img_file_name)
        else:
            rgb_path = os.path.join(self.test_image_dir, img_file_name)
        
        img = imread(rgb_path)
        mask = masks_as_image(self.image_masks[idx])
       
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            
        pixel_num = img.shape[-2] * img.shape[-3]
        sz = int(np.sqrt(pixel_num))

        if self.mode == 'train':
            return self.img_transform(img), torch.from_numpy(np.moveaxis(mask, -1, 0)).float().reshape(1, sz, sz)
        else:
            return self.img_transform(img), str(img_file_name)

# ref https://github.com/ternaus/robot-surgery-segmentation
class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask
    
class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask
    
class RandomCrop:
    def __init__(self, size):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, img, mask=None):
        height, width, _ = img.shape

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

# ref https://github.com/ternaus/robot-surgery-segmentation
class LossBinary:
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1.0).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss

def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim = -1)
    union = y_true.sum(dim=-2).sum(dim=-1).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1).sum(dim = -1)

    return (intersection / (union - intersection + epsilon)).mean()

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))

def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()
    
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
    
    '''
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
    '''

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
    
