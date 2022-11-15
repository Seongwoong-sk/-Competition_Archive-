import os

os.system('pip install --upgrade albumentations -qqq')
os.system('pip install timm -qqq')
os.system('pip install ttach -qqq')

import cv2
import copy
import torch
import argparse
import ttach as tta
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import timm
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

class DatasetLMT(Dataset):
    def __init__(self, image_folder, label_df, transforms):
        self.image_folder = image_folder
        self.label_df = label_df
        self.transforms = transforms

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        image_fn = self.image_folder + str(self.label_df.iloc[index,0])

        image = cv2.imread(image_fn)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        label = self.label_df.iloc[index,1]

        if self.transforms:
            image = self.transforms(image=image)['image'] / 255.0
        
        return image, label



base_transforms = A.Compose([
        A.Resize(380,380),
        A.Normalize(max_pixel_value=1.0, p=1),        
        ToTensorV2(),
        ])

tta_transforms = tta.Compose([
    tta.Rotate90([0,90,180,270]),
])

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',type=str, default='./data/')
    parser.add_argument('--label_path',type=str, default='./data/sample_submission.csv')
    parser.add_argument('--weight_path',type=str, default='./save/')
    parser.add_argument('--out_path',type=str,default='./output/')

    parser.add_argument('--model',type=str,default='efficientnet-b0')
    parser.add_argument('--batch_size', type=int,default=4)

    parser.add_argument('--device',type=str,default=device)

    args = parser.parse_args()

    assert os.path.isdir(args.image_path), 'Wrong Image Path'
    assert os.path.isfile(args.label_path), 'Wrong Label Path'
    assert os.path.isfile(args.weight_path), 'Wrong Weight Path'
    assert os.path.isdir(args.out_path), 'Wrong Out Path'

    print('=' * 70)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)

    weights = glob(os.path.join(args.weight_path))

    test_df = pd.read_csv(args.label_path)

    test_set = DatasetLMT(
        image_folder = args.image_path,
        label_df = test_df,
        transforms = base_transforms
    )    

    submission_df = copy.copy(test_df)


    for weight in weights:
        model = timm.create_model(args.model, pretrained=False, num_classes=10)
        model.load_state_dict(torch.load(weight, map_location=args.device))
        print('=' * 70)
        print(f'[info msg] weight {weight} is loaded')
        print(f'[Info msg] model {args.model} is loaded')

        test_data_loader = DataLoader(
            test_set,
            batch_size = args.batch_size,
            shuffle=False,
        )

        model.to(args.device)
        

        model.eval()
        tta_model = tta.ClassificationTTAWrapper(model, tta_transforms) # Initializes internal Module state, shared by both nn.Module and Script

        batch_size = args.batch_size
        batch_index = 0

        print('=' * 70)
        print('[Info msg] INFERENCE Start !!!')

        with torch.inference_mode():

            for i, (imgs, _) in enumerate(tqdm(test_data_loader)):
                imgs = imgs.to(args.device)
                outputs = tta_model(imgs)  # soft
                _, preds = torch.max(outputs,1)
                # outputs = (outputs > 0.5).astype(int) # hard vote
                batch_index = i * batch_size
                submission_df.iloc[batch_index : batch_index+batch_size, 1:] += preds.long().cpu().detach().numpy()[:,np.newaxis]

    ###
    # submission_df.iloc[:,1:] = (submission_df.iloc[:,1:] / len(weights) > 0.35)
    SAVE_FN = os.path.join(args.out_path, datetime.now().strftime('%m%d%H%M') + '_ensemble_submission.csv')

    submission_df.to_csv(
        SAVE_FN,
        index=False
    )

    print('=' * 70)
    print(f'[info msg] submission fils is saved to {SAVE_FN}')

if __name__ == '__main__':
    main()
