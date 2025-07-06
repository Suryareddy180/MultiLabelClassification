import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np
from loguru import logger as printer


class ODIR5K(Dataset):
    def __init__(self, data_path: str, annotation_path: str, train_test_size: float, is_train: bool, augment: bool = False):
        self.data_path = data_path
        self.annotation_path = annotation_path

        df = pd.read_csv(annotation_path)

        df['filepath'] = df['filename'].apply(lambda x: os.path.join(data_path, x))

        # ✅ Convert string target like "[0,1,0,1,0,0,0,0]" to list
        df['target'] = df['target'].apply(lambda x: [int(i) for i in x.strip('[]').split(',')])

        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

        if is_train:
            set_size = int(len(df) * train_test_size)
            df = df.iloc[:set_size]
            printer.info(f"Train set size: {set_size}")
        else:
            set_size = int(len(df) * (1 - train_test_size))
            df = df.iloc[-set_size:]
            printer.info(f"Test set size: {set_size}")

        self.df = df

        # ✅ Transforms
        if augment is None:
            self.img_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif augment == "Imagenet":
            self.img_transform = T.Compose([
                T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif augment == "Cifar10":
            self.img_transform = T.Compose([
                T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            raise Exception("Augmentation not supported")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        img = self.img_transform(img)

        label = torch.tensor(row['target'], dtype=torch.float32)  # ✅ Multi-hot label

        return {"data": img, "label": label}
