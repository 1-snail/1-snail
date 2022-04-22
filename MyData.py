# from dataset_example_4 import ClassDisjointMyDataset
import torch
import os
import numpy as np
from PIL import Image
import cv2
import random

# 数据读入和预处理
class MyDataset:
    def __init__(self, path):

        root_path = os.path.join('/mnt/datasets/metric_learning', path)  # 根目录

        filelist = os.listdir(root_path)
        # files = glob.glob(root_path+'/*/*')
        class_to_idx = {filename.split('.')[1]: int(filename.split('.')[0]) for filename in filelist}
        classes = [filename.split('.')[1] for filename in filelist]  # 类别

        files_paths = [os.path.join(root_path, x) for x in filelist]  # 每个类别文件夹详细地址
        self.img_names = []

        for file_path in files_paths:
            img_path = os.listdir(file_path)
            for img in img_path:
                self.img_names.append(os.path.join(file_path, img))
        # random.shuffle(self.img_names)
        print("en")
        self.targets = [class_to_idx[img_name.split('/')[-2].split('.')[-1]] for img_name in self.img_names]
        self.data = np.array(self.img_names)
        print("%s : %d "%(path,len(self.targets)))


class ClassDisjointMyDataset(torch.utils.data.Dataset):
    def __init__(self, original_train,original_val,train, transform):
       
        
        rule = (lambda x: x < 100) if train else (lambda x: x >= 100)
        train_filtered_idx = [
            i for i, x in enumerate(original_train.targets) if rule(x)
        ]
        val_filtered_idx = [i for i, x in enumerate(original_val.targets) if rule(x)]
        # 将训练和验证图片放一起
        self.data = np.concatenate(
            [
                original_train.data[train_filtered_idx,],
                original_val.data[val_filtered_idx,],
            ],
            axis=0,
        )
        self.targets = np.concatenate(
            [
                np.array(original_train.targets)[train_filtered_idx],
                np.array(original_val.targets)[val_filtered_idx],
            ],
            axis=0,
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, target = self.data[index], self.targets[index]
        # img = Image.open(img_name)

        # for _,img_name in enumerate(img_names):
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if h <= 608 and w <= 608:
            set_h, set_w = 500, 500
            img_after = np.zeros((set_h, set_w, 3), dtype=np.uint8)
            pose_h, pose_w = (set_h - h) // 2, (set_w - w) // 2
            img_after[pose_h:pose_h + h, pose_w:pose_w + w, :] = img
            img = Image.fromarray(img_after)
        if self.transform is not None:
            img = self.transform(img)
        target = torch.tensor(target, requires_grad=False)
        return img, target

