__all__ = ['read_meta_file', 'IJBDataset', 'generate_image_feature']

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Eval.utils import AlignTransform


class IJBDataset(Dataset):
    def __init__(self, img_path, img_list_path, align=True, to_tensor=True):
        self.read_meta_file(img_list_path)
        self.align = AlignTransform() if align else None
        self.to_tensor = transforms.ToTensor() if to_tensor else None
        self.imgs_path = [os.path.join(img_path, img_name)
                          for img_name in self.img_names]

    def read_meta_file(self, img_list_path):
        # read landmarks
        with open(img_list_path, 'r') as f:
            meta_file = f.read().splitlines()
        meta_file = [line.split(' ') for line in meta_file]
        self.img_names = [line[0] for line in meta_file]
        self.landmarks = [np.array(line[1:-1], dtype=np.float32).reshape((5, 2))
                          for line in meta_file]
        self.faceness_scores = [float(line[-1]) for line in meta_file]
        assert len(self.img_names) == len(self.landmarks) == len(self.faceness_scores)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, item):
        img_path = self.imgs_path[item]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[..., ::-1]  # read to RGB
        if self.align:
            landmark = self.landmarks[item]
            img = self.align(img, landmark)
        if self.to_tensor:
            img = self.to_tensor(img if self.align else img.copy())
        score = self.faceness_scores[item]
        return img, score


def read_meta_file(path, dtypes=None):
    assert len(dtypes) > 0
    with open(path, 'r') as f:
        ijb_meta = f.read().splitlines()
    ijb_meta = [line.split(' ') for line in ijb_meta]

    get_line = lambda idx: [dtypes[idx](line[idx]) for line in ijb_meta]

    return (get_line(i) for i in range(len(dtypes)))


@torch.no_grad()
def generate_image_feature(lst_path, img_path, model, batch_size, num_workers, dtype: torch.dtype,
                           flip=False, use_detector_score=False, aligned_img=False):
    model.eval()
    val_data = IJBDataset(img_path, lst_path, align=not aligned_img)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    device = torch.device("cuda:0")
    img_features = []
    for data, sc in tqdm(val_loader):
        data = data.to(device, dtype, non_blocking=True)
        sc = sc.to(device, dtype, non_blocking=True)
        # if dtype == 'float16':
        #     data = data.half()
        #     sc = sc.half()
        data_flip = data.flip(3) if flip else None
        embeddings = model(data)
        embeddings_flip = model(data_flip) if flip else None
        embeddings = F.normalize(embeddings + embeddings_flip) if flip else embeddings
        if use_detector_score:
            embeddings = embeddings * sc.unsqueeze(1)
        img_features.append(embeddings.cpu().numpy())

    img_features = np.concatenate(img_features, axis=0)
    return img_features
