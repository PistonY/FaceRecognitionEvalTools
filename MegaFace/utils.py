__all__ = ['MegafaceDataset', 'write_bin']

from torch.utils.data import Dataset
import cv2
import os
import struct


class MegafaceDataset(Dataset):
    def __init__(self, lst_path, root_path, transform=None):
        with open(lst_path, 'r') as f:
            self.lst_imgs = f.read().splitlines()

        self.imgs_path = [os.path.join(root_path, lp)
                          for lp in self.lst_imgs]
        self.pre_check()
        self.transform = transform

    def get_file_lst(self):
        return self.lst_imgs

    def __getitem__(self, item):
        img_path = self.imgs_path[item]
        # read img to RGB format.
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[..., ::-1]
        if self.transform is not None:
            img = self.transform(img.copy())
        return img, item

    def __len__(self):
        return len(self.imgs_path)

    def pre_check(self):
        for i, img_path in enumerate(self.imgs_path):
            if not os.path.exists(img_path):
                print(f'{img_path} not found.')
                self.imgs_path.pop(i)
                self.lst_imgs.pop(i)
        assert len(self.lst_imgs) == len(self.imgs_path)


def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))
