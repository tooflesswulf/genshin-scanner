import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image
import numpy as np

import torchvision.transforms as tr
from torchvision.transforms.functional import crop, resized_crop
from torchvision.datasets import VisionDataset


def my_resize_crop(im: Image, i, j, h, w, shape):
    crop_im = np.array(im)[i:i+h, j:j+w, :].transpose(2,0,1)

    h2, w2 = shape
    sx = (w2-1) / (w-1)
    sy = (h2-1) / (h-1)

    xx, yy = np.meshgrid(np.arange(h2), np.arange(w2))
    x1 = np.floor(xx / sx).astype(int)
    x1[x1 >= w - 1] = w - 2
    x2 = x1 + 1
    y1 = np.floor(yy / sy).astype(int)
    y1[y1 >= h - 1] = h - 2
    y2 = y1 + 1

    q11 = (x2 - xx/sx) * (y2 - yy/sy) * crop_im[:, y1, x1]
    q21 = (xx/sx - x1) * (y2 - yy/sy) * crop_im[:, y1, x2]
    q12 = (x2 - xx/sx) * (yy/sy - y1) * crop_im[:, y2, x1]
    q22 = (xx/sx - x1) * (yy/sy - y1) * crop_im[:, y2, x2]
    return Image.fromarray(np.round(q11 + q12 + q21 + q22).astype(np.uint8).transpose(1, 2, 0))


class GenshinArtifactDataset(Dataset):
    resize_impl_p = .5

    def __init__(self, root_dir, anno_db, transform=None, min_bbox_overlap=.95, im_size=100):
        self.root = root_dir
        self.transform = transform
        self.min_bbox_overlap = min_bbox_overlap
        self.im_size = im_size

        # static db
        self.image_files = [f'image{i}.png' for i in range(35)]
        self.anno_files = [f'image{i}-anno.json' for i in range(35)]

        self.annotations = anno_db
        # [json.load(open(os.path.join(root_dir, af), 'r'))
        #                     for af in self.anno_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, ix):
        im = Image.open(os.path.join(self.root, self.image_files[ix])) \
            .convert('RGB')
        anno = self.annotations[ix]

        # Produce random bbox with some minimum overlap.
        bbox = anno['bbox']
        x1, x2, y1, y2 = bbox
        wigx = (1 - self.min_bbox_overlap)*(x2 - x1)
        wigy = (1 - self.min_bbox_overlap)*(y2 - y1)
        a, b, c, d = np.clip(bbox + np.array([wigx, wigy, -wigx, -wigy]), 0, 1)

        z = np.array([-1, -1, 1, 1])
        orig_area = np.prod(bbox @ np.array([[-1, 0, 1, 0], [0, -1, 0, 1]]).T)

        rand_crop = np.random.uniform([0, 0, c, d], [a, b, 1, 1])
        aoi = np.prod(np.amin([z*bbox, z*rand_crop], axis=0)
                      @ np.array([[1, 0, 1, 0], [0, 1, 0, 1]]).T)
        while aoi / orig_area < self.min_bbox_overlap:
            rand_crop = np.random.uniform([0, 0, c, d], [a, b, 1, 1])
            aoi = np.prod(np.amin([z*bbox, z*rand_crop], axis=0)
                          @ np.array([[1, 0, 1, 0], [0, 1, 0, 1]]).T)

        # Use obtained random crop to transform image & boxes.
        x1, y1, x2, y2 = (np.array([*im.size, *im.size]) * rand_crop) \
            .astype(int)
        if np.random.random() < self.resize_impl_p:
            im = resized_crop(im, y1, x1, y2-y1, x2-x1,
                              [self.im_size, self.im_size])
        else:
            im = my_resize_crop(im, y1, x1, y2-y1, x2-x1,
                                [self.im_size, self.im_size])

        anno2 = {}
        fields = ['title', 'slot', 'mainstat', 'level',
                  'rarity', 'substat', 'set', 'lock', 'bbox']

        _a = np.array(rand_crop[:2])
        _b = np.array(rand_crop[2:])
        a = np.array([*_a, *_a])
        b = np.array([*(_b-_a), *(_b-_a)])
        for f in fields:
            anno2[f] = np.clip((anno[f] - a) / b, 0, 1)

        y_regr = []
        for f in fields:
            y_regr.extend(anno2[f])

        if self.transform is not None:
            im = self.transform(im)
        return {
            'image': im,
            'y1': torch.Tensor(y_regr),
            **anno2
        }
