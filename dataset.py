import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image
import numpy as np

import torchvision.transforms as tr
from torchvision.transforms.functional import crop, resized_crop
from torchvision.datasets import VisionDataset


class GenshinArtifactDataset(Dataset):
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
        a, b, c, d = bbox + np.array([wigx, wigy, -wigx, -wigy])

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
        im = resized_crop(im, y1, x1, y2-y1, x2-x1,
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
