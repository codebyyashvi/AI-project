import os
import numpy as np
from PIL import Image
from torch.utils import data

known_things = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'boat', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'chair', 'couch', 'potted plant', 'tv', 'bottle', 'dining table']

stuff = ['banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard',
         'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
         'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile',
         'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other',
         'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin',
         'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing',
         'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other',
         'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent',
         'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
         'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood']

unknown_things = [
    'knife', 'traffic light', 'elephant', 'sandwich', 'suitcase', 'stop sign', 'vase',
    'wine glass', 'bear', 'pizza', 'parking meter', 'baseball glove', 'cell phone', 'tie',
    'microwave', 'cup', 'giraffe', 'clock', 'sink', 'book', 'refrigerator', 'keyboard',
    'fire hydrant', 'zebra', 'backpack', 'baseball bat', 'snowboard', 'apple', 'hot dog', 'skis', 'mouse',
    'tennis racket', 'donut', 'toilet', 'broccoli', 'fork', 'skateboard', 'carrot',
    'teddy bear', 'bowl', 'oven', 'toaster', 'spoon', 'bench', 'handbag', 'remote', 'frisbee',
    'sports ball', 'hair drier', 'bed', 'kite', 'umbrella', 'orange', 'banana', 'truck',
    'laptop', 'toothbrush', 'cake', 'surfboard', 'scissors'
]


class COCO20(data.Dataset):

    def __init__(self, root, split='val', image_transform=None, target_transform=None, joint_transform=None):
        self.root = root

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        assert split in ['train', 'val']

        subset = f'{split}2017'
        self.images_base = os.path.join(self.root, 'images', subset)
        self.annotations_base = os.path.join(self.root, 'labels', subset)

        self.mapper = COCO20.load_mapping(os.path.join(self.root, 'labels.txt'))
        self.annotations = self.load_annotations(split)

        if len(self.annotations) == 0:
            raise Exception("> No files found in %s" % self.annotations_base)

        print("> Found %d images..." % len(self.annotations))

    def load_annotations(self, split):
        with open(f'data/datasets/coco/coco_stuff_{split}_images.txt', 'r') as f:
            annotations = f.readlines()
        annotations = [x.strip() for x in annotations]
        return annotations

    @staticmethod
    def load_mapping(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]

        idx = 0
        unk_id = len(known_things)
        ignore_id = unk_id + 1
        mapper = np.ones(256) * ignore_id
        for line in lines:
            id, name = line.split(': ')
            id = int(id) - 1
            if name in known_things:
                mapper[id] = idx
                # print(f"Class {name} mapped to {idx}")
                idx += 1
            elif name in stuff:
                mapper[id] = ignore_id
            elif name in unknown_things:
                mapper[id] = unk_id
            else:
                mapper[id] = ignore_id
                # print(f"Class removed from coco {name}")

        return mapper

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        lbl_path = os.path.join(self.annotations_base, self.annotations[index])
        img_path = os.path.join(self.images_base, self.annotations[index].replace('.png', '.jpg'))

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = self.image_transform(image)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        label = Image.open(lbl_path)
        label = np.array(label)
        label = self.mapper[label]
        label = self.target_transform(label)
        return self.joint_transform((image, label)) if self.joint_transform else (image, label)
