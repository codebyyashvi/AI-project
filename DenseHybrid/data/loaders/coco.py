from torchvision import transforms as tf
from torch.utils.data import DataLoader
from data.datasets import COCO20
import torch


def load_coco20_train_val(dataroot, bs, train_transforms, val_transforms):
    def mpr(x):
        x = torch.from_numpy(x).long().unsqueeze(0)
        x[x == 21] = 20
        return x

    remap_labels = tf.Lambda(mpr)
    train_set = COCO20(dataroot, split='train',
                       image_transform=None if not train_transforms['image'] else tf.Compose(train_transforms['image']),
                       target_transform=tf.Compose(train_transforms['target'] + [remap_labels]),
                       joint_transform=None if not train_transforms['joint'] else tf.Compose(train_transforms['joint']))
    print(f"> Loaded {len(train_set)} train images.")

    val_set = COCO20(dataroot, split='val',
                     image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                     target_transform=tf.Compose(val_transforms['target'] + [remap_labels]),
                     joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} test images.")
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=6, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    return train_loader, val_loader


def load_coco20_osr(dataroot, val_transforms):
    remap_labels = tf.Lambda(lambda x: torch.from_numpy(x).long().unsqueeze(0))

    val_set = COCO20(dataroot, split='val',
                     image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                     target_transform=tf.Compose(val_transforms['target'] + [remap_labels]),
                     joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} test images.")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    return val_loader


def load_coco20_ood(dataroot, val_transforms):
    def mpr(x):
        x = torch.from_numpy(x).long().unsqueeze(0)
        out = torch.ones_like(x).long() * 2
        out[x < 20] = 0
        out[x == 20] = 1
        return out

    remap_labels = tf.Lambda(mpr)
    val_set = COCO20(dataroot, split='val',
                     image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                     target_transform=tf.Compose(val_transforms['target'] + [remap_labels]),
                     joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} test images.")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    return val_loader
