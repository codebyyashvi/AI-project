from .cityscapes import load_cityscapes, load_cityscapes_uniform_loader
from .lost_found import load_lost_found
from .negatives import load_ade_negatives
from .street_hazards import load_street_hazards_ood, load_street_hazards_full, load_street_hazards_osr
from .fs_static import load_fs_static
from .lost_and_found import load_lost_and_found, load_lost_and_found_with_distance
from .coco import load_coco20_train_val, load_coco20_osr, load_coco20_ood

AVAILABLE_DATASETS = ['cityscapes', 'street-hazards-full', 'coco20']
AVAILABLE_EVAL_DATASETS = ['lf', 'static', 'street-hazards', 'coco20']
AVAILABLE_NEG_DATASETS = ['ade']


def get_dataset(dataset):
    if dataset == 'cityscapes':
        return load_cityscapes
    elif dataset == 'street-hazards-full':
        return load_street_hazards_full
    elif dataset == 'coco20':
        return load_coco20_train_val
    else:
        raise Exception('Invalid dataset!')


def get_eval_dataset(dataset):
    if dataset == 'lf':
        return load_lost_found
    elif dataset == 'street-hazards':
        return load_street_hazards_ood
    elif dataset == 'static':
        return load_fs_static
    else:
        raise Exception('Invalid eval dataset!')


def get_negative_dataset(dataset):
    if dataset == 'ade':
        return load_ade_negatives
    else:
        raise Exception('Invalid nagative dataset!')
