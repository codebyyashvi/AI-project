import os
import torch
import argparse
from utils import Logger
from data import get_dataset, JitterRandomCrop, RandomHorizontalFlip, AVAILABLE_DATASETS, get_negative_dataset, \
    RandomCrop
from data.joint_transforms.transforms import JointResize
import torchvision.transforms as tf
from models import construct_pascal_segmenter_th
from experiments import SemsegADENegativesCOCOExperiment
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('Semseg training')
parser.add_argument('--dataroot',
                    help='dataroot',
                    type=str,
                    default='.')
parser.add_argument('--dataset',
                    help='dataset',
                    type=str,
                    default='coco20',
                    choices=AVAILABLE_DATASETS)
parser.add_argument('--batch_size',
                    help='number of images in a mini-batch.',
                    type=int,
                    default=16)
parser.add_argument('--num_classes',
                    help='num classes of segmentator.',
                    type=int,
                    default=20)
parser.add_argument('--epochs',
                    help='maximum number of training epoches.',
                    type=int,
                    default=10)
parser.add_argument('--lr',
                    help='initial learning rate.',
                    type=float,
                    default=1e-6)
parser.add_argument('--lr_min',
                    help='min learning rate.',
                    type=float,
                    default=1e-6)
parser.add_argument('--momentum',
                    help='beta1 in Adam optimizer.',
                    type=float,
                    default=0.9)
parser.add_argument('--decay',
                    help='beta2 in Adam optimizer.',
                    type=float,
                    default=0.999)
parser.add_argument('--exp_name',
                    help='experiment name',
                    type=str,
                    required=True)
parser.add_argument('--resume',
                    help='Resume experiment',
                    action='store_true',
                    default=False)
parser.add_argument('--beta',
                    help='loss beta',
                    type=float,
                    default=0.15)
parser.add_argument('--neg_dataroot',
                    help='negative dataroot',
                    type=str,
                    default='.')
parser.add_argument('--model',
                    help='Pretrained model',
                    type=str,
                    required=True)
args = parser.parse_args()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = f"./logs/{args.dataset}/{args.exp_name}"
    if os.path.exists(exp_dir):
        raise Exception('Directory exists!')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/imgs", exist_ok=True)

    IMAGE_SIZE = 512
    CROP_SIZE = 512

    logger = Logger(f"{exp_dir}/log.txt")
    logger.log(str(args))

    train_transforms = {
        'image': [
            tf.ToTensor(),
        ],
        'target': [
        ],
        'joint': [
            RandomHorizontalFlip(),
            JitterRandomCrop(size=CROP_SIZE, scale=(0.5, 2.), ignore_id=args.num_classes, input_mean=(123, 116, 103),
                             return_shape=True),  # imagenet mean
        ]
    }

    val_transforms = {
        'image': [
            tf.ToTensor(),
            tf.Resize(IMAGE_SIZE),
            tf.CenterCrop(IMAGE_SIZE)
        ],
        'target': [
            tf.Lambda(lambda x: torch.from_numpy(np.array(x)).unsqueeze(0) / 255.0),
            tf.Resize(IMAGE_SIZE, Image.NEAREST),
            tf.CenterCrop(IMAGE_SIZE),
            tf.Lambda(lambda x: (x * 255).long()[0].numpy()),

        ],
        'joint': None
    }

    neg_transforms = {
        'image': [
            tf.ToTensor(),
        ],
        'target': [
            tf.ToTensor(),
        ],
        'joint': [
            JointResize(256),
            RandomCrop(256),
        ],
    }
    loaders = get_dataset(args.dataset)(args.dataroot, args.batch_size, train_transforms, val_transforms)
    neg_loader = get_negative_dataset('ade')(args.neg_dataroot, args.batch_size, neg_transforms)

    model = construct_pascal_segmenter_th(num_classes=args.num_classes, size='base').cuda()
    out = model.load_state_dict(torch.load(args.model), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    experiment = SemsegADENegativesCOCOExperiment(
        model, optimizer, loaders, args.epochs, logger, device, f"{exp_dir}/checkpoint.pt", args, f"{exp_dir}/imgs",
        neg_loader
    )
    experiment.eval()
    if args.resume:
        experiment.resume()
    else:
        experiment.start()
    logger.close()


if __name__ == '__main__':
    main(args)
