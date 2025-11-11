import os
import torch
import argparse
from utils import Logger
from data import get_dataset, RandomHorizontalFlip, JitterRandomCrop
import torchvision.transforms as tf
from models import construct_pascal_segmenter
from experiments import SemsegExperiment
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
                    default='coco20')
parser.add_argument('--batch_size',
                    help='number of images in a mini-batch.',
                    type=int,
                    default=16)
parser.add_argument('--num_classes',
                    help='num classes',
                    type=int,
                    default=20)
parser.add_argument('--epochs',
                    help='maximum number of training epoches.',
                    type=int,
                    default=80)
parser.add_argument('--lr',
                    help='initial learning rate.',
                    type=float,
                    default=1e-3)
parser.add_argument('--exp_name',
                    help='experiment name',
                    type=str,
                    required=True)
args = parser.parse_args()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = f"./logs/{args.dataset}/{args.exp_name}"
    if os.path.exists(exp_dir):
        raise Exception('Directory exists!')
    os.makedirs(exp_dir, exist_ok=True)

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
            JitterRandomCrop(size=CROP_SIZE, scale=(0.5, 2), ignore_id=args.num_classes, input_mean=(123, 116, 103)),
            RandomHorizontalFlip()
        ]
    }

    val_transforms = {
        'image': [
            tf.ToTensor(),
        ],
        'target': [
        ],
        'joint': None
    }
    loaders = get_dataset(args.dataset)(args.dataroot, args.batch_size, train_transforms, val_transforms)

    model = construct_pascal_segmenter(num_classes=args.num_classes, size='base').cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    experiment = SemsegExperiment(
        model, optimizer, loaders, args.epochs, logger, device, f"{exp_dir}/checkpoint.pt", args)

    experiment.start()
    logger.close()


if __name__ == '__main__':
    main(args)
