import os
import torch
import argparse
from utils import Logger
from data import get_dataset, JitterRandomCrop, RandomHorizontalFlip, AVAILABLE_DATASETS
import torchvision.transforms as tf
from models import DeepWV3PlusTH, DenseFlow
from experiments import SemsegFlowNegativesTrafficExperiment
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
                    default='cityscapes',
                    choices=AVAILABLE_DATASETS)
parser.add_argument('--batch_size',
                    help='number of images in a mini-batch.',
                    type=int,
                    default=16)
parser.add_argument('--num_classes',
                    help='num classes of segmentator.',
                    type=int,
                    default=19)
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
parser.add_argument('--exp_name',
                    help='experiment name',
                    type=str,
                    required=True)
parser.add_argument('--beta',
                    help='loss beta',
                    type=float,
                    default=0.01)
parser.add_argument('--flow_state',
                    help='flow state',
                    type=str)

args = parser.parse_args()


class Args:
    def __init__(self):
        self.last_block_pooling = 0


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = f"./logs/{args.dataset}/{args.exp_name}"
    if os.path.exists(exp_dir):
        raise Exception('Directory exists!')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/imgs", exist_ok=True)

    CROP_SIZE = 512

    logger = Logger(f"{exp_dir}/log.txt")
    logger.log(str(args))

    train_transforms = {
        'image': [
            tf.ToTensor(),
        ],
        'target': [
            tf.ToTensor(),
        ],
        'joint': [
            JitterRandomCrop(size=CROP_SIZE, scale=(0.5, 2), ignore_id=args.num_classes, input_mean=(73, 83, 72)),
            # city mean
            RandomHorizontalFlip()
        ]
    }

    val_transforms = {
        'image': [
            tf.ToTensor(),
        ],
        'target': [
            tf.ToTensor(),
        ],
        'joint': None
    }
    loaders = get_dataset(args.dataset)(args.dataroot, args.batch_size, train_transforms, val_transforms)

    mask_loader = None
    model = DeepWV3PlusTH(num_classes=args.num_classes).to(device)
    model.load_pretrained_weights_cv0()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-7)

    flow = DenseFlow(checkpointing=True).to(device)
    if args.flow_state:
        flow.load_state_dict(torch.load(args.flow_state)['model'])
        flow_optim = torch.optim.Adamax(flow.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-7)
    else:
        print('WARNING: Flow is randomly initialized!!!!')
        flow_optim = torch.optim.Adamax(flow.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    experiment = SemsegFlowNegativesTrafficExperiment(
        model, optimizer, loaders, args.epochs, logger, device, f"{exp_dir}/checkpoint.pt", args, flow, flow_optim,
        f"{exp_dir}/imgs", mask_loader
    )
    experiment.start()
    logger.close()


if __name__ == '__main__':
    main(args)
