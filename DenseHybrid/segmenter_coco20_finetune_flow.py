import os
import torch
import argparse
from utils import Logger
from data import get_dataset, JitterRandomCrop, RandomHorizontalFlip, AVAILABLE_DATASETS
import torchvision.transforms as tf
from models import DenseFlow, construct_pascal_segmenter_th
from experiments import SemsegFlowNegativesCOCOExperiment
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('Fine-tuning with negatives')
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
parser.add_argument('--exp_name',
                    help='experiment name',
                    type=str,
                    required=True)
parser.add_argument('--beta',
                    help='loss beta',
                    type=float,
                    default=0.15)
parser.add_argument('--flow_state',
                    help='flow state',
                    type=str,
                    required=True)
parser.add_argument('--model',
                    help='Pretrained model',
                    type=str,
                    required=True)
args = parser.parse_args()


def load_flow_params(path):
    state = torch.load(path)['model']
    new_state = dict()
    for k, v in state.items():
        new_state[k[len('flow.'):]] = v
    return new_state


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
        'target': [],
        'joint': [
            RandomHorizontalFlip(),
            JitterRandomCrop(size=CROP_SIZE, scale=(0.75, 2), ignore_id=args.num_classes, input_mean=(123, 116, 103),
                             return_shape=True)
        ]
    }

    val_transforms = {
        'image': [
            tf.ToTensor(),
            tf.Resize(IMAGE_SIZE)
        ],
        'target': [
            tf.Lambda(lambda x: torch.from_numpy(np.array(x)).unsqueeze(0) / 255.0),
            tf.Resize(IMAGE_SIZE, Image.NEAREST),
            tf.Lambda(lambda x: (x * 255).long()[0].numpy()),

        ],
        'joint': None
    }
    loaders = get_dataset(args.dataset)(args.dataroot, args.batch_size, train_transforms, val_transforms)

    model = construct_pascal_segmenter_th(num_classes=args.num_classes, size='base').cuda()
    out = model.load_state_dict(torch.load(args.model), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    flow = DenseFlow(checkpointing=True).to(device)
    if args.flow_state:
        flow.load_state_dict(load_flow_params(args.flow_state))
        flow_optim = torch.optim.Adamax(flow.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-7)
    else:
        print('WARNING: Flow is randomly initialized!!!!')
        flow_optim = torch.optim.Adamax(flow.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    experiment = SemsegFlowNegativesCOCOExperiment(
        model, optimizer, loaders, args.epochs, logger, device, f"{exp_dir}/checkpoint.pt", args, flow, flow_optim,
        f"{exp_dir}/imgs", None
    )
    experiment.start()
    logger.close()


if __name__ == '__main__':
    main(args)
