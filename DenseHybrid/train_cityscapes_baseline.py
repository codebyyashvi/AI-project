import os
import torch
import argparse
from utils import Logger
from data import get_dataset, JitterRandomCrop, RandomHorizontalFlip
import torchvision.transforms as tf
from models import DeepWV3Plus
from torch.nn import CrossEntropyLoss
import warnings
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Cityscapes DeeplabV3+ Baseline (CPU Safe)")
parser.add_argument('--dataroot', type=str, default='.', help='Cityscapes root')
parser.add_argument('--batch_size', type=int, default=2)   # SMALL for CPU
parser.add_argument('--num_classes', type=int, default=19)
parser.add_argument('--epochs', type=int, default=5)       # start small
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--exp_name', type=str, required=True)
args = parser.parse_args()


def main(args):
    # Force CPU always
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("🚀 Training on:", device)

    # experiment folder
    exp_dir = f"./logs/{args.exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/imgs", exist_ok=True)

    logger = Logger(f"{exp_dir}/log.txt")
    logger.log(str(args))

    CROP_SIZE = 256   # SMALL CROP FOR CPU (512 too heavy)

    # -----------------------------
    # Data transforms
    # -----------------------------
    train_transforms = {
        "image": [tf.ToTensor()],
        "target": [tf.ToTensor()],
        "joint": [
            JitterRandomCrop(
                size=CROP_SIZE,
                scale=(0.5, 1.2),
                ignore_id=args.num_classes,
                input_mean=(73, 83, 72)
            ),
            RandomHorizontalFlip()
        ]
    }

    val_transforms = {
        "image": [tf.ToTensor()],
        "target": [tf.ToTensor()],
        "joint": None
    }

    # Load Cityscapes
    train_loader, val_loader = get_dataset("cityscapes")(
        args.dataroot,
        args.batch_size,
        train_transforms,
        val_transforms
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = DeepWV3Plus(num_classes=args.num_classes)

    # IMPORTANT: do NOT use .cuda() inside model file
    # You MUST remove .cuda() in DeepWV3Plus __init__ manually!

    model.to(device)

    try:
        model.load_pretrained_weights_cv0()
        print("Loaded pretrained backbone.")
    except:
        print("Warning: Could not load pretrained weights.")

    optimizer = torch.optim.Adam(
        [{'params': model.parameters(), 'lr': args.lr}],
        betas=(0.9, 0.999),
        eps=1e-7
    )

    criterion = CrossEntropyLoss(ignore_index=args.num_classes)

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for img, lbl in train_loader:
            pass
            img, lbl = img.to(device), lbl.squeeze(1).long().to(device)

            optimizer.zero_grad()
            pred = model(img)

            loss = criterion(pred, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss:.4f}")
        logger.log(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), f"{exp_dir}/checkpoint_epoch{epoch+1}.pth")

    logger.close()
    print("Training completed successfully.")


if __name__ == "__main__":
    main(args)
