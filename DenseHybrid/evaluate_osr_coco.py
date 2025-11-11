import torch
import numpy as np
from sklearn.metrics import average_precision_score, auc
from data import load_coco20_ood, load_coco20_osr, load_coco20_train_val
from models import construct_pascal_segmenter_th
import argparse
from utils import Logger, IoU, OpenIoU
import torchvision.transforms as tf
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_curve


class OpenModel(nn.Module):
    def __init__(self, model, threshold=None):
        super(OpenModel, self).__init__()
        self.model = model
        self.register_buffer('threshold', threshold)

    def set_threshold(self, t):
        self.register_buffer('threshold', t)

    def ood_score(self, img, shape=None):
        logit, logit_ood = self.model(img, shape)
        out = F.softmax(logit_ood, dim=1)
        p1 = torch.logsumexp(logit, dim=1)
        p2 = out[:, 1]  # p(~din|x)
        conf_probs = (- p1) + p2.log()
        return conf_probs

    def forward(self, img, shape=None):
        assert self.threshold != None
        logit, logit_ood = self.model(img, shape)
        out = F.softmax(logit_ood, dim=1)
        p1 = torch.logsumexp(logit, dim=1)
        p2 = out[:, 1]  # p(~din|x)
        conf_probs = (- p1) + p2.log()  # - ln hat_p(x, din) + ln p(~din|x)
        classes = logit.max(1)[1].clone()
        classes[conf_probs > self.threshold] = logit.size(1)

        return classes


class OODCalibration:
    def __init__(self, model, loader, device, ignore_id, logger):
        self.model = model
        self.loader = loader
        self.device = device
        self.ignore_id = ignore_id
        self.logger = logger

    def calculate_stats(self, conf, gt, rate=0.95):
        fpr, tpr, threshold = roc_curve(gt, conf)
        fpr = np.array(fpr)
        tpr = np.array(tpr)
        threshold = np.array(threshold)
        roc_auc = auc(fpr, tpr)
        fpr_best = fpr[tpr >= rate][0]
        treshold = threshold[tpr >= rate][0]
        return roc_auc, fpr_best, treshold

    def calculate_ood_scores(self, desired_tpr=0.95, scale=1.):
        total_conf = []
        total_gen = []
        total_disc = []
        total_gt = []

        for step, batch in enumerate(self.loader):
            img, lbl = batch
            img = img.to(self.device)
            lbl = lbl[:, 0]
            lbl = lbl.to(self.device)
            with torch.no_grad():
                conf_probs = self.model.ood_score(img, lbl.shape[1:])
            if scale != 1.:
                conf_probs = F.interpolate(conf_probs.unsqueeze(1), scale_factor=scale, mode='bilinear')[:, 0]
                lbl = F.interpolate(lbl.unsqueeze(1).float(), scale_factor=scale, mode='nearest')[:, 0].long()

            label = lbl.view(-1)
            conf_probs = conf_probs.view(-1)
            gt = label[label != 2].cpu()
            total_gt.append(gt)
            conf = conf_probs.cpu()[label.cpu() != 2]
            total_conf.append(conf)

        total_gt = torch.cat(total_gt, dim=0).numpy()
        total_conf = torch.cat(total_conf, dim=0).numpy()

        AP = average_precision_score(total_gt, total_conf)
        roc_auc, fpr, treshold = self.calculate_stats(total_conf, total_gt, rate=desired_tpr)
        self.logger.log(f"> Average precision: {round(AP * 100., 2)}%")
        self.logger.log(f"> FPR: {round(fpr * 100., 2)}%")
        self.logger.log(f"> AUROC: {round(roc_auc * 100., 2)}%")
        self.logger.log(f"> Treshold: {round(treshold, 2)}")

        return treshold


def evaluate_open_dataset(loader, model, num_classes):
    metrics = OpenIoU(num_classes + 2, ignore_index=num_classes + 1)
    for i, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()[:, 0]
        with torch.no_grad():
            preds = model(x, y.shape[1:])
        metrics.add(preds, y)

    iou = metrics.iou_value()
    iou = np.nan_to_num(iou, copy=False, nan=0.)
    print(f"OPEN SET: open-mIoU {np.nanmean(iou[:-2]) * 100.}")
    print(f"OPEN SET: mIoU over K+1 {np.nanmean(iou[:-1]) * 100.}")


def evaluate_closed_dataset(loader, model, num_classes):
    metrics = IoU(num_classes + 1, ignore_index=num_classes)
    for i, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()[:, 0]
        with torch.no_grad():
            logits, _ = model.forward(x, y.shape[1:])
        preds = logits.max(1)[1]
        y[y >= num_classes] = num_classes
        metrics.add(preds, y)

    iou, miou = metrics.value()
    print(f"CLOSED SET: mIoU over {num_classes} classes {miou * 100.} %")
    print(f"Per class IoUs: {[iu * 100 for iu in iou[:-1]]}")


def compute_osr_perf(loader, loader_anom, model, num_classes, desired_tpr=0.95, scale=0.5):
    # evaluate_closed_dataset(loader, model)

    model = OpenModel(model).cuda()
    model.eval()

    calibrator = OODCalibration(model, loader_anom, 'cuda', ignore_id=2, logger=logger)
    treshold = calibrator.calculate_ood_scores(desired_tpr=desired_tpr, scale=scale)
    model.set_threshold(torch.tensor(treshold))

    evaluate_open_dataset(loader, model.eval(), num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Calibrate OSR model')
    parser.add_argument('--dataroot',
                        help='dataroot',
                        type=str,
                        default='.')
    parser.add_argument('--num_classes',
                        help='num classes of segmentator.',
                        type=int,
                        default=20)
    parser.add_argument('--tpr',
                        help='desired tpr',
                        type=float,
                        default=0.95)
    parser.add_argument('--model',
                        help='cp file',
                        type=str,
                        required=True)
    args = parser.parse_args()

    model = construct_pascal_segmenter_th(size='base', num_classes=args.num_classes).cuda()
    out = model.load_state_dict(torch.load(args.model), strict=True)
    model.eval()

    exp_dir = '/'.join(args.model.split('/')[:-1])
    logger = Logger(f"{exp_dir}/log_eval.txt")
    logger.log(str(args))

    val_transforms = {
        'image': [tf.ToTensor()],
        'target': [],
        'joint': None
    }

    loader_osr = load_coco20_osr(args.dataroot, val_transforms)
    loader_ood = load_coco20_ood(args.dataroot, val_transforms)
    loader_closed = load_coco20_train_val(args.dataroot, 1, val_transforms, val_transforms)[1]
    assert args.num_classes == 20

    print('>>> Closed-set performance')
    # evaluate_closed_dataset(loader_closed, model, args.num_classes)

    print('>>> Open-set performance')
    compute_osr_perf(loader_osr, loader_ood, model, num_classes=args.num_classes, desired_tpr=args.tpr, scale=1)
