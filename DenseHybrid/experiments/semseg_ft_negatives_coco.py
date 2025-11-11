import random
import torch
import torch.nn.functional as F
import math
from .base import Experiment
from utils import IoU
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc


class SemsegADENegativesCOCOExperiment(Experiment):

    def __init__(self, model, optimizer, loaders, epochs, logger, device, cp_file, args, img_dir, data_loader):
        super(SemsegADENegativesCOCOExperiment, self).__init__(model, optimizer, loaders, epochs, logger, device,
                                                               cp_file, args)

        self.img_dir = img_dir
        self.iter = 0

        self.ood_loader = data_loader
        self.outlier_iter = iter(data_loader)
        self.ood_classes_per_item = 2
        self.logger.log(f"> OOD classes per patch: {self.ood_classes_per_item}")
        self.rect_ood_prob = .5
        self.logger.log(f"> Rectangular patch prob: {round(self.rect_ood_prob * 100, 2)}%")
        self.delta = 0.9
        self.use_ma = True
        self.mov_avg = None
        self.logger.log(f"> Using moving average: {self.use_ma}, delta: {self.delta}")

    def sample_shape(self, max_dim):
        sizes = [i for i in range(16, max_dim, 8)]
        w = np.random.choice(sizes)
        h = np.random.choice(sizes)
        return (h, w)

    def _get_outlier_with_mask(self, batch_shape):
        # _, _, h, w = batch_shape
        try:
            outlier, ood_label = next(self.outlier_iter)
        except:
            self.outlier_iter = iter(self.ood_loader)
            outlier, ood_label = next(self.outlier_iter)

        assert outlier.size(0) >= batch_shape[0]
        mask = torch.zeros_like(ood_label)
        for i in range(ood_label.size(0)):
            if np.random.uniform() < self.rect_ood_prob:
                h, w = mask.shape[-2:]
                p_h, p_w = self.sample_shape(min(mask.shape[-1], mask.shape[-2]))
                pos_i = random.randint(0, h - p_h)
                pos_j = random.randint(0, w - p_w)
                mask[i, :, pos_i:pos_i + p_h, pos_j:pos_j + p_w] = 1
            else:
                values_, counts = ood_label[i].view(-1).unique(return_counts=True)
                values = values_[counts < 10000]  # remove classes with too many pixels e.g. background
                if len(values) == 0:
                    indices = [values_[counts.argmin()]]
                else:
                    indices = np.random.choice(values.numpy(), self.ood_classes_per_item)
                for idx in indices:
                    mask[i][ood_label[i] == idx] = 1
        return outlier, mask

    def _paste_anomaly(self, x, label, ood_patch, ood_lbl, ood_id, shapes=None):
        N, _, p_h, p_w = ood_patch.shape
        _, _, H, W = x.shape
        for i in range(x.size(0)):
            if shapes != None:
                h, w = shapes[0][i], shapes[1][i]
            else:
                h, w = 0, 0
            h = H if h - p_h < 0 else h
            w = W if w - p_w < 0 else w
            delta_h = (H - h) // 2 if h != H else 0
            delta_w = (W - w) // 2 if w != W else 0
            pos_i = random.randint(0, h - p_h) + delta_h
            pos_j = random.randint(0, w - p_w) + delta_w
            x[i, :, pos_i: pos_i + p_h, pos_j: pos_j + p_w] = x[i, :, pos_i: pos_i + p_h, pos_j: pos_j + p_w] * (
                        1 - ood_lbl[i]) \
                                                              + ood_lbl[i] * ood_patch[i]
            label[i, pos_i: pos_i + p_h, pos_j: pos_j + p_w][ood_lbl[i, 0] == 1] = ood_id
        return x, label

    def get_mc1(self, logits, label_ood):
        return - logits.mean(1) * label_ood

    def get_mc_neighbour(self, logits, label_ood):
        kh = kw = 7
        k = torch.ones(1, 1, kh, kw).cuda() / (kh * kw)
        mean = F.conv2d(logits.mean(1, keepdim=True), k, padding=(kh // 2, kw // 2), dilation=1)
        return - mean[:, 0] * label_ood

    def get_batch_avg(self, logits, label_ood):
        N, _, H, W = logits.shape
        m = logits.mean(1).mean()
        ma = - m.view(1, 1, 1).repeat(N, H, W) * label_ood
        return ma

    def get_avg(self, logits, label_ood):
        N, C, H, W = logits.shape
        logit_mean = torch.zeros(N).to(logits)
        for i in range(N):
            logit_mean[i] = (logits.mean(dim=1)[i][label_ood[i] == 1]).mean()
        # ma = - logit_mean.view(N,1,1).repeat(1,H,W) * label_ood
        ma = - logit_mean.mean().view(1, 1, 1).repeat(N, H, W) * label_ood
        return ma

    def train(self):
        self.logger.log(f"Epoch {self.current_epoch}")
        self.model.train()
        running_loss = 0.
        metrics = IoU(self.args.num_classes + 1, ignore_index=self.args.num_classes)
        with tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            for batch_idx, data in enumerate(self.train_loader, 1):
                x, label, orig_res = data
                x = x.to(self.device)
                label = label[:, 0].to(self.device)

                self.optimizer.zero_grad()

                anomaly_x, anomaly_lbl = self._get_outlier_with_mask(x.shape)
                x, label = self._paste_anomaly(x, label, anomaly_x.to(self.device), anomaly_lbl.to(self.device),
                                               self.args.num_classes + 1, orig_res)

                anomaly_x, anomaly_lbl = self._get_outlier_with_mask(x.shape)
                x, label = self._paste_anomaly(x, label, anomaly_x.to(self.device), anomaly_lbl.to(self.device),
                                               self.args.num_classes + 1, orig_res)

                anomaly_x, anomaly_lbl = self._get_outlier_with_mask(x.shape)
                x, label = self._paste_anomaly(x, label, anomaly_x.to(self.device), anomaly_lbl.to(self.device),
                                               self.args.num_classes + 1, orig_res)

                label_ood = torch.zeros_like(label)
                label_ood[label == self.args.num_classes + 1] = 1
                label_ood[label == self.args.num_classes] = 2  # ignore ignored pixels
                label[label == self.args.num_classes + 1] = self.args.num_classes
                logits, logits_ood = self.model(x, label.size()[1:3])
                cls_out = F.log_softmax(logits, dim=1)
                ood_out = F.log_softmax(logits_ood, dim=1)
                loss_th = F.nll_loss(ood_out, label_ood, ignore_index=2)
                label_ood = label_ood.clone()
                label_ood[label_ood == 2] = 0  # change back so that lse can be computed

                lse = torch.logsumexp(logits, 1) * label_ood
                if self.use_ma:
                    reg = self.get_batch_avg(logits, label_ood)
                else:
                    reg = - logits.mean(1) * label_ood
                loss_ood = (lse + reg.detach()).sum() / label_ood[label_ood == 1].numel()
                loss_seg = F.nll_loss(cls_out, label, ignore_index=self.args.num_classes)

                loss = loss_seg + self.args.beta * loss_ood + self.args.beta * 10 * loss_th  # for beta 0.03
                loss.backward()
                self.optimizer.step()

                pred = cls_out.max(1)[1]
                metrics.add(pred, label)

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item(), loss_ood=loss_ood.item(), loss_seg=loss_seg.item(),
                                         loss_th=loss_th.item())
                progress_bar.update(x.size(0))

        mean_loss = running_loss / batch_idx
        self.logger.log(f"> Average train loss: {round(mean_loss, 2)}")
        _, miou = metrics.value()
        self.logger.log(f"===> Train mIoU: {round(miou * 100., 2)}%")
        torch.cuda.empty_cache()
        if hasattr(self.train_loader.dataset, 'build_epoch'):
            self.train_loader.dataset.build_epoch()
            self.logger.log("> Builded new epoch!")

    def eval(self):
        running_loss = 0.
        self.model.eval()
        metrics = IoU(self.args.num_classes + 1, ignore_index=self.args.num_classes)
        correct = 0
        total = 0
        numel = 0
        total_AP = []
        total_fpr = []
        total_auroc = []
        with tqdm(total=len(self.val_loader.dataset)) as progress_bar:
            with torch.no_grad():
                for batch_idx, data in enumerate(self.val_loader, 1):
                    x, y = data
                    x = x.to(self.device)
                    y = y[:, 0]
                    label = y.to(self.device)

                    anomaly_x, anomaly_lbl = self._get_outlier_with_mask(x.shape)
                    x, label = self._paste_anomaly(x, label, anomaly_x.to(self.device), anomaly_lbl.to(self.device),
                                                   self.args.num_classes + 1)

                    label_ood = torch.zeros_like(label)
                    label_ood[label == self.args.num_classes + 1] = 1
                    label_ood[label == self.args.num_classes] = 2
                    label[label == self.args.num_classes + 1] = self.args.num_classes

                    cls_out, ood_out = self.model(x, label.size()[1:3])
                    cls_prob = F.log_softmax(cls_out, dim=1)
                    loss = F.nll_loss(cls_prob, label, ignore_index=self.args.num_classes)
                    running_loss += loss.item()

                    if self.current_epoch % 5 == 0:
                        out = torch.nn.functional.softmax(ood_out, dim=1)
                        reg = - (cls_out).mean(1)
                        p1 = torch.logsumexp(cls_out, dim=1)  # + reg # ln p(x, din)
                        p2 = out[:, 1]  # p(~din|x)
                        conf_probs = (- p1) + p2.log()  # - ln p(x, din) + ln p(~din|x)
                        item_ap = average_precision_score(label_ood[label_ood != 2].view(-1).cpu(),
                                                          conf_probs[label_ood != 2].view(-1).cpu())
                        total_AP.append(item_ap)
                        roc_auc, fpr = self.calculate_auroc(conf_probs[label_ood != 2].view(-1).cpu(),
                                                            label_ood[label_ood != 2].view(-1).cpu())
                        total_fpr.append(fpr)
                        total_auroc.append(roc_auc)

                    pred = cls_prob.max(1)[1]
                    metrics.add(pred, label)
                    correct += pred.eq(label).cpu().sum().item()
                    total += label.size(0) * label.size(1) * label.size(2)
                    progress_bar.set_postfix(loss=loss.item())
                    numel += 1

                    progress_bar.update(x.size(0))

                mean_loss = running_loss / batch_idx
                if self.current_epoch % 5 == 0:
                    ap = float(np.nanmean(total_AP)) * 100.
                    auroc = float(np.nanmean(total_auroc)) * 100.
                    fpr = float(np.nanmean(total_fpr)) * 100.
                    self.logger.log(f"> Average precision: {round(ap, 2)}%")
                    self.logger.log(f"> AUROC: {round(auroc, 2)}%")
                    self.logger.log(f"> FPR: {round(fpr, 2)}%")

                self.logger.log(f"> Average validation loss: {round(mean_loss, 2)}")
                self.logger.log(f"> Average validation accuracy: {round(correct * 100. / total, 2)}%")
                iou, miou = metrics.value()
                self.logger.log(f"> Validation mIoU: {round(miou * 100., 2)}")
        torch.cuda.empty_cache()
        return miou

    def process_lr(self):
        if self.args.lr == self.args.lr_min:
            self.logger.log(f"Skipping lr modification since {self.args.lr, self.args.lr_min}")
            return
        lr = self.args.lr_min \
             + (self.args.lr - self.args.lr_min) * (1 + math.cos(self.current_epoch / self.epochs * math.pi)) / 2
        self.optimizer.param_groups[0]['lr'] = lr / 4
        self.optimizer.param_groups[1]['lr'] = lr
        self.logger.log(f"LR set to ({lr},{lr / 4})")

    def store_checkpoint(self):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epochs': self.epochs,
            'current_epoch': self.current_epoch
        }, self.cp_file)

    def calculate_auroc(self, conf, gt):
        fpr, tpr, threshold = roc_curve(gt, conf)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        for i, j in zip(tpr, fpr):
            if i > 0.95:
                fpr_best = j
                break
        return roc_auc, fpr_best
