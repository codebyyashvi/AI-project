import random
import torch
import torch.nn.functional as F
import math
from .base import Experiment
from utils import IoU
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc
from torchvision.utils import save_image


class SemsegFlowNegativesCOCOExperiment(Experiment):

    def __init__(self, model, optimizer, loaders, epochs, logger, device, cp_file, args, flow, flow_optim, img_dir,
                 data_loader):
        super(SemsegFlowNegativesCOCOExperiment, self).__init__(model, optimizer, loaders, epochs, logger, device,
                                                                cp_file, args)

        self.flow = flow
        self.flow_optim = flow_optim
        self.img_dir = img_dir
        self.iter = 0

        # self.ood_loader = data_loader
        # self.outlier_iter = iter(data_loader)
        # print('>>> USING MASK')

    def _get_outlier_with_mask(self, shape):
        _, _, oh, ow = shape
        try:
            outlier, mask = next(self.outlier_iter)
        except:
            self.outlier_iter = iter(self.ood_loader)
            outlier, mask = next(self.outlier_iter)
        _, _, H, W = outlier.shape
        assert outlier.shape[2:] == mask.shape[2:]
        i = np.random.randint(H - oh)
        j = np.random.randint(W - ow)
        return outlier[:, :, i:i + oh, j:j + ow], mask[:, :, i:i + oh, j:j + ow]

    def _paste_patch_with_mask(self, x, label, ood_patch, ood_masks, ood_id):
        N, _, p_h, p_w = ood_patch.shape
        _, _, h, w = x.shape
        id_patch = torch.zeros_like(ood_patch)
        for i in range(x.size(0)):
            mask = ood_masks[i, 0]  # N 1 H W
            classes = torch.unique(mask)
            classes = classes[torch.randperm(classes.size(0))]
            classes = classes[:classes.size(0) // 3]
            zero_one_mask = torch.zeros_like(mask)
            for c in classes:
                zero_one_mask[mask == c] = 1

            pos_i = random.randint(0, h - p_h)
            pos_j = random.randint(0, w - p_w)
            id_patch[i] = x[i, :, pos_i: pos_i + p_h, pos_j: pos_j + p_w].detach()
            x[i, :, pos_i: pos_i + p_h, pos_j: pos_j + p_w] = ood_patch[i] * zero_one_mask + (1 - zero_one_mask) * x[i,
                                                                                                                   :,
                                                                                                                   pos_i: pos_i + p_h,
                                                                                                                   pos_j: pos_j + p_w]
            label[i, pos_i: pos_i + p_h, pos_j: pos_j + p_w][zero_one_mask == 1] = ood_id
        return x, id_patch, label

    def _paste_square_patch(self, x, label, ood_patch, ood_id, shapes=None):
        N, _, p_h, p_w = ood_patch.shape
        _, _, H, W = x.shape
        id_patch = torch.zeros_like(ood_patch)
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
            id_patch[i] = x[i, :, pos_i: pos_i + p_h, pos_j: pos_j + p_w].detach()
            x[i, :, pos_i: pos_i + p_h, pos_j: pos_j + p_w] = ood_patch[i]
            label[i, pos_i: pos_i + p_h, pos_j: pos_j + p_w] = ood_id
        return x, id_patch, label

    def sample_shape(self, bs):
        sizes = list(range(16, 105, 8))
        w = np.random.choice(sizes) // 8
        h = np.random.choice(sizes) // 8
        # w,h  = 297//8, 297//8
        return (bs, None, h, w)

    def get_batch_avg(self, logits, label_ood):
        N, _, H, W = logits.shape
        m = logits.mean(1).mean()
        ma = - m.view(1, 1, 1).repeat(N, H, W) * label_ood
        return ma

    def train(self):
        self.logger.log(f"Epoch {self.current_epoch}")
        self.model.train()
        self.flow.train()
        running_loss = 0.
        metrics = IoU(self.args.num_classes + 1, ignore_index=self.args.num_classes)
        with tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            for batch_idx, data in enumerate(self.train_loader, 1):
                x_, label_, orig_res = data
                x_ = x_.to(self.device)
                label_ = label_[:, 0].to(self.device)

                self.optimizer.zero_grad()
                num_patches = x_.size(0) * 3
                shape_ = self.sample_shape(num_patches)
                with torch.no_grad():
                    ood_patch = self.flow.sample(shape_) / 255.
                ood_patch = ood_patch.detach()
                num_img = x_.size(0)

                # paste three patches
                x, id_patch, label = self._paste_square_patch(x_.clone(), label_.clone(), ood_patch[:num_img],
                                                              self.args.num_classes + 1, orig_res)
                x, id_patch, label = self._paste_square_patch(x, label, ood_patch[num_img: 2 * num_img],
                                                              self.args.num_classes + 1, orig_res)
                x, id_patch, label = self._paste_square_patch(x, label, ood_patch[2 * num_img:],
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
                reg = self.get_batch_avg(logits, label_ood)

                loss_ood = (lse + reg.detach()).sum() / label_ood[label_ood == 1].numel()
                loss_seg = F.nll_loss(cls_out, label, ignore_index=self.args.num_classes)
                loss_dh = self.args.beta * loss_ood + self.args.beta * 10 * loss_th
                loss = loss_seg + loss_dh  # for beta 0.03
                loss.backward()
                self.optimizer.step()

                self.flow_optim.zero_grad()

                shape_ = self.sample_shape(x_.size(0))
                ood_patch = self.flow.sample(shape_) / 255.
                assert ood_patch.requires_grad
                x, id_patch, label = self._paste_square_patch(x_, label_.clone(), ood_patch, self.args.num_classes + 1,
                                                              orig_res)

                label_ood = torch.zeros_like(label)
                label_ood[label == self.args.num_classes + 1] = 1
                label_ood = label_ood.unsqueeze(1).repeat(1, self.args.num_classes, 1, 1)
                label[label == self.args.num_classes + 1] = self.args.num_classes
                logits, _ = self.model(x, label.size()[1:3])
                cls_out = F.log_softmax(logits, dim=1)
                uniform_dist = torch.ones_like(cls_out) * 1 / self.args.num_classes
                m = (cls_out.exp() + uniform_dist) / 2.
                kl_p_m = (F.kl_div(m.log(), cls_out, log_target=True, reduction='none') * label_ood).sum()
                kl_u_m = (F.kl_div(m.log(), uniform_dist, reduction='none') * label_ood).sum()
                loss_ood = (0.5 * kl_p_m + 0.5 * kl_u_m) / id_patch[:, 0].shape.numel()
                (loss_ood * 0.03).backward()

                # step towards ID
                id_patch = (id_patch * 255.).long()
                loss_mle = - self.flow.log_prob(id_patch).sum() / (math.log(2) * id_patch.shape.numel())
                loss_mle.backward()
                self.flow_optim.step()

                pred = cls_out.max(1)[1]
                metrics.add(pred, label)

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item(), loss_ood=loss_ood.item(), loss_seg=loss_seg.item(),
                                         loss_mle=loss_mle.item(), loss_dh=loss_dh.item())
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
        mle = 0.
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

                    ood_patch = self.flow.sample(self.sample_shape(x.size(0))) / 255.

                    x, id_patch, label = self._paste_square_patch(x, label, ood_patch, self.args.num_classes + 1)
                    label_ood = torch.zeros_like(label)
                    label_ood[label == self.args.num_classes + 1] = 1
                    label_ood[label == self.args.num_classes] = 2  # ignore ignored pixels

                    label[label == self.args.num_classes + 1] = self.args.num_classes

                    cls_out, ood_out = self.model(x, label.size()[1:3])
                    cls_prob = F.log_softmax(cls_out, dim=1)
                    loss = F.nll_loss(cls_prob, label, ignore_index=self.args.num_classes)
                    running_loss += loss.item()

                    if self.current_epoch % 5 == 0:
                        out = torch.nn.functional.softmax(ood_out, dim=1)
                        # reg = get_mc_neighbour(logit)
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

                    # save_image(x.cpu(), f"./debug_kl/im{batch_idx}.png")
                    # save_image(conf_probs.cpu(), f"./debug_kl/probs_{batch_idx}.png")
                    # save_image(colorize_cityscapes_labels(label.cpu()), f"./debug_kl/lb{batch_idx}.png")
                    # save_image(colorize_cityscapes_labels(label_ood.cpu()), f"./debug_kl/lo{batch_idx}.png")

                    pred = cls_prob.max(1)[1]
                    metrics.add(pred, label)
                    correct += pred.eq(label).cpu().sum().item()
                    total += label.size(0) * label.size(1) * label.size(2)
                    progress_bar.set_postfix(loss=loss.item())

                    id_patch = (id_patch * 255.).long()
                    mle += - (self.flow.log_prob(id_patch).sum() / (math.log(2) * id_patch.shape.numel())).cpu().item()
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
                self.logger.log(f"> Flow BPD: {round(mle / numel, 2)}")
        with torch.no_grad():
            samples = self.flow.sample(16)
            samples = samples / 255.
            self.iter += 1
            save_image(samples, f"{self.img_dir}/im_{self.iter}.png")
        torch.cuda.empty_cache()
        return miou

    def process_lr(self):
        if self.args.lr == self.args.lr_min:
            print(f"LR {self.args.lr, self.args.lr_min}")
            return
        lr = self.args.lr_min \
             + (self.args.lr - self.args.lr_min) * (1 + math.cos(self.current_epoch / self.epochs * math.pi)) / 2
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[0]['lr'] = lr / 4
            self.optimizer.param_groups[1]['lr'] = lr

        lrb = self.flow_optim.param_groups[0]['lr']
        coef = 0.975 if self.args.dataset != 'cityscapes' else 0.99
        self.flow_optim.param_groups[0]['lr'] = lrb * coef
        self.logger.log(f"LR set to ({lr},{lr / 4}) and {lrb}")

    def store_checkpoint(self):
        torch.save({
            'model': self.model.state_dict(),
            'flow': self.flow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_flow': self.flow_optim.state_dict(),
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
