# taken from https://github.com/rstrudel/segmenter
from .config import load_config
from .factory import create_segmenter, create_segmenter_th
from .utils import inference, inference_th, checkpoint_filter_fn
import torch.nn as nn
import torch


class SegmenterWrapper(nn.Module):
    def __init__(self, model, cfg, imagenet_mean=False):
        super().__init__()
        self.model = model
        self.cfg = cfg

        if imagenet_mean:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        else:
            self.mean = torch.tensor([0.2869, 0.3251, 0.2839]).view(1, 3, 1, 1).cuda()
            self.std = torch.tensor([0.1761, 0.1810, 0.1777]).view(1, 3, 1, 1).cuda()

    def forward(self, x, target_shape=None):

        x = (x - self.mean) / self.std

        if self.training:
            out = self.model(x)
            return out
        else:
            assert x.shape[0] == 1
            all_logits = []
            x_ = x
            orig_h, orig_w = x.shape[-2:]
            # scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
            scales = [1.]
            for sf in scales:
                x = torch.nn.functional.interpolate(x_, scale_factor=sf, mode='bilinear', align_corners=False)
                ims = [x]
                im_meta = dict(flip=False)
                logits = inference(
                    self.model,
                    ims,
                    [im_meta],
                    ori_shape=x.shape[2:4],
                    window_size=self.cfg["inference_kwargs"]["window_size"],
                    window_stride=self.cfg["inference_kwargs"]["window_stride"],
                    batch_size=2,
                )
                logits = torch.nn.functional.interpolate(logits.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear',
                                                         align_corners=False)
                all_logits.append(logits)
            logits = sum(all_logits) / len(all_logits)
            return logits


class SegmenterTHWrapper(nn.Module):
    def __init__(self, model, cfg, imagenet_mean=False):
        super().__init__()
        self.model = model
        self.cfg = cfg

        if imagenet_mean:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        else:
            self.mean = torch.tensor([0.2869, 0.3251, 0.2839]).view(1, 3, 1, 1).cuda()
            self.std = torch.tensor([0.1761, 0.1810, 0.1777]).view(1, 3, 1, 1).cuda()

    def forward(self, x, target_shape=None):

        x = (x - self.mean) / self.std

        if self.training:
            out = self.model(x)
            return out
        else:
            assert x.shape[0] == 1 and len(x.shape) == 4
            all_logits = []
            all_ood_logits = []
            x_ = x
            orig_h, orig_w = x.shape[-2:]
            # scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
            # scales = [1, 1.5, 1.75, 2.]
            scales = [1.]
            for sf in scales:
                x = torch.nn.functional.interpolate(x_, scale_factor=sf, mode='bilinear', align_corners=False)
                # ims = [x[i].unsqueeze(0) for i in range(len(x))]
                ims = [x]
                im_meta = dict(flip=False)
                logits, logits_ood = inference_th(
                    self.model,
                    ims,
                    [im_meta],
                    ori_shape=x.shape[2:4],
                    window_size=self.cfg["inference_kwargs"]["window_size"],
                    window_stride=self.cfg["inference_kwargs"]["window_stride"],
                    batch_size=2,
                )
                logits = torch.nn.functional.interpolate(logits.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear',
                                                         align_corners=False)
                logits_ood = torch.nn.functional.interpolate(logits_ood.unsqueeze(0), size=(orig_h, orig_w),
                                                             mode='bilinear', align_corners=False)
                all_logits.append(logits)
                all_ood_logits.append(logits_ood)
            logits = sum(all_logits) / len(all_logits)
            logits_ood = sum(all_ood_logits) / len(all_ood_logits)
            return logits, logits_ood


def construct_pascal_segmenter(size='base', num_classes=20):
    cfg = load_config('models/segmenter/pascal.yaml')
    cfg['net_kwargs']['n_cls'] = num_classes

    if size == 'base':
        cfg['net_kwargs']['backbone'] = 'vit_base_patch16_384'
        cfg['net_kwargs']['d_model'] = 768
        cfg['net_kwargs']['n_heads'] = 16  # used for upsampling only
        cfg['net_kwargs']['n_layers'] = 12
    elif size == 'large':
        cfg['net_kwargs']['backbone'] = 'vit_large_patch16_384'
        cfg['net_kwargs']['d_model'] = 1024
        cfg['net_kwargs']['n_heads'] = 16
        cfg['net_kwargs']['n_layers'] = 24
    else:
        raise ValueError

    model = create_segmenter(cfg['net_kwargs'])
    if size == 'large':
        state = torch.load('vit_large_patch16_384_imagenet.cp')
    elif size == 'base':
        state = torch.load('vit_base_p16_384.pth')
    else:
        raise ValueError

    state = checkpoint_filter_fn(state, model.encoder)
    out = model.encoder.load_state_dict(state)
    print(out)
    return SegmenterWrapper(model, cfg, imagenet_mean=True)


def construct_pascal_segmenter_th(size='base', num_classes=20):
    cfg = load_config('models/segmenter/pascal.yaml')
    cfg['net_kwargs']['n_cls'] = num_classes

    if size == 'base':
        cfg['net_kwargs']['backbone'] = 'vit_base_patch16_384'
        cfg['net_kwargs']['d_model'] = 768
        cfg['net_kwargs']['n_heads'] = 16  # used for upsampling only
        cfg['net_kwargs']['n_layers'] = 12
    elif size == 'large':
        cfg['net_kwargs']['backbone'] = 'vit_large_patch16_384'
        cfg['net_kwargs']['d_model'] = 1024
        cfg['net_kwargs']['n_heads'] = 16
        cfg['net_kwargs']['n_layers'] = 24
    else:
        raise ValueError

    model = create_segmenter_th(cfg['net_kwargs'])
    if size == 'large':
        state = torch.load('vit_large_patch16_384_imagenet.cp')
    elif size == 'base':
        state = torch.load('vit_base_p16_384.pth')
    else:
        raise ValueError

    state = checkpoint_filter_fn(state, model.encoder)
    model.encoder.load_state_dict(state)
    return SegmenterTHWrapper(model, cfg, imagenet_mean=True)
