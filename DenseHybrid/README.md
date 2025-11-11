# DenseHybrid

Official implementation of ECCV2022 paper **DenseHybrid: Hybrid Anomaly Detection for Dense Open-set Recognition** 
[[arXiv]](https://arxiv.org/pdf/2207.02606.pdf).

**Update April 2024:** The extended version of the paper is now publised in IEEE TPAMI with the title **Hybrid Open-set
Segmentation with Synthetic Negative Data** [[URL]](https://ieeexplore.ieee.org/document/10496197).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/densehybrid-hybrid-anomaly-detection-for/scene-segmentation-on-streethazards)](https://paperswithcode.com/sota/scene-segmentation-on-streethazards?p=densehybrid-hybrid-anomaly-detection-for)
 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/densehybrid-hybrid-anomaly-detection-for/anomaly-detection-on-fishyscapes-l-f)](https://paperswithcode.com/sota/anomaly-detection-on-fishyscapes-l-f?p=densehybrid-hybrid-anomaly-detection-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/densehybrid-hybrid-anomaly-detection-for/anomaly-detection-on-fishyscapes-1)](https://paperswithcode.com/sota/anomaly-detection-on-fishyscapes-1?p=densehybrid-hybrid-anomaly-detection-for)

![visitors](https://visitor-badge.laobi.icu/badge?page_id=matejgrcic.DenseHybrid)

### Abstract

Anomaly detection can be conceived either through generative modelling of regular training data or by discriminating
with respect to negative training data. These two approaches exhibit different failure modes. Consequently, hybrid
algorithms present an attractive research goal. Unfortunately, dense anomaly detection requires translational
equivariance and very large input resolutions. These requirements disqualify all previous hybrid approaches to the best
of our knowledge. We therefore design a novel hybrid algorithm based on reinterpreting discriminative logits as a
logarithm of the unnormalized joint distribution p*(x,y). Our model builds on a shared convolutional representation from
which we recover three dense predictions: i) the closed-set class posterior P(y|x), ii) the dataset posterior P(din|x),
iii) unnormalized data likelihood p*(x). The latter two predictions are trained both on the standard training data and
on a generic negative dataset. We blend these two predictions into a hybrid anomaly score which allows dense open-set
recognition on large natural images. We carefully design a custom loss for the data likelihood in order to avoid
backpropagation through the untractable normalizing constant Z(θ). Experiments evaluate our contributions on standard
dense anomaly detection benchmarks as well as in terms of open-mIoU - a novel metric for dense open-set performance. Our
submissions achieve state-of-the-art performance despite neglectable computational overhead over the standard semantic
segmentation baseline.

## Project setup

Create a new conda environment with the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

## Datasets

Cityscapes can be downloaded from [here](https://www.cityscapes-dataset.com/).

StreetHazards can be downloaded from [here](https://github.com/hendrycks/anomaly-seg).

COCO dataset is available at the offical [GitHub repo](https://github.com/nightrome/cocostuff).

Fishyscapes validation subsets with the appropriate
structure: [FS LAF](https://drive.google.com/file/d/1fwl8jn4NLAp0LShOEZHYNS4CKdyEAt4L/view?usp=sharing), [FS Static](https://drive.google.com/file/d/1iWuoA218HweS9uuaPZvD5SJ-R93cTBHo/view?usp=sharing).

ADE20k dataset (used as the negative examples) can be downloaded by
running `wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip`.

## Weights

ViT-B ImageNet: [weights](https://drive.google.com/file/d/1AlCpbDZhFiZ99teZKXu80OnskWagnvw4/view?usp=sharing)

DeepLabV3+ trained on Cityscapes by
NVIDIA: [weights](https://drive.google.com/file/d/1CKB7gpcPLgDLA7LuFJc46rYcNzF3aWzH/view?usp=sharing)

DeepLabV3+ fine-tuned with ADE20k negatives (Fishyscapes
benchmark): [weights](https://drive.google.com/file/d/1MZhINlNrXQlEyByUxypBebZQECqWAhlL/view?usp=sharing)

LDN-121 trained on
StreetHazards: [weights](https://drive.google.com/file/d/1Mf1sNVUhTtT1XexO-afco9577hzV5_kQ/view?usp=sharing)

LDN-121 fine-tuned on StreetHazards with ADE20k
negatives: [weights](https://drive.google.com/file/d/1vDXp-rySo-ASRh71O4h_MNiv_f-gFDm1/view?usp=sharing)

LDN-121 fine-tuned with ADE20k negatives (SMIYC
benchmark): [weights](https://drive.google.com/file/d/1bXzZ5B6YGqVKEKcFgcDbMHj5sHYSsjss/view?usp=sharing)

Segmenter trained on
COCO20: [weights](https://drive.google.com/file/d/18PXFz7uXRMk_iw50Nv-fIMwFMsEXaCdB/view?usp=sharing)

Segmenter fine-tuned on COCO20 with synthetic
negatives: [weights](https://drive.google.com/file/d/14xSfVBQmmbC8Yohn5gTG8zQCn8VgkSki/view?usp=sharing)

DenseFlow pretrained on traffic
scenes: [weights](https://drive.google.com/file/d/1_5vnfxNmC4X-kAbUVGlmFrfxLtYjZEog/view?usp=sharing)

DenseFlow pretrained on
COCO: [weights](https://drive.google.com/file/d/1Znc85uK30Z8keFrLepNpFeC6A8NlcM45/view?usp=sharing)

## Evaluation

### Dense anomaly detection

Fishyscapes LostAndFound val results:

```bash
python evaluate_ood.py --dataroot LF_DATAROOT --dataset lf --folder OUTPUT_DIR --params WEIGHTS_FILE
```

Fishyscapes Static val results:

```bash
python evaluate_ood.py --dataroot STATIC_DATAROOT --dataset static --folder OUTPUT_DIR --params WEIGHTS_FILE
```

StreetHazards results:

```bash
python evaluate_ood.py --dataroot SH_DATAROOT --dataset street-hazards --folder OUTPUT_DIR --params WEIGHTS_FILE
```

### Open-set Segmentation

StreetHazards:

```bash
python evaluate_osr_sh.py --dataroot SH_DATAROOT --model WEIGHTS_FILE
```

COCO20/80:

```bash
python evaluate_osr_coco.py --dataroot COCO_DATAROOT --model WEIGHTS_FILE
```

## Training

Fine-tune DeepLabV3+ on Cityscapes with real negatives:

```bash
python dlv3_cityscapes_finetune.py --dataroot CITY_DATAROOT --neg_dataroot ADE_DATAROOT --exp_name EXP_NAME
```

Fine-tune DeepLabV3+ on Cityscapes with synthetic negatives:

```bash
python dlv3_cityscapes_finetune_flow.py --dataroot CITY_DATAROOT --flow_state FLOW_CHECKPOINT --exp_name EXP_NAME
```

Train LDN-121 on StreetHazards:

```bash
python ldn_streethazards.py --dataroot SH_DATAROOT --exp_name EXP_NAME
```

Fine-tune LDN-121 on StreetHazards with real negatives:

```bash
python ldn_streethazards_finetune.py --dataroot SH_DATAROOT --neg_dataroot ADE_DATAROOT --exp_name EXP_NAME --model MODEL_INIT
```

Train Segmenter on COCO20 dataset:

```bash
python segmenter_coco20.py --dataroot COCO_DATAROOT --exp_name EXP_NAME
```

Fine-tune Segmenter with real negatives on COCO20 dataset:

```bash
python segmenter_coco20_finetune.py --dataroot COCO_DATAROOT --model TRAINED_MODEL --neg_dataset --exp_name EXP_NAME
```

Fine-tune Segmenter with synthetic negatives on COCO20 dataset:

```bash
python segmenter_coco20_finetune_flow.py --dataroot COCO_DATAROOT --model TRAINED_MODEL --flow_state TRAINED_FLOW --exp_name EXP_NAME
```

## Issues

If you encounter any issues with the code, please open an issue in this repository.

## Citation

If you find this code useful in your research, please consider citing the following papers:

```
@inproceedings{grcic22eccv,
  author    = {Matej Grcic and
               Petra Bevandic and
               Sinisa Segvic},
  title     = {DenseHybrid: Hybrid Anomaly Detection for Dense Open-Set Recognition},
  booktitle = {17th European Conference on Computer Vision {ECCV} 2022},
  publisher = {Springer},
  year      = {2022}
}

@article{grcic24tpami,
  author={Grcić, Matej and Šegvić, Siniša},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Hybrid Open-set Segmentation with Synthetic Negative Data}, 
  year={2024}
}
```
