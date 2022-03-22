# Influence-balanced Loss for Imbalanced Visual Classification (ICCV, 2021)

This is the official implementation of [Influence-balanced Loss for Imbalanced Visual Classification](https://arxiv.org/abs/2110.02444) in PyTorch.
The code heavily relies on [LDAM-DRW](https://github.com/kaidic/LDAM-DRW).

[Paper](https://arxiv.org/abs/2110.02444) | [Bibtex](#Citation) | [Video](https://youtu.be/IScwCt_xYoY) | [Slides](https://www.slideshare.net/SeulkiPark10/iccv-21-influencebalanced-loss-for-imbalanced-visual-classification)


## Requirements

All codes are written by Python 3.7, and 'requirements.txt' contains required Python packages. 
To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

Create 'data/' directory and download original data in the directory to make imbalanced versions. 
- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The original data will be downloaded and converted by `imbalancec_cifar.py`.
- Imbalanced [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip). Download the data first, and convert them by `imbalance_tinyimagenet.py`.
- The paper also reports results on iNaturalist 2018. We will update the code for iNaturalist 2018 later.

## Training 

We provide several training examples:

### CIFAR
- CE baseline (CIFAR-100, long-tailed imabalance ratio of 100)

```bash
python cifar_train.py --dataset cifar100 --loss_type CE --train_rule None --imb_type exp --imb_factor 0.01 --epochs 200 --num_classes 100 --gpu 0
```
- IB (CIFAR-100, long-tailed imabalance ratio of 100)

```bash
python cifar_train.py --dataset cifar100 --loss_type IB --train_rule IBReweight --imb_type exp --imb_factor 0.01 --epochs 200 --num_classes 100 --start_ib_epoch 100 --gpu 0

```
- IB + CB (CIFAR-100, long-tailed imabalance ratio of 100)

```bash
python cifar_train.py --dataset cifar100 --loss_type IB --train_rule CBReweight --imb_type exp --imb_factor 0.01 --epochs 200 --num_classes 100 --start_ib_epoch 100 --gpu 0

```
- IB + Focal (CIFAR-100, long-tailed imabalance ratio of 100)

```bash
python cifar_train.py --dataset cifar100 --loss_type IBFocal --train_rule IBReweight --imb_type exp --imb_factor 0.01 --epochs 200 --num_classes 100 --start_ib_epoch 100 --gpu 0

```

### Tiny ImageNet
- CE baseline (long-tailed imabalance ratio of 100)

```bash
python tinyimage_train.py --dataset tinyimagenet -a resnet18 --loss_type CE --train_rule None --imb_type exp --imb_factor 0.01 --epochs 100 --lr 0.1  --num_classes 200

```
- IB (long-tailed imabalance ratio of 100)

```bash
python tinyimage_train.py --dataset tinyimagenet -a resnet18 --loss_type IB --train_rule IBReweight --imb_type exp --imb_factor 0.01 --epochs 100 --lr 0.1  --num_classes 200 --start_ib_epoch 50

```

## Citation

If you find our paper and repo useful, please cite our paper

```
@InProceedings{Park_2021_ICCV,
    author    = {Park, Seulki and Lim, Jongin and Jeon, Younghan and Choi, Jin Young},
    title     = {Influence-Balanced Loss for Imbalanced Visual Classification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {735-744}
}
```
