# Domain-Adaptive Object Detection via Uncertainty-Aware Distribution Alignment

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/domain-adaptive-object-detection-via-1/weakly-supervised-object-detection-on-1)](https://paperswithcode.com/sota/weakly-supervised-object-detection-on-1?p=domain-adaptive-object-detection-via-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/domain-adaptive-object-detection-via-1/weakly-supervised-object-detection-on-2)](https://paperswithcode.com/sota/weakly-supervised-object-detection-on-2?p=domain-adaptive-object-detection-via-1)


Multi-level Entropy Attention Alignment (MEAA) is an end-to-end approach for unsupervised domain adaptation of object detector. Specifically, MEAA consists of two main components: 

(1) Local Uncertainty Attentional Alignment (LUAA) module to accelerate the model better perceiving structure-invariant objects of interest by utilizing information theory to measure the uncertainty of each local region via the entropy of the pixel-wise domain classifier 

(2) Multi-level Uncertainty-Aware Context Alignment (MUCA) module to enrich domain-invariant information of relevant objects based on the entropy of multi-level domain classifiers

![Overall architecture design](https://github.com/basiclab/DA-OD-MEAA-PyTorch/blob/main/imgs/architecture.png)



## Setup Introduction
Follow [faster-rcnn repository](https://github.com/jwyang/faster-rcnn.pytorch)
 to setup the environment. When installing pytorch-faster-rcnn, you may encounter some issues.
Many issues have been reported there to setup the environment. We used Pytorch 0.4.1 for this project.
The different version of pytorch will cause some errors, which have to be handled based on each envirionment.

### Tested Hardwards & Softwares
- GTX 1080
- Pytorch 0.4.1
- CUDA 9.2
```
conda install pytorch=0.4.1 torchvision==0.2.1 cuda92 -c pytorch
```
- Before training:
```
mkdir data
cd lib
sh make.sh (add -gencode arch=compute_70,code=sm_70" # added for GTX10XX)
```

- Note to set number of classes = 20 in lib/datasets/water.py
- Tensorboard
`tensorboard --logdir='your/path/here'`


### Data Preparation

* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets.
* **Clipart, WaterColor**: Dataset preparation instruction link [Cross Domain Detection ](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets). 
* **Sim10k**: Website [Sim10k](https://fcav.engin.umich.edu/sim-dataset/)
* **CitysScape, FoggyCityscape**: Download website [Cityscape](https://www.cityscapes-dataset.com/), see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data)

All codes are written to fit for the format of PASCAL_VOC.
For example, the dataset [Sim10k](https://fcav.engin.umich.edu/sim-dataset/) is stored as follows.

```
$ cd Sim10k/VOC2012/
$ ls
Annotations  ImageSets  JPEGImages
$ cat ImageSets/Main/val.txt
3384827.jpg
3384828.jpg
3384829.jpg
.
.
.
```
If you want to test the code on your own dataset, arange the dataset
 in the format of PASCAL, make dataset class in lib/datasets/. and add
 it to  lib/datasets/factory.py, lib/datasets/config_dataset.py. Then, add the dataset option to lib/model/utils/parser_func.py.

### Data Path
Write your dataset directories' paths in lib/datasets/config_dataset.py.

### Pretrained Model

We used two models pre-trained on ImageNet in our experiments, VGG and ResNet101. You can download these two models from:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and write the path in __C.VGG_PATH and __C.RESNET_PATH at lib/model/utils/config.py.


## Train
- Cityscapes --> Foggy_cityscapes
```
python trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t foggy_cityscape
```
### use tensorboard
```
python trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t foggy_cityscape --use_tfb
```
--use_tfb will enable tensorboard to record training results

## Test
- Cityscapes --> Foggy_cityscapes
```
python test_net_MEAA.py --cuda --net vgg16 --dataset foggy_cityscape --load_name models/vgg16/cityscape/*.pth
```
Our trained model for foggy_cityscape:
https://drive.google.com/file/d/17pDu7mrxtx4cbpV2HNCGm2fzqCM1BZqd/view?usp=sharing

- Results:

![command line output results](https://github.com/basiclab/DA-OD-MEAA-PyTorch/blob/main/imgs/result.png)

## Reminder 
For training "pascasl_voc_0712 -> water"  results, since we only use 6 classes for evaluation.
We need to calculate results manually.
Just use chosen 6 classes to calculate mAP.

## Demo
This function is under construction now.
```
python demo_global.py --net vgg16 --load_name models/vgg16/cityscape/*.pth --cuda --dataset cityscape
```
## References

```
@inproceedings{10.1145/3394171.3413553,
author = {Nguyen, Dang-Khoa and Tseng, Wei-Lun and Shuai, Hong-Han},
title = {Domain-Adaptive Object Detection via Uncertainty-Aware Distribution Alignment},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394171.3413553},
doi = {10.1145/3394171.3413553},
}
```
