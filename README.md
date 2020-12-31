# Domain-Adaptive Object Detection via Uncertainty-Aware Distribution Alignment

## Introduction
Follow [faster-rcnn repository](https://github.com/jwyang/faster-rcnn.pytorch)
 to setup the environment. When installing pytorch-faster-rcnn, you may encounter some issues.
Many issues have been reported there to setup the environment. We used Pytorch 0.4.1 for this project.
The different version of pytorch will cause some errors, which have to be handled based on each envirionment.

### Tested Hardwards & Softwares
- GTX 1080
- Pytorch 0.4.1
- CUDA 9.2
```
conda install pytorch=0.4.1 cuda92 -c pytorch
```
- Before training:
```
mkdir data
cd lib
sh make.sh (add -gencode arch=compute_70,code=sm_70" # added for RTX20xx)
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

## Test
- Cityscapes --> Foggy_cityscapes
```
python test_net_MEAA.py --cuda --net vgg16 --dataset foggy_cityscape --load_name models/vgg16/cityscape/*.pth
```
Our trained model for foggy_cityscape:
https://drive.google.com/file/d/17pDu7mrxtx4cbpV2HNCGm2fzqCM1BZqd/view?usp=sharing

Results:
![alt text](https://github.com/basiclab/DA-OD-MEAA-PyTorch/blob/main/result.png)



## Demo
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
