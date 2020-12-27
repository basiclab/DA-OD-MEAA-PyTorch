from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg

from model.faster_rcnn.faster_rcnn_MEAA  import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)
class netD_forward1(nn.Module):
    def __init__(self):
        super(netD_forward1, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                  padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
        return feat

class netD_forward2(nn.Module):
    def __init__(self):
        super(netD_forward2, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1,
                  padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
        return feat

class netD_forward3(nn.Module):
    def __init__(self):
        super(netD_forward3, self).__init__()
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1,
                  padding=1, bias=False)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
        return feat


class netD_inst(nn.Module):
  def __init__(self, fc_size=2048):
    super(netD_inst, self).__init__()
    self.fc_1_inst = nn.Linear(fc_size, 512)
    self.fc_2_inst = nn.Linear(512, 128)
    self.fc_3_inst = nn.Linear(128, 2)
    self.relu = nn.ReLU(inplace=True)
    #self.softmax = nn.Softmax()
    #self.logsoftmax = nn.LogSoftmax()
    # self.bn = nn.BatchNorm1d(128)
    self.bn2 = nn.BatchNorm1d(2)

  def forward(self, x):
    x = self.relu(self.fc_1_inst(x))
    x = self.relu((self.fc_2_inst(x)))
    x = self.relu(self.bn2(self.fc_3_inst(x)))
    return x

class netD1(nn.Module):
    def __init__(self,context=False):
        super(netD1, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context

        self.fc256 = nn.Linear(256, 1, bias=False) # att for base_feat1
        self.fc = nn.Linear(128, 1, bias=False) # att
        # Class Activation Map
        self.gap_fc = nn.Linear(128, 1, bias=False)
        self.gmp_fc = nn.Linear(128, 1, bias=False)

        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))

        #---
        # up

        # down
        # yy = x.view(-1, 256)
        # yy = self.fc256(yy)
        # yy_w = list(self.fc256.parameters())[0]
        # yy_w = yy_w.unsqueeze(2).unsqueeze(3) # atten 256 weight for base_feat1
        #---

        # self-attention
        # x = x * yy_w
        x = self.conv2(x)
        # print('\nx bf cat: ', x.shape) # 1x128xhxw
        #---

        # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        # gap_weight = list(self.gap_fc.parameters())[0]
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        # # print('\ngap: ', gap.shape)
        # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        # gmp_weight = list(self.gmp_fc.parameters())[0]
        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        # # print('\ngmp: ', gmp.shape)

        # cam_logit = torch.cat([gap_logit, gmp_logit], 1)

        # x = torch.cat([gap, gmp], 1) # 1x128xhxw + 1x128xhxw
        # # print('\nx af cat: ', x.shape)
        # # print('\ncam_logit: ', cam_logit.shape)
        # # x = self.leaky_relu(self.conv1x1(x))

        # # heatmap = torch.sum(x, dim=1, keepdim=True)
        # x = self.conv2(x)
        #---
        x = F.relu(x)
        # print('\nx relu: ', x.shape)
        # heatmap = torch.sum(x, dim=1, keepdim=True)

        # xx = x.view(-1, 128)
        # xx = self.fc(xx)
        # xx_w = list(self.fc.parameters())[0]
        # xx_w = xx_w.unsqueeze(2).unsqueeze(3) # attention 128 weight for feat1

        # print('\nxx_w: ', xx_w.shape)

        # self-attention
        # x = x * xx_w
        
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          x = self.conv3(x)
          return F.sigmoid(x),feat
        else:
          x = self.conv3(x)
          # return F.sigmoid(x), xx_w, yy_w
          return F.sigmoid(x)#, cam_logit, heatmap


class netD12(nn.Module):
    def __init__(self,context=False):
        super(netD12, self).__init__()
        self.conv0 = nn.Conv2d(512, 512, kernel_size=1, stride=1,
          padding=0, bias=False)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context

        self.fc256 = nn.Linear(256, 1, bias=False) # att for base_feat1
        self.fc = nn.Linear(128, 1, bias=False) # att
        
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))

        #--- get Attention map---
        # up

        # down

        # gap, gmp
        yy = x.view(-1, 256)
        yy = self.fc256(yy)
        yy_w = list(self.fc256.parameters())[0]
        yy_w = yy_w.unsqueeze(2).unsqueeze(3) # atten 256 weight for base_feat1
        #---

        # self-attention
        # x = x * yy_w

        x = F.relu(self.conv2(x))

        # xx = x.view(-1, 128)
        # xx = self.fc(xx)
        # xx_w = list(self.fc.parameters())[0]
        # xx_w = xx_w.unsqueeze(2).unsqueeze(3) # attention 128 weight for feat1

        # print('\nxx_w: ', xx_w.shape)

        # self-attention
        # x = x * xx_w
        
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          x = self.conv3(x)
          return F.sigmoid(x),feat
        else:
          x = self.conv3(x)
          # return F.sigmoid(x), xx_w, yy_w
          return F.sigmoid(x), yy_w
 
class netD13(nn.Module):
    def __init__(self,context=False):
        super(netD13, self).__init__()


        self.conv0 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1,
          padding=0, bias=False)
        self.conv01 = nn.Conv2d(1024, 512, kernel_size=1, stride=1,
          padding=0, bias=False)

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context

        self.fc256 = nn.Linear(256, 1, bias=False) # att for base_feat1
        self.fc = nn.Linear(128, 1, bias=False) # att
        
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv01(x))
        x = F.relu(self.conv1(x))

        #--- Attention map ---
        # up
        # down
        # gap, gmp

        yy = x.view(-1, 256)
        yy = self.fc256(yy)
        yy_w = list(self.fc256.parameters())[0]
        yy_w = yy_w.unsqueeze(2).unsqueeze(3) # atten 256 weight for base_feat1
        #---

        # self-attention
        # x = x * yy_w

        x = F.relu(self.conv2(x))

        # xx = x.view(-1, 128)
        # xx = self.fc(xx)
        # xx_w = list(self.fc.parameters())[0]
        # xx_w = xx_w.unsqueeze(2).unsqueeze(3) # attention 128 weight for feat1

        # print('\nxx_w: ', xx_w.shape)

        # self-attention
        # x = x * xx_w
        
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          x = self.conv3(x)
          return F.sigmoid(x),feat
        else:
          x = self.conv3(x)
          # return F.sigmoid(x), xx_w, yy_w
          return F.sigmoid(x), yy_w

class netD21(nn.Module):
    def __init__(self,context=False):
        super(netD21, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
          padding=0, bias=False)

        # self.conv1 = conv3x3(256, 256, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv3x3(256, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        #-----------------
        self.fc512= nn.Linear(512, 1, bias=False) # att for base_feat1
        self.fc_att = nn.Linear(128, 1, bias=False) # att
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        # atten 512
        # print('\nx:', x.shape)


        # up
        # down
        # gap, gmp
        xx = x.view(-1, 512)
        # print('\nxx:', xx.shape)

        xx = self.fc512(xx)
        xx_w = list(self.fc512.parameters())[0]
        xx_w = xx_w.unsqueeze(2).unsqueeze(3) # atten 512 weight for base_feat1

        # self-attention
        # print('\nx: ', x.shape)
        # print('\nxx_w:', xx_w.shape)
        # x = x * xx_w

        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)

        # atten 128
        # xx2 = self.fc_att(x)
        # xx2_w = list(self.fc_att.parameters())[0]
        #         # self-attention
        # # print('\nx bf: ', x.shape)
        # # print('\nxx2_w: ', xx2_w.shape)
        # x = x * xx2_w
        # # print('\nx at: ', x.shape)
        # xx2_w = xx2_w.unsqueeze(2).unsqueeze(3)


        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          # return x, xx2_w, xx_w
          # return x, xx2_w, xx_w
          return x, xx_w


class netD2(nn.Module):
    def __init__(self,context=False):
        super(netD2, self).__init__()
        self.conv0 = nn.Conv2d(512, 512, kernel_size=1, stride=1,
          padding=0, bias=False)
        self.conv1 = conv3x3(512, 256, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv3x3(256, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        #-----------------
        self.fc512= nn.Linear(512, 1, bias=False) # att for base_feat1
        self.fc_att = nn.Linear(128, 1, bias=False) # att
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        #---
        self.conv4 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
          padding=0, bias=False)
        self.gap_fc = nn.Linear(128, 1, bias=False)
        self.gmp_fc = nn.Linear(128, 1, bias=False)

        #---

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        # atten 512
        # print('\nx:', x.shape)

        # xx = x.view(-1, 512)
        # # print('\nxx:', xx.shape)

        # xx = self.fc512(xx)
        # xx_w = list(self.fc512.parameters())[0]
        # xx_w = xx_w.unsqueeze(2).unsqueeze(3) # atten 512 weight for base_feat1

        # self-attention
        # print('\nx: ', x.shape)
        # print('\nxx_w:', xx_w.shape)
        # x = x * xx_w

        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)


        #---
        #---
        # x = self.conv3(x) # 1x128xhxw
        # print('\nx bf cam: ', x.shape)

        # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        # gap_weight = list(self.gap_fc.parameters())[0]
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        # # print('\ngap: ', gap.shape)
        # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        # gmp_weight = list(self.gmp_fc.parameters())[0]
        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        # # print('\ngmp: ', gmp.shape)

        # cam_logit = torch.cat([gap_logit, gmp_logit], 1)

        # x = torch.cat([gap, gmp], 1) # 1x128xhxw + 1x128xhxw

        # x = self.conv4(x)
        # # print('\nx af cam: ', x.shape)
        # # print('\ncam_logit: ', cam_logit.shape)

        #---
        # x = F.dropout(F.relu(self.bn3(x)),training=self.training)

        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)


        # atten 128
        # xx2 = self.fc_att(x)
        # xx2_w = list(self.fc_att.parameters())[0]
        #         # self-attention
        # # print('\nx bf: ', x.shape)
        # # print('\nxx2_w: ', xx2_w.shape)
        # x = x * xx2_w
        # # print('\nx at: ', x.shape)
        # xx2_w = xx2_w.unsqueeze(2).unsqueeze(3)


        if self.context:
          feat = x

        x = self.fc(x)

        if self.context:
          return x,feat
        else:
          # return x, xx2_w, xx_w
          return x#, cam_logit


class netD3(nn.Module):
    def __init__(self,context=False):
        super(netD3, self).__init__()
        self.conv0 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1,
          padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(1024)

        self.conv1 = conv3x3(1024, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        # -------------------------
        self.fc1024= nn.Linear(1024, 1, bias=False) # att for base_feat1
        self.fc128 = nn.Linear(128, 1, bias=False) # att
        #--------------------------
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        #---

        self.conv4 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
          padding=0, bias=False)
        self.gap_fc = nn.Linear(128, 1, bias=False)
        self.gmp_fc = nn.Linear(128, 1, bias=False)

        #---

    def forward(self, x):
        # print('\nx:', x.shape)
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn0(self.conv0(x)))

        # x = F.relu(self.bn1(self.conv0(x)))
        # atten map 1024
        # # print('\nx:', x.shape)
        # xxx = x.view(-1, 1024)
        # xxx = self.fc1024(xxx)
        # # print('\nxxx:', xxx.shape)
        # xxx_w = list(self.fc1024.parameters())[0]
        # xxx_w = xxx_w.unsqueeze(2).unsqueeze(3)
        # print('\nxxx_w: ', xxx_w.shape)
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)


        # self-attention
        # x = x * xxx_w

        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        #---

        x = self.conv3(x) # 1x128xhxw
        # print('\nx bf cam: ', x.shape)

        # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        # gap_weight = list(self.gap_fc.parameters())[0]
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        # # print('\ngap: ', gap.shape)
        # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        # gmp_weight = list(self.gmp_fc.parameters())[0]
        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        # # print('\ngmp: ', gmp.shape)

        # cam_logit = torch.cat([gap_logit, gmp_logit], 1)

        # x = torch.cat([gap, gmp], 1) # 1x128xhxw + 1x128xhxw

        # x = self.conv4(x)
        # print('\nx af cam: ', x.shape)
        # print('\ncam_logit: ', cam_logit.shape)
        #---

        #---

        x = F.dropout(F.relu(self.bn3(x)),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)


        # atten map 128
        # xx = self.fc128(x)
        # xx_w = list(self.fc128.parameters())[0]
        # # self-attention
        # x = x * xx_w
        # xx_w = xx_w.unsqueeze(2).unsqueeze(3)


        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          # return x, xx_w, xxx_w
          return x#, cam_logit

class netD_dc(nn.Module):
    def __init__(self):
        super(netD_dc, self).__init__()
        self.fc1 = nn.Linear(2048,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False,gc1=False,gc2=False,gc3=False):
    self.model_path = cfg.RESNET_PATH
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.gc1 = gc1
    self.gc2 = gc2
    self.gc3 = gc3
    self.layers = num_layers
    if self.layers == 50:
      self.model_path = '/home/grad3/keisaito/data/pretrained_model/resnet50_caffe.pth'
    _fasterRCNN.__init__(self, classes, class_agnostic,gc1,gc2,gc3)

  def _init_modules(self):

    resnet = resnet101()
    if self.layers == 50:
      resnet = resnet50()
    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
    # Build resnet.
    self.RCNN_base1 = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1)
    self.RCNN_base2 = nn.Sequential(resnet.layer2)
    self.RCNN_base3 = nn.Sequential(resnet.layer3)

    self.netD1 = netD1()
    self.netD_forward1 = netD_forward1()
    self.netD2 = netD2()
    self.netD_forward2 = netD_forward2()
    self.netD3 = netD3()
    self.netD_forward3 = netD_forward3()

    #---
    # self.netD12 = netD12()
    # self.netD13 = netD13()
    # self.netD21 = netD21()
    # #---

    self.RCNN_top = nn.Sequential(resnet.layer4)
    feat_d = 2048
    feat_d += 128
    feat_d += 128
    feat_d += 128
    self.netD_inst = netD_inst(fc_size = feat_d)
    self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base1[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base1[1].parameters(): p.requires_grad=False

    # assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    # if cfg.RESNET.FIXED_BLOCKS >= 3:
    #   for p in self.RCNN_base1[6].parameters(): p.requires_grad=False
    # if cfg.RESNET.FIXED_BLOCKS >= 2:
    #   for p in self.RCNN_base1[5].parameters(): p.requires_grad=False
    #if cfg.RESNET.FIXED_BLOCKS >= 1:
    #  for p in self.RCNN_base1[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base1.apply(set_bn_fix)
    self.RCNN_base2.apply(set_bn_fix)
    self.RCNN_base3.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base1.eval()
      self.RCNN_base1[4].train()
      self.RCNN_base2.train()
      self.RCNN_base3.train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base1.apply(set_bn_eval)
      self.RCNN_base2.apply(set_bn_eval)
      self.RCNN_base3.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7