import torch

from torchvision import models

from .resnet import ResNet18, ResNet34
from .vgg_ca import vgg16
from .resnet_ca import resnet110
from .densenet import densenet100
from .densenet_ca import dense_bc
from .nbsnet import get_conv, NbsCls,ConvNet
from .deeplabv3 import NbsDeepLabHead, GeneralDeepLabHead
from .wideresnet import wresnet28_2, wresnet28_10, wresnet16_8
from .regression import *

MODEL_DICT = {'mlp': [None, 'none', 28 * 28 * 3],
              'alexnet': [models.alexnet, 'avgpool', 256 * 6 * 6],
              'vgg16': [vgg16, 'features', 512],
              'resnet18': [ResNet18, 'layer4', 512],
              'resnet34': [ResNet34, 'layer4', 512],
              'resnet50': [models.resnet50, 'avgpool', 2048],
              'resnet110': [resnet110, 'avgpool', 64],
              'densenet100': [densenet100, 'bn1', 342],
              'densebc': [dense_bc, 'avgpool', 342],
              'squeeze1_0': [models.squeezenet1_0, 'features', 512],
              'mnasnet0_5': [models.mnasnet0_5, 'layers', 1280],
              'wresnet28_2': [wresnet28_2, 'avgpool', 128],
              'wresnet16_8': [wresnet16_8, 'avgpool', 512],
              'wresnet28_10': [wresnet28_10, 'avgpool', 640],
              'deeplabv3_res50': [models.resnet.resnet50, 'layer4', 2048],
              'deeplabv3_res101': [models.resnet.resnet101, 'layer4', 2048],
              'Reg_model': [Reg_model,'none',100],
              'Reg_model2': [Reg_model2,'none',1],
              'Reg_model3': [Reg_model3,'none',1],
              'Reg_model4': [Reg_model4,'none',4]}


def _get_model(name, model_type, num_classes, dropout_rate=0.):
    backbone, return_layer, in_feat = MODEL_DICT[name]
    backbone = backbone(in_feat)
    if model_type == 'nbs':
        classifier = NbsCls(in_feat, num_classes) # last layer of the model 
        classifier.num_classes = num_classes
    # elif model_type == 'nbs_seg':
    #     classifier = NbsDeepLabHead(in_feat, num_classes)
    # elif 'seg' in model_type:
    #     classifier = GeneralDeepLabHead(in_feat, num_classes)
    # else:
    #     classifier = torch.nn.Linear(in_feat, num_classes)
    #     classifier.num_classes = num_classes
        
    if backbone: 

        return ConvNet(backbone, classifier, last_drop=0.0) # Reg_model + last layer 
    else:
        return classifier # self.model in nbs_runner

