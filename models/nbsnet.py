import math
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict


class NbsCls(nn.Module):
    '''
    앞에서 사용한 모델 뒤에 추가적으로 붙이는 layer를 정의한 부분.
    여기서 randomness가 들어가서 output의 결과를 다양하게 만든다.

    in_feat : input feature size
    num_classes : output feature size
    x : 앞에서 사용한 모델의  output
    

    forward func : 
        alpha가 int인 경우, rand_like로 randomness 만들어서 x에 곱해서 결과를 내보낸다.
        아닌 경우 다른 방식으로 결과를 내보낸다. 
    '''
    def __init__(self, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self.fc_out = nn.Linear(in_feat, num_classes)
        self.num_classes = num_classes

    def forward(self, x, alpha):
        out1 = x
        if isinstance(alpha, int):
            res_ = torch.zeros([alpha, out1.size(0), self.num_classes]).cuda()
            for i in range(alpha):
                w = torch.rand_like(out1).cuda()
                res = self.fc_out(out1 * w)
                res_[i] += res
            return res_
        else:
            out2 = torch.exp(-F.interpolate(alpha[:, None], self.in_feat))[:, 0].cuda()
            return self.fc_out(out1 * out2)


class ConvNet(nn.Module):
    def __init__(self, backbone, classifier, last_drop=.0):
        super().__init__()
        self.backbone = backbone
        self.classifer = classifier
        self.dropout = nn.Dropout(p=last_drop)

    def forward(self, *x):
        x = list(x)
        out = self.backbone(x[0]) # x[0] : batch size x_train, x[1] : batch size y_train 
        # if out.size(-1) != 1:
        #     out = F.relu(out, inplace=True).mean([2, 3])
        # else:
        #     out = out.squeeze()
        out = self.dropout(out)
        x[0] = out
        return self.classifer(*x)


class SegNet(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.bacbone = backbone
        self.classifer = classifier

    def forward(self, *x):
        x = list(x)
        input_shape = x[0].shape[-2:]
        out = self.bacbone(x[0])
        x[0] = out
        out = self.classifer(*x)
        if out.dim() != 5:
            return F.interpolate(out, size=input_shape,
                                 mode='bilinear', align_corners=False)
        else:
            return out


class BackboneGetter(nn.Sequential):
    def __init__(self, model, return_layer):
        if not set([return_layer]).issubset([name for name, _ in
                                             model.named_children()]):
            raise ValueError("return_layer is not present in model")

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name == return_layer:
                break

        super().__init__(layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


def get_conv(backbone, return_layer, classifier, model_type, drop_rate=0.0):
    backbone = BackboneGetter(backbone(.0), return_layer)
    model = ConvNet(backbone, classifier, drop_rate)
    model.num_classes = classifier.num_classes
    return model
