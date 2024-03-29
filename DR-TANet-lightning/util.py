import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix
from pytorch_lightning.callbacks import Callback

__all__ = ['Upsample', 'upsample']

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)

class criterion_CEloss(nn.Module):
    def __init__(self, weight=None):
        super(criterion_CEloss, self).__init__()
        self.loss = nn.NLLLoss(weight)
    def forward(self, output, target):
        return self.loss(F.log_softmax(output, dim=1), target)

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2  # same conv
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))

class Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3):
        super(Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x

def cal_metrics_sklearn(pred,target):
    
    TN, FP, FN, TP = confusion_matrix(np.matrix.flatten(pred), np.matrix.flatten(target)).ravel()

    precision = TP / (TP + FP) if (TP + FP) != 0.0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0.0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) != 0.0 else 0.0
    f1_score = 2.0 * recall * precision / (precision + recall) if (precision + recall) != 0.0 else 0.0

    return precision, recall, accuracy, f1_score

def cal_metrics(pred,target):

    temp = np.dstack((pred == 1, target == 1))
    TP = sum(sum(np.all(temp,axis=2)))

    temp = np.dstack((pred == 1, target == 0))
    FP = sum(sum(np.all(temp,axis=2)))

    temp = np.dstack((pred == 0, target == 1))
    FN = sum(sum(np.all(temp, axis=2)))

    temp = np.dstack((pred == 0, target == 0))
    TN = sum(sum(np.all(temp, axis=2)))

    precision = TP / (TP + FP) if (TP + FP) != 0.0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0.0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) != 0.0 else 0.0
    f1_score = 2 * recall * precision / (precision + recall) if (precision + recall) != 0.0 else 0.0

    return (precision, recall, accuracy, f1_score)
    
def load_config(hparams_path):
    with open(hparams_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config

class F1tracker(Callback):

  def __init__(self):
    self.f1_scores = []
    
  def on_validation_epoch_end(self, trainer, module):
    f1_score = trainer.logged_metrics["f1-score"] # access it here
    self.f1_scores.append(f1_score)