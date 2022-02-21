from tabnanny import check
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import join as pjoin
from params import dir_img
import torch
import cv2


__all__ = ['Upsample', 'upsample']

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)

class criterion_CEloss(nn.Module):
    def __init__(self, weight=None):
        super(criterion_CEloss, self).__init__()
        self.loss = nn.NLLLoss(weight)
    def forward(self, output, target):
        #print("MIN: " + str(torch.min(F.softmax(output, dim=1))))
        #print("MAX: " + str(torch.max(F.softmax(output, dim=1))))
        #print("MIN: " + str(torch.min(target)))
        #print("MAX: " + str(torch.max(target)))
        return self.loss(F.log_softmax(output, dim=1), target)
    
def l1_loss(input, target):
    """ L1 Loss without reduce flag.
    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor
    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1
    
    def forward(self, predict, target):
#         target = target.unsqueeze(1)
#         print(predict.shape,target.shape)
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        
        pre = torch.sigmoid(predict).view(num, -1)
#         pre = predict.view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum()  #鍒╃敤棰勬祴鍊间笌鏍囩鐩镐箻褰撲綔浜ら泦
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score

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

def cal_metrics(pred,target):

    temp = np.dstack((pred == 255, target == 255))
    TP = sum(sum(np.all(temp, axis=2)))

    temp = np.dstack((pred == 255, target == 0))
    FP = sum(sum(np.all(temp, axis=2)))

    temp = np.dstack((pred == 0, target == 255))
    FN = sum(sum(np.all(temp, axis=2)))

    temp = np.dstack((pred == 0, target == 0))
    TN = sum(sum(np.all(temp, axis=2)))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-8)
    f1_score = 2 * recall * precision / (precision + recall + 1e-8)

    return (precision, recall, accuracy, f1_score)
    
def generate_output_metrics(t0, t1, mask_gt, mask_pred, w_r, h_r, w_ori, h_ori, set_, ds, index, STORE=False):
    
    fn_img = pjoin(dir_img, '{0}-{1:08d}.png'.format(ds, index))
    w, h = w_r, h_r
    
    t0_r = np.transpose(t0.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    t1_r = np.transpose(t1.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    
    mask_target = cv2.cvtColor(mask_gt.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    mask_prediction = cv2.cvtColor(mask_pred.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    input_images = np.hstack((t0_r, t1_r))                      # Input images side-by-side
    mask_images = np.hstack((mask_target, mask_prediction))     # Prediction and target images side-by-side
    
    img_save = np.vstack((input_images, mask_images))           # Stack input and mask horizontally
    
    if w != w_ori or h != h_ori:
            img_save = cv2.resize(img_save, (h_ori, w_ori))
            
    if STORE:
        cv2.imwrite(fn_img, img_save)
    else:
        return (img_save, fn_img)