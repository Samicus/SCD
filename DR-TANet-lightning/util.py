import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from os.path import join as pjoin

__all__ = ['Upsample', 'upsample']

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)

class criterion_CEloss(nn.Module):
    def __init__(self,weight=None):
        super(criterion_CEloss, self).__init__()
        self.loss = nn.NLLLoss(weight)
    def forward(self,output,target):
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

def cal_metrcis(pred,target):
    print(pred.shape)
    target = target[0,:,:]
    temp = np.dstack((pred == 0, target == 0))
    TP = sum(sum(np.all(temp,axis=2)))

    temp = np.dstack((pred == 0, target == 255))
    FP = sum(sum(np.all(temp,axis=2)))

    temp = np.dstack((pred == 255, target == 0))
    FN = sum(sum(np.all(temp, axis=2)))

    temp = np.dstack((pred == 255, target == 255))
    TN = sum(sum(np.all(temp, axis=2)))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    f1_score = 2 * recall * precision / (precision + recall)

    return (precision, recall, accuracy, f1_score)

def store_imgs_and_cal_matrics(t0, t1, mask_gt, mask_pred, w_r, h_r, w_ori, h_ori, set_, ds, index):
        
        #move to params
        dir_img = "/home/elias/sam_dev/SCD/dir_img"
        resultdir  = "/home/elias/sam_dev/SCD/resultdir"
        print("\n",index,"\n")
        fn_img = pjoin(dir_img, '{0}-{1:08d}.png'.format(ds, index))
        w, h = w_r, h_r
        img_save = np.zeros((w * 2, h * 2, 3), dtype=np.uint8)
        print("w: ", w)
        print("mask_gt: ", mask_gt.shape)
        img_save[0:w, 0:h, :] = np.transpose(t0.numpy()[0], (1, 2, 0)).astype(np.uint8)
        img_save[0:w, h:h * 2, :] = np.transpose(t1.numpy()[0], (1, 2, 0)).astype(np.uint8)
        img_save[w:w * 2, 0:h, :] = cv2.cvtColor(mask_gt[0,:,:].astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_save[w:w * 2, h:h * 2, :] = cv2.cvtColor(mask_pred.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        print("w: ", w)
        print("w_ori: ", w_ori)
        if w != w_ori or h != h_ori:
            img_save = cv2.resize(img_save, (h_ori, w_ori))

        fn_save = fn_img
        if not os.path.exists(dir_img):
            os.makedirs(dir_img)

        print('Writing' + fn_save + '......')
        cv2.imwrite(fn_save, img_save)
        """
        if set_ is not None:
            f_metrics = open(pjoin(resultdir, "eval_metrics_set{0}(single_image).csv".format(set_)), 'a+')
        else:
            f_metrics = open(pjoin(resultdir, "eval_metrics(single_image).csv"), 'a+')
        """
        #metrics_writer = csv.writer(f_metrics)
        fn = '{0}-{1:08d}'.format(ds,index)
        precision, recall, accuracy, f1_score = cal_metrcis(mask_pred,mask_gt)
        #metrics_writer.writerow([fn, precision, recall, accuracy, f1_score])
        #f_metrics.close()
        print("Precision: ", precision, " Recall: ", recall, " Accuracy: ", accuracy, " F1_Score: ", f1_score)
        return (precision, recall, accuracy, f1_score)
