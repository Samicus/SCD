from util import cal_metrics, generate_output_metrics, upsample
from params import MAX_EPOCHS, BATCH_SIZE
from network.TANet_element import *
from torchmetrics.functional import precision, recall, f1_score, accuracy
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image as pil_image
from aim import Image
import numpy as np
import cv2


CHANNEL = 0
NUM_OUT_CHANNELS = 1

class TANet(LightningModule):

    def __init__(self, encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, len_train_loader):
        super(TANet, self).__init__()
        
        self.len_train_loader = len_train_loader
        self.set_ = 0
        self.save_hyperparameters()

        self.encoder1, channels = get_encoder(encoder_arch,pretrained=True)
        self.encoder2, _ = get_encoder(encoder_arch,pretrained=True)
        self.attention_module = get_attentionmodule(local_kernel_size, stride, padding, groups, drtam, refinement, channels)
        self.decoder = get_decoder(channels=channels)
        self.classifier = nn.Conv2d(channels[CHANNEL], NUM_OUT_CHANNELS, 1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(channels[CHANNEL])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        
        img_t0, img_t1 = torch.split(img, 3, 1)
        features_t0 = self.encoder1(img_t0)
        features_t1 = self.encoder2(img_t1)
        features = features_t0 + features_t1
        features_map = self.attention_module(features)
        pred_ = self.decoder(features_map)
        pred_ = upsample(pred_,[pred_.size()[2]*2, pred_.size()[3]*2])
        pred_ = self.bn(pred_)
        pred_ = upsample(pred_,[pred_.size()[2]*2, pred_.size()[3]*2])
        pred_ = self.relu(pred_)
        pred = self.classifier(pred_)

        return pred
    
    def training_step(self, batch, batch_idx):
                
        inputs_train, mask_train = batch
        output_train = self(inputs_train)
        
        train_loss = F.binary_cross_entropy_with_logits(output_train, mask_train)
        self.log("train loss", train_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return train_loss

    def test_step(self, batch, batch_idx):
        t0_b, t1_b, mask_b, w_ori_b, h_ori_b, w_r_b, h_r_b = batch
        
        precision_total = 0
        recall_total = 0
        accuracy_total = 0
        f1_score_total = 0
        
        img_cnt = len(t0_b)
        
        for idx in range(img_cnt):
            index = BATCH_SIZE * batch_idx + idx
            t0, t1, mask, w_ori, h_ori, w_r, h_r = t0_b[idx], t1_b[idx], mask_b[idx], w_ori_b[idx].item(), h_ori_b[idx].item(), w_r_b[idx].item(), h_r_b[idx].item()
            input = torch.from_numpy(np.concatenate((t0.cpu(), t1.cpu()), axis=0)).contiguous()
            input = input.view(1, -1, w_r, h_r)
            input = input.cuda()
            output = self(input)

            input = input[0].cpu().data
            img_t0 = input[0:3, :, :]
            img_t1 = input[3:6, :, :]
            img_t0 = (img_t0+1) * 128
            img_t1 = (img_t1+1) * 128
            output = output[0].cpu().data
            
            #print("MIN: " + str(torch.min(F.softmax(output, dim=0))))
            #print("MAX: " + str(torch.max(F.softmax(output, dim=0))))
            
            activated_output = torch.sigmoid(output)
            
            #print(torch.max(activated_output))
            #print(torch.min(activated_output))
            #exit()
            mask_pred = np.where(activated_output[0]>0.5, 255, 0)
            mask_gt = np.squeeze(np.where(mask.cpu()==True, 255, 0),axis=0)
            
            ds = "TSUNAMI"
            
            (precision, recall, accuracy, f1_score) = generate_output_metrics(img_t0, img_t1, mask_gt, mask_pred, w_r, h_r, w_ori, h_ori, self.set_, ds, index, STORE=True)
            
            precision_total += precision
            recall_total += recall
            accuracy_total += accuracy
            f1_score_total += f1_score
            
        metrics = {'precision': precision_total/img_cnt, 'recall': recall_total/img_cnt, 'accuracy': accuracy_total/img_cnt, 'f1-score': f1_score_total/img_cnt}
        self.log_dict(metrics, sync_dist=True)
        
        return metrics
    
    
    def validation_step(self, batch, batch_idx):
                
        inputs_val, mask_val = batch
        output_val = self(inputs_val)

        output_val = torch.sigmoid(output_val)
        output_val[output_val <= 0.5] = 0
        output_val[output_val > 0.5] = 1
        
        mask_val = mask_val.int()
        
        precision_val = precision(output_val, mask_val)
        recall_val = recall(output_val, mask_val)
        f1_score_val = f1_score(output_val, mask_val)
        accuracy_val = accuracy(output_val, mask_val)

        metrics = {'precision': precision_val, 'recall': recall_val, 'accuracy': accuracy_val, 'f1-score': f1_score_val}

        if self.logger:
            self.log_dict(metrics, sync_dist=True)
            for idx, (inputs, pred, target) in enumerate(zip(inputs_val, output_val, mask_val)):

                # Convert input to image
                t0 = ((inputs[0:3] + 1.0) * 128.0).type(torch.uint8)  # (RGB, height, width)
                t1 = ((inputs[3:6] + 1.0) * 128.0).type(torch.uint8)  # (RGB, height, width)

                # Convert prediction to image
                pred_img = pred.type(torch.uint8)
                pred_img[pred_img == 1] = 255

                # Convert target to image
                target_img = target.type(torch.uint8)
                target_img[target_img == 1] = 255

                # Stitch together inputs, prediction and target in a final image.
                input_images = torch.cat((t0, t1), 2)                       # Horizontal stack of inputs t0 and t1.
                comparison_img = torch.cat((target_img, pred_img), 2)       # Horizontal stack of prediction and target.
                #whole_img = torch.cat((input_images, comparison_img), 2)    # Vertical stack of inputs, prediction and target.

                self.logger.experiment.track(
                    Image(comparison_img, "pred_{}".format(idx)), # Pass image data and/or caption
                    name="val_batch_{}".format(batch_idx),  # The name of image set
                    step=idx,   # Step index (optional)
                    #epoch=0,   # Epoch (optional)
                    context={   # Context (optional)
                        'subset': 'validation',
                    },
                )
        
        return metrics
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, "min")
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'val_loss'
            }

