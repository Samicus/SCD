from util import upsample
from network.TANet_element import *
from torchmetrics.functional import precision, recall, f1_score, accuracy
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from aim import Image
import cv2
from os.path import join as pjoin


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
        self.log("train loss", train_loss, on_epoch=True, prog_bar=True, logger=True)
        
        return train_loss
    
    def test_step(self, batch, batch_idx):
        metrics = self.evaluation(batch, batch_idx, LOG_IMG=False)
        return metrics
    
    def validation_step(self, batch, batch_idx):
        if self.logger:
            metrics = self.evaluation(batch, batch_idx, LOG_IMG=True)
            self.log_dict(metrics)
            return metrics
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, "max")
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'f1-score'
            }
        
    def evaluation(self, batch, batch_idx, LOG_IMG=False):
        
        inputs, mask = batch
        mask = mask.int()
        current_batch_size = len(inputs)
        
        precision_tot = 0
        recall_tot = 0
        accuracy_tot = 0
        f1_score_tot = 0
        
        for idx, (inputs, target) in enumerate(zip(inputs, mask)):
            
            # Forward propagation
            inputs_forward = torch.unsqueeze(inputs, dim=0)
            pred = self(inputs_forward)
            pred = torch.squeeze(pred, dim=0)
            
            # Activation
            pred = torch.sigmoid(pred)
            pred[pred <= 0.5] = 0
            pred[pred > 0.5] = 1
            
            # Calculate metrics
            precision_tot += precision(pred, target)
            recall_tot += recall(pred, target)
            accuracy_tot += accuracy(pred, target)
            f1_score_tot += f1_score(pred, target)

            # Convert input to image
            t0 = ((inputs[0:3] + 1.0) * 128.0).type(torch.uint8)  # (RGB, height, width)
            t1 = ((inputs[3:6] + 1.0) * 128.0).type(torch.uint8)  # (RGB, height, width)

            # Convert prediction to image
            pred_img = pred.type(torch.uint8)
            pred_img = torch.cat((pred_img, pred_img, pred_img), 0)
            pred_img[pred_img == 1] = 255

            # Convert target to image
            target_img = target.type(torch.uint8)
            target_img = torch.cat((target_img, target_img, target_img), 0)
            target_img[target_img == 1] = 255

            # Stitch together inputs, prediction and target in a final image.
            input_images = torch.cat((t0, t1), 2)                       # Horizontal stack of inputs t0 and t1.
            mask_images = torch.cat((target_img, pred_img), 2)       # Horizontal stack of prediction and target.
            img_save = torch.cat((input_images, mask_images), 1)    # Vertical stack of inputs, prediction and target.
            
            if LOG_IMG:
                self.logger.experiment.track(
                    Image(img_save, "pred_{}".format(idx)), # Pass image data and/or caption
                    name="val_batch_{}".format(batch_idx),  # The name of image set
                    step=idx,   # Step index (optional)
                    #epoch=0,   # Epoch (optional)
                    context={   # Context (optional)
                        'subset': 'validation',
                    }
                )
            else:
                cv2.imwrite(pjoin("dir_img", "pred_{}_batch_{}.png".format(idx, batch_idx)), img_save.cpu().numpy())
                
        metrics = {'precision': precision_tot / current_batch_size, 
                       'recall': recall_tot / current_batch_size, 
                       'accuracy': accuracy_tot / current_batch_size, 
                       'f1-score': f1_score_tot / current_batch_size}
        
        return metrics
