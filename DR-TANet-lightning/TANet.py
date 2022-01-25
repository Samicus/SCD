import torch
import torch.nn as nn
from util import upsample, criterion_CEloss
from TANet_element import *
import pytorch_lightning as pl
from params import MAX_EPOCHS
import torchmetrics


class TANet(pl.LightningModule):

    def __init__(self, encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, len_train_loader):
        super(TANet, self).__init__()
        self.len_train_loader = len_train_loader


        self.encoder1, channels = get_encoder(encoder_arch,pretrained=True)
        self.encoder2, _ = get_encoder(encoder_arch,pretrained=True)
        self.attention_module = get_attentionmodule(local_kernel_size, stride, padding, groups, drtam, refinement, channels)
        self.decoder = get_decoder(channels=channels)
        self.classifier = nn.Conv2d(channels[0], 2, 1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        
        self.train_accuracy = torchmetrics.Accuracy()

    def forward(self, img):

        img_t0,img_t1 = torch.split(img,3,1)
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
        
        weight = torch.ones(2)
        criterion = criterion_CEloss(weight.cuda())   
        inputs_train, mask_train = batch
        output_train = self(inputs_train)
        loss = criterion(output_train, mask_train[:, 0])
        
        # Log data to view in AIM
        self.log("train loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.train_accuracy(output_train, mask_train[:, 0])
        self.log("train accuracy", self.train_accuracy, on_epoch=True, prog_bar=True, logger=True)
        lambda_lr = self.model_lr_scheduler.get_last_lr()[0]
        self.log("lambda_lr", lambda_lr, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9,0.999))
        lambda_lr = lambda epoch:(float)(MAX_EPOCHS*self.len_train_loader-self.global_step)/(float)(MAX_EPOCHS*self.len_train_loader)
        self.model_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
        
        return [optimizer], [self.model_lr_scheduler]

    def training_epoch_end(self, outputs):
        
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(self.model_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.model_lr_scheduler.step()