import torch
import torch.nn as nn
from util import cal_metrics, upsample, criterion_CEloss, store_imgs_and_cal_metrics, return_imgs_and_cal_metrics
from TANet_element import *
from pytorch_lightning import LightningModule
from params import MAX_EPOCHS, BATCH_SIZE
import numpy as np
import torch.nn.functional as F
from torchmetrics.functional import jaccard_index, precision, recall, f1_score
from aim import Image

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
        self.classifier = nn.Conv2d(channels[0], 2, 1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(channels[0])
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
        train_loss = F.cross_entropy(output_train, mask_train[:, 0])
        #self.log("train loss", train_loss, on_epoch=True, prog_bar=True, logger=True)
        
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
            output= self(input)

            input = input[0].cpu().data
            img_t0 = input[0:3, :, :]
            img_t1 = input[3:6, :, :]
            img_t0 = (img_t0+1) * 128
            img_t1 = (img_t1+1) * 128
            output = output[0].cpu().data
            
            #print("MIN: " + str(torch.min(F.softmax(output, dim=0))))
            #print("MAX: " + str(torch.max(F.softmax(output, dim=0))))
            
            mask_pred = np.where(F.softmax(output,dim=0)[0]>0.5, 0, 255)
            mask_gt = np.squeeze(np.where(mask.cpu()==True,255,0),axis=0)
            
            ds = "TSUNAMI"
            
            (precision, recall, accuracy, f1_score) = store_imgs_and_cal_metrics(img_t0, img_t1, mask_gt, mask_pred, w_r, h_r, w_ori, h_ori, self.set_, ds, index)
            
            precision_total += precision
            recall_total += recall
            accuracy_total += accuracy
            f1_score_total += f1_score
            
        metrics = {'precision': precision_total/img_cnt, 'recall': recall_total/img_cnt, 'accuracy': accuracy_total/img_cnt, 'f1-score': f1_score_total/img_cnt}
        #self.log_dict(metrics)
        
        return metrics
    
    
    def validation_step(self, batch, batch_idx):
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
            output= self(input)

            input = input[0].cpu().data
            img_t0 = input[0:3, :, :]
            img_t1 = input[3:6, :, :]
            img_t0 = (img_t0+1) * 128
            img_t1 = (img_t1+1) * 128
            output = output[0].cpu().data
            
            #print("MIN: " + str(torch.min(F.softmax(output, dim=0))))
            #print("MAX: " + str(torch.max(F.softmax(output, dim=0))))

            mask_pred = np.where(F.softmax(output,dim=0)[0]>0.5, 0, 255)
            mask_gt = np.squeeze(np.where(mask.cpu()==True,255,0),axis=0)
            
            #(precision, recall, accuracy, f1_score) = cal_metrics(mask_pred, mask_gt)
            ds = "TSUNAMI"
            (precision, recall, accuracy, f1_score, img_save) = return_imgs_and_cal_metrics(img_t0, img_t1, mask_gt, mask_pred, w_r, h_r, w_ori, h_ori, self.set_, ds, index)
            
            precision_total += precision
            recall_total += recall
            accuracy_total += accuracy
            f1_score_total += f1_score

            self.logger.experiment.track(
                Image(img_save, "Prediction"), # Pass image data and/or caption
                name='pred_{}'.format(idx), # The name of image set
                #step=step,   # Step index (optional)
                #epoch=0,     # Epoch (optional)
                context={    # Context (optional)
                    'subset': 'validation',
                },
            )

        f1_score = f1_score_total/img_cnt
        metrics = {'precision': precision_total/img_cnt, 'recall': recall_total/img_cnt, 'accuracy': accuracy_total/img_cnt, 'f1-score': f1_score}
        #self.log_dict(metrics)

        print("F1-Score: {}".format(f1_score))
        
        return metrics
    
    def configure_optimizers(self):
        
        #weights = torch.ones(2)
        #self.criterion = criterion_CEloss(weights.cuda())
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        lambda_lr = lambda epoch:(float)(MAX_EPOCHS*self.len_train_loader-self.global_step)/(float)(MAX_EPOCHS*self.len_train_loader)
        self.model_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
        
        return [optimizer], [self.model_lr_scheduler]

    def training_epoch_end(self, outputs):
        
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(self.model_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.model_lr_scheduler.step()