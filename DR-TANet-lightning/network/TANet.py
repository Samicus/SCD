from util import upsample
from network.TANet_element import *
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from aim import Image
from os.path import join as pjoin
from torchvision.utils import save_image
import os

dirname = os.path.dirname
dir_img = pjoin(dirname(dirname(dirname(__file__))), "ABLATION_RESULTS")

CHANNEL = 0
NUM_OUT_CHANNELS = 1

class TANet(LightningModule):

    def __init__(self, encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement, EXPERIMENT_NAME, DETERMINISTIC=False):
        super(TANet, self).__init__()
        self.EXPERIMENT_NAME = EXPERIMENT_NAME
        if 'PCD' in EXPERIMENT_NAME:
            self.WEIGHT = torch.tensor(2)
        elif 'VL_CMU_CD' in EXPERIMENT_NAME:
            self.WEIGHT = torch.tensor(4)
        else:
            self.WEIGHT = torch.tensor(1)
        self.DETERMINISTIC = DETERMINISTIC
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.confusion_matrix = ConfusionMatrix(num_classes=2)
        #self.f1_score = F1Score(num_classes=2, average=None, mdmc_average='samplewise')
        #self.precision = Precision(num_classes=2, average=None, mdmc_average='samplewise')
        #self.recall = Recall(num_classes=2, average=None, mdmc_average='samplewise')
        #self.accuracy = Accuracy(num_classes=2, average=None, mdmc_average='samplewise')
        
        # Network Layers
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
        
        opt = self.optimizers()
        opt.zero_grad()
        
        inputs_train, mask_train = batch
        output_train = self(inputs_train)
        
        train_loss = F.binary_cross_entropy_with_logits(output_train, mask_train, pos_weight=self.WEIGHT)
        
        if self.DETERMINISTIC:
            torch.use_deterministic_algorithms(False)
            self.manual_backward(train_loss)
            torch.use_deterministic_algorithms(True)
        else:
            self.manual_backward(train_loss)
        opt.step()
        
        self.log("train loss", train_loss, on_epoch=True, prog_bar=True, logger=True)
        
        return train_loss
    
    def test_step(self, batch, batch_idx):
        metrics = self.evaluate_batch(batch, batch_idx, LOG_IMG=None)
        self.log_dict(metrics)
        return metrics
    
    def validation_step(self, batch, batch_idx):
        LOG_IMG = False
        if self.logger:
            metrics = LOG_IMG = True
        metrics = self.evaluate_batch(batch, batch_idx, LOG_IMG)
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return metrics
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, "max")
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'f1-score'
            }
        
    def evaluate_batch(self, batch, batch_idx, LOG_IMG=False):
        
        inputs_test, mask_test = batch
        preds = self(inputs_test)
        val_loss = F.binary_cross_entropy_with_logits(preds, mask_test.float(), self.WEIGHT)

        # Activation
        preds = torch.sigmoid(preds)
        preds[preds > 0.5] = 1
        preds[preds <= 0.5] = 0
        
        [[TN, FP], [ FN, TP]] = self.confusion_matrix(preds, mask_test.int())
        
        precision_change = TP / (TP + FP) if (TP + FP) != 0.0 else 0.0
        recall_change = TP / (TP + FN) if (TP + FN) != 0.0 else 0.0
        accuracy_change = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) != 0.0 else 0.0
        f1_score_change = 2.0 * recall_change * precision_change / (precision_change + recall_change) if (precision_change + recall_change) != 0.0 else 0.0
        
        # Invert (F1-Score for no change)
        preds = 1.0 - preds
        mask_test = 1.0 - mask_test
        
        [[TN, FP], [ FN, TP]] = self.confusion_matrix(preds, mask_test.int())
        
        precision_no_change = TP / (TP + FP) if (TP + FP) != 0.0 else 0.0
        recall_no_change = TP / (TP + FN) if (TP + FN) != 0.0 else 0.0
        accuracy_no_change = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) != 0.0 else 0.0
        f1_score_no_change = 2.0 * recall_no_change * precision_no_change / (precision_no_change + recall_no_change) if (precision_no_change + recall_no_change) != 0.0 else 0.0
        
        # Mean for metrics of change and no change
        precision = (precision_change + precision_no_change) / 2.0
        recall = (recall_change + recall_no_change) / 2.0
        accuracy = (accuracy_change + accuracy_no_change) / 2.0
        f1_score = (f1_score_change + f1_score_no_change) / 2.0
        
        if LOG_IMG == True or LOG_IMG == None:
            
            self.gen_img(inputs_test, preds, mask_test, batch_idx, LOG_IMG)
            self.on_train_start

        metrics = {
            'precision': precision,
            'recall': recall,
            "accuracy": accuracy,
            'f1-score': f1_score,
            "val_loss": val_loss
            }
        
        torch.cuda.empty_cache()
        
        return metrics
    
    def gen_img(self, inputs_test, preds, mask_test, batch_idx, LOG_IMG):
        for idx, (inputs, pred, target) in enumerate(zip(inputs_test, preds, mask_test)):
            
            # Convert input to image
            t0 = (inputs[0:3] * 255.0).type(torch.uint8).transpose(2, 1)  # (RGB, height, width)
            t1 = (inputs[3:6] * 255.0).type(torch.uint8).transpose(2, 1)  # (RGB, height, width)

            # Convert prediction to image
            pred_img = pred.type(torch.uint8).transpose(2, 1)
            pred_img = torch.cat((pred_img, pred_img, pred_img), 0) # Grayscale --> RGB
            pred_img *= 255 # 1 --> 255

            # Convert target to image
            target_img = target.type(torch.uint8).transpose(2, 1)
            target_img = torch.cat((target_img, target_img, target_img), 0) # Grayscale --> RGB
            target_img *= 255 # 1 --> 255

            # Stitch together inputs, prediction and target in a final image.
            input_images = torch.cat((t0, t1), 2)                   # Horizontal stack of inputs t0 and t1.
            mask_images = torch.cat((target_img, pred_img), 2)      # Horizontal stack of prediction and target.
            img_save = torch.cat((input_images, mask_images), 1)    # Vertical stack of inputs, prediction and target.
            
            if LOG_IMG and not ("trial" in self.EXPERIMENT_NAME):
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
                save_to = pjoin(dir_img, self.EXPERIMENT_NAME)
                os.makedirs(save_to, exist_ok=True)
                save_image(mask_images.type(torch.float), pjoin(save_to, "pred_{}_batch_{}.png".format(idx, batch_idx)))