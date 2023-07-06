import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics

import pytorch_lightning as pl

from utils import GreedyPerplexity

class LitBaseCBM(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitCBM")
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--latent_dim", type=int)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--alpha", type=float, default=0.5)
        parser.add_argument("--ignore_concept", action='store_true')

        parser.add_argument("--concept_act", type=str, default='sigmoid',
                            choices=['sigmoid', 'relu', 'None'])
        return parent_parser

    @staticmethod
    def load_checkpoint_as_model(ckpt_path):
        raise NotImplementedError

    def initialize_model(self):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if (self.hparams.alpha < 0) or (self.hparams.alpha > 1):
            raise ValueError(f"alpha must be in [0, 1] but got {self.hparams.alpha} instead.")
        
        if not hasattr(self.hparams, 'ignore_concept'):
            self.hparams.ignore_concept = False

        self.acc_y_trn = torchmetrics.Accuracy(task='multiclass', 
                                               num_classes=self.hparams.num_classes)
        self.acc_y_val = torchmetrics.Accuracy(task='multiclass', 
                                               num_classes=self.hparams.num_classes)
        self.acc_y_tst = torchmetrics.Accuracy(task='multiclass', 
                                               num_classes=self.hparams.num_classes)

        if not self.hparams.ignore_concept:
            self.acc_c_trn = torchmetrics.Accuracy(task='multilabel', 
                                                   num_labels=self.hparams.latent_dim)
            self.acc_c_val = torchmetrics.Accuracy(task='multilabel', 
                                                   num_labels=self.hparams.latent_dim)
            self.acc_c_tst = torchmetrics.Accuracy(task='multilabel', 
                                                   num_labels=self.hparams.latent_dim)

        self.model = self.initialize_model()
        self.loss_y = None

        if not self.hparams.ignore_concept:
            self.loss_c = None

    def initialize_criterions(self, dl):
        cs, ys = [], [] 
        for x, c, y in dl:
            cs.append(c)
            ys.append(y)
        ys = torch.cat(ys, dim=0)
        weight_y = torch.tensor([1. / (ys == i).sum() 
                                 for i in range(self.hparams.num_classes)])

        self.loss_y = nn.CrossEntropyLoss(weight=weight_y)

        if not self.hparams.ignore_concept:
            weight_c = (cs == 1).sum(0) / (cs == 0).sum(0)
            cs = torch.cat(cs, dim=0)
            self.loss_c = nn.BCEWithLogitsLoss(pos_weight=weight_c)

    def training_step(self, batch, batch_idx):
        x, c, y = batch # float32, int64, int64

        if not self.hparams.ignore_concept:
            c_pred, y_pred = self.model(x, return_concept=True)
        else:
            y_pred = self.model(x)

        loss = 0

        loss_y = self.loss_y(y_pred, y)
        loss += loss_y * (1 - self.hparams.alpha)
        self.acc_y_trn.update(y_pred.softmax(-1), y)
        self.log('acc_y_trn', self.acc_y_trn, on_step=False, on_epoch=True)

        if not self.hparams.ignore_concept:
            loss_c = self.loss_c(c_pred, c.float())
            loss += loss_c * self.hparams.alpha
            self.acc_c_trn.update(torch.sigmoid(c_pred), c)
            self.log('acc_c_trn', self.acc_c_trn, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        if not self.hparams.ignore_concept:
            c_pred, y_pred = self.model(x, return_concept=True)
        else:
            y_pred = self.model(x)

        self.acc_y_val.update(y_pred.softmax(-1), y)
        self.log('acc_y_val', self.acc_y_val, on_step=False, on_epoch=True)

        if not self.hparams.ignore_concept:
            self.acc_c_val.update(torch.sigmoid(c_pred), c)
            self.log('acc_c_val', self.acc_c_val, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, c, y = batch
        if not self.hparams.ignore_concept:
            c_pred, y_pred = self.model(x, return_concept=True)
        else:
            y_pred = self.model(x)

        self.acc_y_tst.update(y_pred.softmax(-1), y)
        self.log('acc_y_tst', self.acc_y_tst, on_step=False, on_epoch=True)

        if not self.hparams.ignore_concept:
            self.acc_c_tst.update(torch.sigmoid(c_pred), c)
            self.log('acc_c_tst', self.acc_c_tst, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        return optimizer

class LitBaseLFCBM(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitBaseLFCBM")
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--num_concepts", type=int)
        parser.add_argument("--tokens_per_concept", type=int)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--alpha", type=float, default=0.5)

        return parent_parser

    @staticmethod
    def load_checkpoint_as_model(ckpt_path):
        raise NotImplementedError

    def initialize_model(self):
        raise NotImplementedError

    def get_concept_prefix(self, decode=False):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if (self.hparams.alpha < 0) or (self.hparams.alpha > 1):
            raise ValueError(f"alpha must be in [0, 1] but got {self.hparams.alpha} instead.")

        self.acc_y_trn = torchmetrics.Accuracy(task='multiclass', 
                                               num_classes=self.hparams.num_classes)
        self.acc_y_val = torchmetrics.Accuracy(task='multiclass', 
                                               num_classes=self.hparams.num_classes)
        self.acc_y_tst = torchmetrics.Accuracy(task='multiclass', 
                                               num_classes=self.hparams.num_classes)

        self.model = self.initialize_model()
        self.loss_y = None

    def initialize_criterions(self, dl):
        cs, ys = [], [] 
        for x, c, y in dl:
            cs.append(c)
            ys.append(y)
        cs = torch.cat(cs, dim=0)
        ys = torch.cat(ys, dim=0)

        weight_y = torch.tensor([1. / (ys == i).sum() 
                                 for i in range(self.hparams.num_classes)])

        self.loss_y = nn.CrossEntropyLoss(weight=weight_y)

    def training_step(self, batch, batch_idx):
        x, c, y = batch # float32, int64, int64
        y_pred = self.model(x)

        loss_y = self.loss_y(y_pred, y)
        self.acc_y_trn.update(y_pred.softmax(-1), y)
        self.log('acc_y_trn', self.acc_y_trn, on_step=False, on_epoch=True)

        c_decode_output = self.get_concept_prefix(decode=True)
        print(c_decode_output.keys())
        total_log_probs, count = _greedy_perplexity_update(c_decode_output['decoder_logits'])
        print(total_log_probs.shape)
        loss_c = total_log_probs.mean()

        loss = loss_y * (1 - self.hparams.alpha) + loss_c * self.hparams.alpha

        self.log('loss_y_trn', loss_y, on_step=True, on_epoch=True)
        self.log('loss_c_trn', loss_c, on_step=True, on_epoch=True)
        self.log('loss_trn', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        y_pred = self.model(x) # unnormalized logits
        self.acc_y_val.update(y_pred.softmax(-1), y)
        self.log('acc_y_val', self.acc_y_val, on_step=False, on_epoch=True)

    def validation_epoch_end(self, validation_step_outputs):
        c_decode_output = self.get_concept_prefix(decode=True)
        total_log_probs, count = _greedy_perplexity_update(c_decode_output)
        gpp = torch.exp(total_log_probs / count)
        self.log('gpp_c_val', gpp)

    def test_step(self, batch, batch_idx):
        x, c, y = batch
        y_pred = self.model(x) # unnormalized logits
        self.acc_y_tst.update(y_pred.softmax(-1), y)
        self.log('acc_y_tst', self.acc_y_tst, on_step=False, on_epoch=True)

    def test_epoch_end(self, test_step_outputs):
        c_decode_output = self.get_concept_prefix(decode=True)
        total_log_probs, count = _greedy_perplexity_update(c_decode_output)
        gpp = torch.exp(total_log_probs / count)
        self.log('gpp_c_tst', gpp)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        return optimizer

