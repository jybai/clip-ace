import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from transformers import CLIPFeatureExtractor, CLIPImageProcessor, CLIPVisionModelWithProjection, logging

from base_cbm import LitBaseCBM
from utils import get_named_module, log_level

class CLIPClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes, freeze_clip=True, concept_act='sigmoid'):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes 

        with log_level(logging.ERROR):
            # self.clip_feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_projector = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            if freeze_clip:
                for param in self.clip_projector.parameters():
                    param.requires_grad = False

        self.linear = nn.Linear(self.clip_projector.config.projection_dim,
                                self.latent_dim)
        if concept_act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif concept_act == 'relu':
            self.act = nn.ReLU()
        elif (concept_act == 'None') or (concept_act is None):
            self.act = nn.Identity()
        else:
            raise NotImplementedError

        self.classification = nn.Linear(self.latent_dim, self.num_classes)

    def load_weight(self, module_name, weight_npy_path):
        module = get_named_module(self, module_name)
        weight = np.load(weight_npy_path)
        weight_pt = torch.from_numpy(weight).to(next(module.parameters()).device)
        with torch.no_grad():
            assert module.weight.shape == weight_pt.shape
            module.weight = nn.Parameter(weight_pt)
        print(f"Loaded {module_name} weights from {weight_npy_path}.")

    def forward(self, x, return_concept=False):
        # expect x is the output of CLIPImageProcessor
        x = self.clip_projector(**x) # {'last_hidden_state': [B, ViT_N_TOKENS, HIDDEN_SIZE]', 'image_embeds': [B, PROJECTION_DIM]}
        c = self.linear(x['image_embeds']) # weight: PROJECTION_DIM * d -> d concept
        x = self.act(c)
        x = self.classification(x) # weight d * c -> c class

        if return_concept:
            return c, x
        else:
            return x

class LitTSClassifier(pl.LightningModule):
# {{{
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitTSClassifier")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--num_classes", type=int, default=200)
        parser.add_argument("--latent_dim", type=int, default=8)
        parser.add_argument("--teacher_ckpt", type=str, default="/nfs/data/andrewbai/concept-gradients/scripts/models/cuba/cuba_x2y_resnet50_all-data/version_0/model.ckpt")
        return parent_parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.teacher = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', num_classes=self.hparams.num_classes, verbose=False)
        self.teacher.load_state_dict(torch.load(self.hparams.teacher_ckpt))
        self.teacher = self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = CLIPClassifier(latent_dim=self.hparams.latent_dim, num_classes=self.hparams.num_classes)
        self.student = self.student.train()

        self.acc_trn = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.acc_val = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.acc_tst = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)

    def training_step(self, batch, batch_idx):
        x = batch
        y_student = self.student(x) # unnormalized logits
        with torch.no_grad():
            y_teacher = self.teacher(x)
        loss = F.cross_entropy(y_student, y_teacher.softmax(-1))
        self.acc_trn.update(y_student.softmax(-1), y_teacher.argmax(-1))
        self.log('acc_trn', self.acc_trn, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y_student = self.student(x) # unnormalized logits
        with torch.no_grad():
            y_teacher = self.teacher(x)
        self.acc_val.update(y_student.softmax(-1), y_teacher.argmax(-1))
        self.log('acc_val', self.acc_val, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x = batch
        y_student = self.student(x) # unnormalized logits
        with torch.no_grad():
            y_teacher = self.teacher(x)
        self.acc_tst.update(y_student.softmax(-1), y_teacher.argmax(-1))
        self.log('acc_tst', self.acc_tst, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        return optimizer
# }}}

class LitClassifier(pl.LightningModule):
# {{{
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitClassifier")
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--latent_dim", type=int, default=8)

        parser.add_argument("--load_linear_path", type=str, default=None)
        parser.add_argument("--freeze_linear", action="store_true")
        return parent_parser

    @staticmethod
    def load_checkpoint_as_model(ckpt_path):
        self = LitClassifier.load_from_checkpoint(ckpt_path)
        return self.model

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = CLIPClassifier(latent_dim=self.hparams.latent_dim, 
                                    num_classes=self.hparams.num_classes)
        if self.hparams.load_linear_path is not None:
            self.model.load_weight('linear', self.hparams.load_linear_path)
        if self.hparams.freeze_linear:
            self.model.linear.weight.requires_grad = False

        self.acc_trn = torchmetrics.Accuracy(task='multiclass', 
                                             num_classes=self.hparams.num_classes)
        self.acc_val = torchmetrics.Accuracy(task='multiclass', 
                                             num_classes=self.hparams.num_classes)
        self.acc_tst = torchmetrics.Accuracy(task='multiclass', 
                                             num_classes=self.hparams.num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)
        self.acc_trn.update(y_pred.softmax(-1), y)
        self.log('acc_trn', self.acc_trn, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x) # unnormalized logits
        self.acc_val.update(y_pred.softmax(-1), y)
        self.log('acc_val', self.acc_val, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x) # unnormalized logits
        self.acc_tst.update(y_pred.softmax(-1), y)
        self.log('acc_tst', self.acc_tst, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        return optimizer
# }}}

class LitClipCBM(LitBaseCBM):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = LitBaseCBM.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("LitCLipCBM")
        parser.add_argument("--load_linear_path", type=str, default=None)
        parser.add_argument("--freeze_linear", action="store_true")
        parser.add_argument("--load_classification_path", type=str, default=None)
        parser.add_argument("--freeze_classification", action="store_true")

        return parent_parser

    @staticmethod
    def load_checkpoint_as_model(ckpt_path, strict=False):
        self = LitClipCBM.load_from_checkpoint(ckpt_path, strict=strict)
        return self.model

    def initialize_model(self):
        self.model = CLIPClassifier(latent_dim=self.hparams.latent_dim, 
                                    num_classes=self.hparams.num_classes,
                                    concept_act=self.hparams.concept_act)
        if hasattr(self.hparams, "load_linear_path") and \
                (self.hparams.load_linear_path is not None):
            self.model.load_weight('linear', self.hparams.load_linear_path)
        if hasattr(self.hparams, "freeze_linear") and \
                self.hparams.freeze_linear:
            self.model.linear.weight.requires_grad = False
        if hasattr(self.hparams, "load_classification_path") and \
                self.hparams.load_classification_path is not None:
            self.model.load_weight('classification', self.hparams.load_classification_path)
        if hasattr(self.hparams, "freeze_classification") and \
                self.hparams.freeze_classification:
            self.model.classification.weight.requires_grad = False        

        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        def transform(*args, **kwargs):
            out = self.processor.preprocess(*args, return_tensors='pt', **kwargs)
            for k in out.keys():
                out[k].squeeze_(0)
            return out

        self.transform_aug = transform
        self.transform_const = transform

        return self.model

# {{{
'''
class LitCBM(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitCBM")
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--latent_dim", type=int, default=8)
        parser.add_argument("--alpha", type=float, default=0.5)

        parser.add_argument("--load_linear_path", type=str, default=None)
        parser.add_argument("--freeze_linear", action="store_true")

        parser.add_argument("--concept_act", type=str, default='sigmoid',
                            choices=['sigmoid', 'relu', 'None'])
        return parent_parser

    @staticmethod
    def load_checkpoint_as_model(ckpt_path):
        self = LitCBM.load_from_checkpoint(ckpt_path)
        return self.model

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if (self.hparams.alpha < 0) or (self.hparams.alpha > 1):
            raise ValueError(f"alpha must be in [0, 1] but got {self.hparams.alpha} instead.")

        self.model = CLIPClassifier(latent_dim=self.hparams.latent_dim, 
                                    num_classes=self.hparams.num_classes,
                                    concept_act=self.hparams.concept_act)
        if self.hparams.load_linear_path is not None:
            self.model.load_linear(self.hparams.load_linear_path)
        if self.hparams.freeze_linear:
            self.model.linear.weight.requires_grad = False

        self.acc_y_trn = torchmetrics.Accuracy(task='multiclass', 
                                               num_classes=self.hparams.num_classes)
        self.acc_y_val = torchmetrics.Accuracy(task='multiclass', 
                                               num_classes=self.hparams.num_classes)
        self.acc_y_tst = torchmetrics.Accuracy(task='multiclass', 
                                               num_classes=self.hparams.num_classes)

        self.acc_c_trn = torchmetrics.Accuracy(task='multilabel', 
                                               num_labels=self.hparams.latent_dim)
        self.acc_c_val = torchmetrics.Accuracy(task='multilabel', 
                                               num_labels=self.hparams.latent_dim)
        self.acc_c_tst = torchmetrics.Accuracy(task='multilabel', 
                                               num_labels=self.hparams.latent_dim)

        self.loss_y = None
        self.loss_c = None

    def initialize_criterions(self, dl):
        cs, ys = [], [] 
        for x, c, y in dl:
            cs.append(c)
            ys.append(y)
        cs = torch.cat(cs, dim=0)
        ys = torch.cat(ys, dim=0)

        weight_c = (cs == 1).sum(0) / (cs == 0).sum(0)
        weight_y = torch.tensor([1. / (ys == i).sum() 
                                 for i in range(self.hparams.num_classes)])

        # print(weight_c.shape, weight_y.shape)

        self.loss_c = nn.BCEWithLogitsLoss(pos_weight=weight_c)
        self.loss_y = nn.CrossEntropyLoss(weight=weight_y)

    def training_step(self, batch, batch_idx):
        x, c, y = batch # float32, int64, int64
        c_pred, y_pred = self.model(x, return_concept=True)

        loss_y = self.loss_y(y_pred, y)
        self.acc_y_trn.update(y_pred.softmax(-1), y)
        self.log('acc_y_trn', self.acc_y_trn, on_step=False, on_epoch=True)

        loss_c = self.loss_c(c_pred, c.float())
        self.acc_c_trn.update(torch.sigmoid(c_pred), c)
        self.log('acc_c_trn', self.acc_c_trn, on_step=False, on_epoch=True)

        loss = loss_y * self.hparams.alpha + loss_c * (1 - self.hparams.alpha)

        return loss

    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        c_pred, y_pred = self.model(x, return_concept=True) # unnormalized logits
        self.acc_y_val.update(y_pred.softmax(-1), y)
        self.log('acc_y_val', self.acc_y_val, on_step=False, on_epoch=True)
        self.acc_c_val.update(torch.sigmoid(c_pred), c)
        self.log('acc_c_val', self.acc_c_val, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, c, y = batch
        c_pred, y_pred = self.model(x, return_concept=True) # unnormalized logits
        self.acc_y_tst.update(y_pred.softmax(-1), y)
        self.log('acc_y_tst', self.acc_y_tst, on_step=False, on_epoch=True)
        self.acc_c_tst.update(torch.sigmoid(c_pred), c)
        self.log('acc_c_tst', self.acc_c_tst, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        return optimizer

'''
# }}}
