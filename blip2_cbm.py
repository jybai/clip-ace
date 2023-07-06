import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from transformers import logging, Blip2VisionModel, BlipImageProcessor

from base_cbm import LitBaseCBM
from utils import get_named_module, log_level

class BLIP2Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes, freeze_proj=True, concept_act='sigmoid'):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes 

        with log_level(logging.ERROR):
            self.vision_projector = Blip2VisionModel.from_pretrained("Salesforce/blip2-opt-2.7b")
            if freeze_proj:
                for param in self.vision_projector.parameters():
                    param.requires_grad = False

        self.linear = nn.Linear(self.vision_projector.config.hidden_size,
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
        # expect x is the output of BlipImageProcessor
        x = self.vision_projector(**x) # {'last_hidden_state': [B, ViT_N_TOKENS, HIDDEN_SIZE]', 'pooler_output': [B, PROJECTION_DIM]}
        c = self.linear(x['pooler_output']) # weight: PROJECTION_DIM * d -> d concept
        x = self.act(c)
        x = self.classification(x) # weight d * c -> c class

        if return_concept:
            return c, x
        else:
            return x

class LitBlip2CBM(LitBaseCBM):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = LitBaseCBM.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("LitBlip2CBM")
        parser.add_argument("--load_linear_path", type=str, default=None)
        parser.add_argument("--freeze_linear", action="store_true")
        parser.add_argument("--load_classification_path", type=str, default=None)
        parser.add_argument("--freeze_classification", action="store_true")

        return parent_parser

    @staticmethod
    def load_checkpoint_as_model(ckpt_path, strict=False):
        self = LitBlip2CBM.load_from_checkpoint(ckpt_path, strict=strict)
        return self.model

    @staticmethod
    def get_processor():
        return BlipImageProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    def initialize_model(self):
        self.model = BLIP2Classifier(latent_dim=self.hparams.latent_dim, 
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

        return self.model

