import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

from base_cbm import LitBaseCBM

class ResNet50CBM(nn.Module):
    def __init__(self, latent_dim, num_classes, freeze_proj=False, concept_act='sigmoid'):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes 

        self.projector = resnet50(weights=ResNet50_Weights.DEFAULT)
        if freeze_proj:
            for param in self.projector.parameters():
                param.requires_grad = False
        self.projector.fc = nn.Linear(512 * 4, self.latent_dim)

        if concept_act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif concept_act == 'relu':
            self.act = nn.ReLU()
        elif (concept_act == 'None') or (concept_act is None):
            self.act = nn.Identity()
        else:
            raise NotImplementedError

        self.classification = nn.Linear(self.latent_dim, self.num_classes)

    def forward(self, x, return_concept=False):
        c = self.projector(x)
        y = self.classification(self.act(c))

        if return_concept:
            return c, y
        else:
            return y

class LitResNet50CBM(LitBaseCBM):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = LitBaseCBM.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("LitResNet50CBM")

        return parent_parser

    @staticmethod
    def load_checkpoint_as_model(ckpt_path):
        self = LitResNet50CBM.load_from_checkpoint(ckpt_path)
        return self.model

    def initialize_model(self):
        self.model = ResNet50CBM(latent_dim=self.hparams.latent_dim, 
                                 num_classes=self.hparams.num_classes,
                                 concept_act=self.hparams.concept_act)
        return self.model

    transform_aug = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    transform_const = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

