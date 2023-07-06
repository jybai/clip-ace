import numpy as np
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from transformers import logging, BlipModel, BlipImageProcessor

from vector_gather import vector_gather
from base_cbm import LitBaseCBM
from utils import get_named_module, log_level
from qkv_modules import ScaledDotProductAttention

class BLIPViT(nn.Module):
    def __init__(self, num_classes, freeze_proj=True):
        super().__init__()
        self.num_classes = num_classes 

        with log_level(logging.ERROR):
            self.projector = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
            if freeze_proj:
                for param in self.projector.parameters():
                    param.requires_grad = False

        emb_dim = self.projector.config.vision_config.hidden_size
        self.classification = nn.Linear(emb_dim, self.num_classes)

    def forward(self, x, return_probs=False):
        # expect x is the output of BlipImageProcessor
        # x = self.projector.get_image_features(**x) # {'last_hidden_state': [B, ViT_N_TOKENS, HIDDEN_SIZE]', 'pooler_output': [B, PROJECTION_DIM]}
        x = self.projector.vision_model(**x)[0] # {'last_hidden_state': [B, ViT_N_TOKENS, HIDDEN_SIZE]', 'pooler_output': [B, PROJECTION_DIM]}
        x = x[:, 0] # extract the CLS token, shape = [B, HIDDEN_SIZE]
        x = self.classification(x)

        if return_probs:
            x = x.softmax(-1)

        return x

class BLIPClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes, freeze_proj=True, concept_act='sigmoid', concept_bn=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes 

        with log_level(logging.ERROR):
            self.projector = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
            if freeze_proj:
                for param in self.projector.parameters():
                    param.requires_grad = False

        # src: https://github.com/huggingface/transformers/blob/7bce8042606b01d0dba3b83d5d606f47529d6ba4/src/transformers/models/blip/configuration_blip.py#L328
        in_dim = self.projector.config.vision_config.hidden_size * 577
        # in_dim = self.projector.config.projection_dim
        self.linear = nn.Linear(in_dim, self.latent_dim)
        self.bn = nn.Identity() if not concept_bn else nn.BatchNorm1d(self.latent_dim)
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
        # x = self.projector.get_image_features(**x) # {'last_hidden_state': [B, ViT_N_TOKENS, HIDDEN_SIZE]', 'pooler_output': [B, PROJECTION_DIM]}
        x = self.projector.vision_model(**x)[0] # {'last_hidden_state': [B, ViT_N_TOKENS, HIDDEN_SIZE]', 'pooler_output': [B, PROJECTION_DIM]}
        x = torch.flatten(x, start_dim=1)
        c = self.linear(x) # weight: PROJECTION_DIM * d -> d concept
        x = self.bn(c)
        x = self.act(c)
        x = self.classification(x) # weight d * c -> c class

        if return_concept:
            return c, x
        else:
            return x

class BLIPConceptPrefixModel(nn.Module):
    def __init__(self, num_concepts, num_classes, num_tokens_per_concept=1):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes 
        self.num_tokens_per_concept = num_tokens_per_concept

        with log_level(logging.ERROR):
            self.projector = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
            for param in self.projector.parameters():
                param.requires_grad = False
        self.emb_dim = self.projector.config.vision_config.hidden_size

        # src: https://github.com/huggingface/transformers/blob/7bce8042606b01d0dba3b83d5d606f47529d6ba4/src/transformers/models/blip/configuration_blip.py#L328
        self.linear = nn.Linear(self.emb_dim, self.num_concepts * self.num_tokens_per_concept, bias=False)
        self.classification = nn.Linear(self.num_concepts, self.num_classes)

    def load_weight(self, module_name, weight_npy_path, transpose=False):
        module = get_named_module(self, module_name)
        weight = np.load(weight_npy_path)
        weight_pt = torch.from_numpy(weight).to(next(module.parameters()).device)
        with torch.no_grad():
            if module.weight.shape == weight_pt.shape:
                module.weight = nn.Parameter(weight_pt)
            elif module.weight.shape == weight_pt.T.shape:
                module.weight = nn.Parameter(weight_pt.T)
            else:
                raise ValueError(f"Pretrained weight.shape ({weight_pt.shape}) does not match module.shape ({module.weight.shape}).")
        print(f"Loaded {module_name} weights from {weight_npy_path}.")

    def forward(self, x, return_concept=False):
        # expect x is the output of BlipImageProcessor
        x = self.projector.vision_model(**x)[0] # {'last_hidden_state': [B, ViT_N_TOKENS, HIDDEN_SIZE]', 'pooler_output': [B, PROJECTION_DIM]}
        B, N_TOKENS = x.shape[0], x.shape[1]
        x = torch.reshape(x, (-1, self.emb_dim))
        qk = self.linear(x) # out.shape = [B*N_TOKENS, NUM_CONCEPTS]
        qk = torch.reshape(qk, (B, N_TOKENS, self.num_concepts, self.num_tokens_per_concept))
        score = torch.exp(qk) # out.shape = [B, N_TOKENS, NUM_CONCEPTS, N_TOKENS_PER_CONCEPT]
        score = score.sum(dim=3) # out.shape = [B, N_TOKENS, NUM_CONCEPTS]
        score = score / score.sum(dim=2, keepdim=True) # out.shape = [B, N_TOKENS, NUM_CONCEPTS]
        c = score.mean(dim=1) # global_avg_pooling, out.shape = [B, NUM_CONCEPTS]
        x = self.classification(c) # out.shape = [B, NUM_CLASSES]

        if return_concept:
            return c, x
        else:
            return x

class BLIPConceptPrefixModelV2(nn.Module):
    def __init__(self, num_concepts, num_classes, num_tokens_per_concept=1):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes 
        self.num_tokens_per_concept = num_tokens_per_concept

        with log_level(logging.ERROR):
            self.projector = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
            for param in self.projector.parameters():
                param.requires_grad = False
        self.emb_dim = self.projector.config.vision_config.hidden_size

        # src: https://github.com/huggingface/transformers/blob/7bce8042606b01d0dba3b83d5d606f47529d6ba4/src/transformers/models/blip/configuration_blip.py#L328
        self.linear = nn.Linear(self.emb_dim, self.num_concepts * self.num_tokens_per_concept, bias=False)
        self.classification = nn.Linear(self.emb_dim, self.num_classes)

    def load_weight(self, module_name, weight_npy_path, transpose=False):
        module = get_named_module(self, module_name)
        weight = np.load(weight_npy_path)
        weight_pt = torch.from_numpy(weight).to(next(module.parameters()).device)
        with torch.no_grad():
            if module.weight.shape == weight_pt.shape:
                module.weight = nn.Parameter(weight_pt)
            elif module.weight.shape == weight_pt.T.shape:
                module.weight = nn.Parameter(weight_pt.T)
            else:
                raise ValueError(f"Pretrained weight.shape ({weight_pt.shape}) does not match module.shape ({module.weight.shape}).")
        print(f"Loaded {module_name} weights from {weight_npy_path}.")

    def forward(self, x, return_concept=False):
        x = self.projector.vision_model(**x)[0] # {'last_hidden_state': [B, ViT_N_TOKENS, HIDDEN_SIZE]', 'pooler_output': [B, PROJECTION_DIM]}
        B, N_TOKENS = x.shape[0], x.shape[1]
        x = torch.reshape(x, (-1, self.emb_dim))
        qk = self.linear(x) # out.shape = [B*N_TOKENS, NUM_CONCEPTS * self.num_tokens_per_concept]
        qk = torch.reshape(qk, (B, N_TOKENS, -1))
        score = F.softmax(qk, dim=-1) # out.shape = [B, N_TOKENS, NUM_CONCEPTS * N_TOKENS_PER_CONCEPT]
        c = score.mean(1) # out.shape = [B, NUM_CONCEPTS * N_TOKENS_PER_CONCEPT]

        v = self.linear.weight # shape = [NUM_CONCEPTS * N_TOKENS_PER_CONCEPT, HIDDEN_SIZE]
        x = torch.matmul(c, v) # out.shape = [B, HIDDEN_SIZE]
        x = self.classification(x)

        if return_concept:
            return c, x
        else:
            return x

class BLIPConceptPrefixModelV3(nn.Module):
    def __init__(self, num_concepts, num_classes, topk):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes 
        self.topk = topk

        with log_level(logging.ERROR):
            self.projector = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
            for param in self.projector.parameters():
                param.requires_grad = False
        self.emb_dim = self.projector.config.vision_config.hidden_size

        # src: https://github.com/huggingface/transformers/blob/7bce8042606b01d0dba3b83d5d606f47529d6ba4/src/transformers/models/blip/configuration_blip.py#L328
        self.concept_cls = nn.Embedding(self.num_concepts, self.emb_dim)
        self.concept_act = nn.ReLU()
        self.classification = nn.Linear(self.emb_dim, self.num_classes)

    def forward(self, x, topk=None, include_concept_cls=False, return_concept=False, return_masked_q=False, 
                return_topk_idx=False, return_dict=False):
        if topk is None:
            topk = self.topk
        q = self.projector.vision_model(**x, return_dict=True)['last_hidden_state'] 
        B = q.shape[0]
        # x = {'last_hidden_state': [B, N_INPUT_TOKENS, HIDDEN_SIZE]', 'pooler_output': [B, PROJECTION_DIM]}

        q = q[:, 1:] # remove CLS token
        # qk.shape = [B, N_INPUT_TOKENS - 1, emb_dim]

        qk = q @ self.concept_cls.weight.T
        # qk.shape = [B, N_INPUT_TOKENS, N_CONCEPT_TOKENS]

        qk = qk.permute(0, 2, 1)
        # qk.shape = [B, N_CONCEPT_TOKENS, N_INPUT_TOKENS]

        qk_topk, idx_topk = torch.topk(qk, topk, dim=2)
        # idx_topk.shape = [B, N_CONCEPT_TOKENS, topk]

        q_topk = torch.stack([vector_gather(q, idx_topk[:, c]) for c in range(self.num_concepts)], dim=1)
        # q_topk.shape = [B, N_CONCEPT_TOKENS, topk, emb_dim]
        score = qk_topk.softmax(dim=2)
        # score.shape = [B, N_CONCEPT_TOKENS, topk]

        qkv_topk = q_topk * score.unsqueeze(-1)
        # qkv_topk.shape = [B, N_CONCEPT_TOKENS, topk, emb_dim]

        h = qkv_topk.sum(2).mean(1)
        c = h @ self.concept_cls.weight.T
        # c.shape = [B, N_CONCEPT_TOKENS]

        h_act = self.concept_act(h)
        y = self.classification(h_act)
        # y.shape = [B, N_CLASSES]

        out, out_names = [], []
        if return_masked_q:
            if include_concept_cls:
                batch_concept_cls = self.concept_cls.weight.unsqueeze(1).unsqueeze(0).repeat(B, 1, 1, 1)
                q_with_ccls = torch.cat([batch_concept_cls, q_topk], dim=2)
                out.append(q_with_ccls)
            else:
                out.append(q_topk)
            out_names.append('masked_q')
        if return_topk_idx:
            out.append(idx_topk)
            out_names.append('topk_idx')
        if return_concept:
            out.append(c)
            out_names.append('pred_c')
        out.append(y)
        out_names.append('pred_y')

        if return_dict:
            return OrderedDict({k: v for k, v in zip(out_names, out)})
        else:
            return out if len(out) > 1 else out[0]

    def get_concept_attribution(self, x, topk=None):
        # pred_y = self.forward(x, topk=topk)
        pass

class LitBlipCBM(LitBaseCBM):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = LitBaseCBM.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("LitBlipCBM")
        parser.add_argument("--load_linear_path", type=str, default=None)
        parser.add_argument("--freeze_linear", action="store_true")
        parser.add_argument("--load_classification_path", type=str, default=None)
        parser.add_argument("--freeze_classification", action="store_true")
        parser.add_argument("--concept_bn", action="store_true")

        return parent_parser

    @staticmethod
    def load_checkpoint_as_model(ckpt_path, strict=False):
        self = LitBlipCBM.load_from_checkpoint(ckpt_path, strict=strict)
        return self.model

    @staticmethod
    def get_processor():
        return BlipImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def initialize_model(self):
        self.model = BLIPClassifier(latent_dim=self.hparams.latent_dim, 
                                    num_classes=self.hparams.num_classes,
                                    concept_act=self.hparams.concept_act,
                                    concept_bn=hasattr(self.hparams, "concept_bn") and \
                                               (self.hparams.concept_bn))

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

class LitBlipPrefixCBM(LitBaseCBM):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = LitBaseCBM.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("LitBlipPrefixCBM")
        parser.add_argument("--num_tokens_per_concept", type=int, default=1)
        parser.add_argument("--load_linear_path", type=str, default=None)
        parser.add_argument("--freeze_linear", action="store_true")
        parser.add_argument("--load_classification_path", type=str, default=None)
        parser.add_argument("--freeze_classification", action="store_true")

        return parent_parser

    @staticmethod
    def load_checkpoint_as_model(ckpt_path, strict=False):
        self = LitBlipPrefixCBM.load_from_checkpoint(ckpt_path, strict=strict)
        return self.model

    @staticmethod
    def get_processor():
        return BlipImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def initialize_model(self):
        self.model = BLIPConceptPrefixModelV2(num_concepts=self.hparams.latent_dim,
                                              num_classes=self.hparams.num_classes,
                                              num_tokens_per_concept=self.hparams.num_tokens_per_concept)

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

class LitBlipConceptCLS(LitBaseCBM):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = LitBaseCBM.add_model_specific_args(parent_parser)
        parent_parser.add_argument("--topk", type=int, default=4)
        return parent_parser

    @staticmethod
    def load_checkpoint_as_model(ckpt_path, strict=False):
        self = LitBlipConceptCLS.load_from_checkpoint(ckpt_path, strict=strict)
        return self.model

    @staticmethod
    def get_processor():
        return BlipImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def initialize_model(self):
        self.model = BLIPConceptPrefixModelV3(num_concepts=self.hparams.latent_dim,
                                              num_classes=self.hparams.num_classes,
                                              topk=self.hparams.topk)
        return self.model

class LitBlipViT(LitBaseCBM):

    @staticmethod
    def load_checkpoint_as_model(ckpt_path, strict=False):
        self = LitBlipViT.load_from_checkpoint(ckpt_path, strict=strict)
        return self.model

    @staticmethod
    def get_processor():
        return BlipImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def initialize_model(self):
        self.model = BLIPViT(num_classes=self.hparams.num_classes)
        return self.model

