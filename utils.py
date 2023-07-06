import numpy as np
from tqdm.auto import tqdm, trange

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.functional.text.perplexity import _perplexity_update

from transformers import logging

def _greedy_perplexity_update(preds, ignore_index=-100):
    target = torch.argmax(preds, dim=-1)
    total_log_probs, count = _perplexity_update(preds, target, ignore_index)
    return total_log_probs, count

class GreedyPerplexity(torchmetrics.text.perplexity.Perplexity):
    def update(self, preds):
        total_log_probs, count = _greedy_perplexity_update(preds, self.ignore_index)
        self.total_log_probs += total_log_probs
        self.count += count

def to_device_collate_fn(device):
    return lambda x: tuple(x_.to(device) for x_ in default_collate(x))

def get_named_module(model, name):
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    raise ValueError(f"{name} not found in model.")

def state_loader(model, path):
    ckpt_data = torch.load(path)
    state_dict = {k[len("model."):]: v for k, v in ckpt_data["state_dict"].items()}
    model.load_state_dict(state_dict)
    return ckpt_data["optimizer_states"][0]['param_groups'][0]['lr']

class log_level:
    # https://github.com/huggingface/transformers/issues/5421#issuecomment-1317784733
    orig_log_level: int
    log_level: int
    def __init__(self, log_level: int, *args, **kwargs):
        self.log_level = log_level
        self.orig_log_level = logging.get_verbosity()
    def __enter__(self, *args, **kwargs):
        logging.set_verbosity(self.log_level)
    def __exit__(self, *args, **kwargs):
        logging.set_verbosity(self.orig_log_level)

