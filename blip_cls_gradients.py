import os
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm, trange

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from captum._utils.gradient import compute_layer_gradients_and_eval

from transformers import logging, BlipProcessor, BlipModel, BlipImageProcessor

from dataset import LitAwA2DM
from utils import log_level, to_device_collate_fn, get_named_module
from blip_cbm import LitBlipViT as LitModel

def extract_blip_cls_gradients(dl, model, layer_name='classification', return_embeddings=False):

    device = next(model.parameters()).device
    layer = get_named_module(model, layer_name)

    layer_grads, acts = [], []
    for ds in tqdm(dl, leave=False):
        inputs = {k: v.to(device) for k, v in ds[0].items()}
        target_ind = ds[2].to(device)
        layer_grads_, acts_ = compute_layer_gradients_and_eval(
                model, layer, inputs, target_ind=target_ind,
                additional_forward_args=(True,), # return_probs
                attribute_to_layer_input=True)
        layer_grads.append(layer_grads_[0].detach().cpu().numpy())
        acts.append(acts_[0].detach().cpu().numpy())
    layer_grads = np.concatenate(layer_grads, axis=0)
    acts = np.concatenate(acts, axis=0)

    if return_embeddings:
        return layer_grads, acts
    else:
        return layer_grads

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("load_ckpt_path", type=str)
    parser.add_argument("save_grads_path", type=str)
    parser.add_argument("--save_embs_path", type=str, default=None)

    parser.add_argument("--bsize", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_root_dir", type=str, default="/home/andrewbai/data")

    return parser.parse_args()

def main():
    # setup
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load
    model = LitModel.load_checkpoint_as_model(args.load_ckpt_path).eval().to(device)
    dm = LitAwA2DM(root_dir=args.data_root_dir,
                   bsize=args.bsize, 
                   num_workers=args.num_workers,
                   processor=LitModel.get_processor(),
                   attr_binarize=True)
    dl = dm.val_dataloader()

    # extract
    grads, embs = extract_blip_cls_gradients(dl, model, return_embeddings=True)

    # save
    np.save(args.save_grads_path, grads)
    if args.save_embs_path is not None:
        np.save(args.save_embs_path, embs)

if __name__ == '__main__':
    main()

