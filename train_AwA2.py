import os
import numpy as np
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# from clip_cbm import LitClassifier
from clip_cbm import LitClipCBM
from resnet50_cbm import LitResNet50CBM
from blip_cbm import LitBlipCBM, LitBlipPrefixCBM, LitBlipConceptCLS, LitBlipViT
from blip2_cbm import LitBlip2CBM
from dataset import LitAwA2DM

def parse_arguments():
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_root_dir", type=str, default="/home/andrewbai/data")
    parser.add_argument("--save_dir", type=str, default="lightning_logs")
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--version", type=str, default=None)

    # add model specific args
    parser.add_argument("--backbone", type=str, choices=['clip', 'resnet50', 'blip', 'blip_prefix', 
                                                         'blip_cls', 'blip_vit', 'blip2'])
    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()
    # let the model add what it wants
    if temp_args.backbone == "clip":
        LitModel = LitClipCBM
    elif temp_args.backbone == "resnet50":
        LitModel = LitResNet50CBM
    elif temp_args.backbone == "blip":
        LitModel = LitBlipCBM
    elif temp_args.backbone == "blip_prefix":
        LitModel = LitBlipPrefixCBM
    elif temp_args.backbone == 'blip_cls':
        LitModel = LitBlipConceptCLS
    elif temp_args.backbone == 'blip_vit':
        LitModel = LitBlipViT
    elif temp_args.backbone == "blip2":
        LitModel = LitBlip2CBM
    else:
        raise NotImplementedError
    parser = LitModel.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser) # max_epochs

    return parser.parse_args()

def main():
    args = parse_arguments()
    torch.set_float32_matmul_precision('high')

    args.num_classes = 50
    if args.backbone == 'clip':
        LitModel = LitClipCBM
        dm = LitAwA2DM(root_dir=args.data_root_dir,
                       bsize=args.bsize, num_workers=args.num_workers,
                       processor=LitModel.get_processor(),
                       attr_binarize=True)
    elif args.backbone == 'resnet50':
        LitModel = LitResNet50CBM
        dm = LitAwA2DM(root_dir=args.data_root_dir,
                       bsize=args.bsize, num_workers=args.num_workers,
                       transform_aug=LitModel.transform_aug,
                       transform_const=LitModel.transform_const,
                       attr_binarize=True)
    elif args.backbone.startswith('blip'):
        if args.backbone == 'blip':
            LitModel = LitBlipCBM
        elif args.backbone == 'blip_prefix':
            LitModel = LitBlipPrefixCBM
        elif args.backbone == 'blip_cls':
            LitModel = LitBlipConceptCLS
        elif args.backbone == 'blip_vit':
            LitModel = LitBlipViT
        elif args.backbone == 'blip2':
            LitModel = LitBlip2CBM
        else:
            raise NotImplementedError
        dm = LitAwA2DM(root_dir=args.data_root_dir,
                       bsize=args.bsize, num_workers=args.num_workers,
                       processor=LitModel.get_processor(),
                       attr_binarize=True)
    else:
        raise NotImplementedError

    model = LitModel(**vars(args))
    model.initialize_criterions(dm.train_dataloader())

    checkpoint_callback = ModelCheckpoint(monitor="acc_y_val", mode='max', save_last=True)

    logger = TensorBoardLogger(save_dir=args.save_dir, name=args.name, version=args.version)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], logger=logger, 
                                            accelerator='gpu', max_epochs=100) # strategy="ddp_find_unused_parameters_false")
    trainer.fit(model=model, datamodule=dm)

if __name__ == '__main__':
    main()

