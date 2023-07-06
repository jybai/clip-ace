import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from clip_cbm import LitClassifier
from dataset import LitCifar10DM

def parse_arguments():
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_root_dir", type=str, default="/home/andrewbai/data")

    # add model specific args
    parser = LitClassifier.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser) # max_epochs

    return parser.parse_args()

def main():

    args = parse_arguments()

    model = LitClassifier(**vars(args))

    dm = LitCifar10DM(root=args.data_root_dir, 
                      bsize=args.bsize, num_workers=args.num_workers)

    checkpoint_callback = ModelCheckpoint(monitor="acc_val", mode='max', save_last=True)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], 
                                            accelerator='gpu', devices=1, max_epochs=100)
    trainer.fit(model=model, datamodule=dm)

if __name__ == '__main__':
    main()

