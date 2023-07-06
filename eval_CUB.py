from argparse import ArgumentParser
import pytorch_lightning as pl

from clip_cbm import CLIPClassifier, LitClassifier
from dataset import LitCubaDM

def parse_arguments():
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--split", type=str, default="val", choices=['val', 'test'])
    parser.add_argument("--data_root_dir", type=str, default="/home/andrewbai/data")

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def main():
    args = parse_arguments()

    model = LitClassifier.load_from_checkpoint(args.ckpt_path)

    dm = LitCubaDM(root_dir=args.data_root_dir, return_class=True,
                   bsize=args.bsize, num_workers=args.num_workers)

    if args.split == 'val':
        dl = dm.val_dataloader()  
    elif args.split == 'test':
        dl = dm.test_dataloader()
    else:
        raise NotImplementedError

    trainer = pl.Trainer.from_argparse_args(args, logger=False, 
                                            accelerator='gpu', devices=1)
    trainer.test(model=model, dataloaders=dl)

if __name__ == '__main__':
    main()

