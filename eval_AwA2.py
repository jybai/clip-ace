from argparse import ArgumentParser
import torch
import pytorch_lightning as pl

from clip_cbm import LitClipCBM
from resnet50_cbm import LitResNet50CBM
from blip_cbm import LitBlipCBM, LitBlipPrefixCBM
from dataset import LitAwA2DM

def parse_arguments():
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--split", type=str, default="val", choices=['val', 'test'])
    parser.add_argument("--data_root_dir", type=str, default="/home/andrewbai/data")

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def main():
    args = parse_arguments()
    torch.set_float32_matmul_precision('high')

    if args.backbone is None:
        args.backbone = torch.load(args.ckpt_path)['hyper_parameters']['backbone']

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
    elif args.backbone == 'blip':
        LitModel = LitBlipCBM
        dm = LitAwA2DM(root_dir=args.data_root_dir,
                       bsize=args.bsize, num_workers=args.num_workers,
                       processor=LitModel.get_processor(),
                       attr_binarize=True)
    elif args.backbone == 'blip_prefix':
        LitModel = LitBlipPrefixCBM
        dm = LitAwA2DM(root_dir=args.data_root_dir,
                       bsize=args.bsize, num_workers=args.num_workers,
                       processor=LitModel.get_processor(),
                       attr_binarize=True)

    elif args.backbone == 'blip2':
        LitModel = LitBlip2CBM
        dm = LitAwA2DM(root_dir=args.data_root_dir,
                       bsize=args.bsize, num_workers=args.num_workers,
                       processor=LitModel.get_processor(),
                       attr_binarize=True)
    else:
        raise NotImplementedError

    model = LitModel.load_from_checkpoint(args.ckpt_path, strict=False)

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

