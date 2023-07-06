import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import os

import torch
import torch.nn as nn

from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    logging
)

from dataset import LitAwA2DM
from utils import log_level

class BlipForConditionalGenerationCLS(BlipForConditionalGeneration):
    # https://github.com/huggingface/transformers/blob/7bce8042606b01d0dba3b83d5d606f47529d6ba4/src/transformers/models/blip/modeling_blip.py#L1034
    @torch.no_grad()
    def generate_cls(self, pixel_values, input_ids=None, attention_mask=None, 
                     image_attention_mask=None, **generate_kwargs,):
        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        image_embeds = vision_outputs[0]

        if image_attention_mask is None:
            image_attention_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long).to(image_embeds.device)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.decoder_input_ids, self.config.text_config.eos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )

        input_ids[:, 0] = self.config.text_config.bos_token_id
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        return outputs

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("--x_shift", type=int, default=0)
    parser.add_argument("--y_shift", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=24)
    parser.add_argument("--patch_size", type=int, default=24)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--n_batches", type=int, default=1)
    parser.add_argument("--data_root_dir", type=str, default='/home/andrewbai/data')
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--mask_cls", action="store_true")

    return parser.parse_args()

def main():

    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert args.x_shift + args.crop_size <= args.patch_size
    assert args.y_shift + args.crop_size <= args.patch_size

    with log_level(logging.ERROR):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        decoder = BlipForConditionalGenerationCLS.from_pretrained("Salesforce/blip-image-captioning-base").to(device, torch.float16)

    dm = LitAwA2DM(bsize=args.bsize, processor=processor.image_processor, 
                   root_dir=args.data_root_dir, num_workers=args.num_workers)
    dl = dm.test_dataloader()

    image_attention_mask = torch.zeros((args.bsize, args.patch_size**2 + 1), dtype=torch.long).to(device)
    if not args.mask_cls:
        image_attention_mask[:, 0] = 1 # CLS
    for j in range(args.y_shift, args.y_shift + args.crop_size):
        image_attention_mask[:, 1 + args.x_shift + j * args.patch_size:1 + args.x_shift + j * args.patch_size + args.crop_size] = 1
    for i in range(args.patch_size):
        print(image_attention_mask[0, 1 + i * args.patch_size:1 + (i + 1) * args.patch_size].detach().cpu().numpy())

    if args.prompt is None:
        input_ids = None
    else:
        input_ids = processor.tokenizer(args.prompt, return_tensors='pt')['input_ids'].repeat(args.bsize, 1).to(device)  

    nls, nls_cls = [], []
    with torch.no_grad():
        for i, (xs, cs, ys) in enumerate(tqdm(dl)):
            xs = xs.to(device, torch.float16)

            output_ids_cls = decoder.generate_cls(**xs, image_attention_mask=image_attention_mask, input_ids=input_ids)
            nls_cls += [l.strip() for l in processor.batch_decode(output_ids_cls, skip_special_tokens=True)]

            output_ids = decoder.generate(**xs)
            nls += [l.strip() for l in processor.batch_decode(output_ids, skip_special_tokens=True)]

            if i + 1 >= args.n_batches:
                break

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'decoded_prompts.txt'), 'w') as f:
            f.write('\n'.join(nls))
        with open(os.path.join(args.save_dir, 'decoded_prompts_cls.txt'), 'w') as f:
            f.write('\n'.join(nls_cls))

if __name__ == '__main__':
    main()

