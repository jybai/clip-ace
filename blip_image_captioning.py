import requests
import numpy as np
from argparse import ArgumentParser
from functools import partial
from PIL import Image
from tqdm.auto import tqdm, trange

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from dataset import LitAwA2DM

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("data_src", type=str, choices=['debug', 'awa2'])
    parser.add_argument("--save_path", type=str, default=None)

    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_root_dir", type=str, default="/home/andrewbai/data")

    return parser.parse_args()

def _decode(inputs, model, processor):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = inputs.to(device, dtype=dtype)

    generated_ids = model.generate(**inputs)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    generated_texts = [t.strip() for t in generated_texts]
    return generated_texts

def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16)
    model.to(device)
    decode = partial(_decode, model=model, processor=processor)

    if args.data_src == 'debug':
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        images = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(images=images, return_tensors="pt")
        # inputs.to(device, torch.float16)

        generated_texts = decode(inputs)
    elif args.data_src == 'awa2':
        dm = LitAwA2DM(bsize=args.bsize, processor=processor.image_processor, 
                       num_workers=args.num_workers, root_dir=args.data_root_dir)
        test_dl = dm.test_dataloader()
        generated_texts = np.concatenate([decode(xs) for xs, cs, ys in tqdm(test_dl, leave=False)], axis=0)
    else:
        raise NotImplementedError

    print(generated_texts)

    if args.save_path is not None:
        with open(args.save_path, 'w') as f:
            f.write('\n'.join(generated_texts))

if __name__ == '__main__':
    main()

