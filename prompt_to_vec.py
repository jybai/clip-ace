import os
import numpy as np
from argparse import ArgumentParser

import torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, logging
from utils import log_level

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("load_prompt_path", type=str)
    parser.add_argument("save_emb_path", type=str)

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read the prompts
    with open(args.load_prompt_path, 'r') as f:
        prompts = f.read().splitlines()
    # pass it into a CLIP text encoder to embed
    with log_level(logging.ERROR):
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
        text_embs = outputs.text_embeds.cpu().numpy()
        text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)
    # save the embeddings
    np.save(args.save_emb_path, text_embs)

if __name__ == '__main__':
    main()

