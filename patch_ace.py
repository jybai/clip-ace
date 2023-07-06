import os
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm, trange

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from transformers import logging, BlipProcessor, BlipModel, BlipImageProcessor

from dataset import LitAwA2DM
from utils import log_level, to_device_collate_fn
from blip_kmeans_prototype import find_kmeans_prototype
from blip_utils import MyBlipForConditionalGeneration

def extract_blip_image_embeddings(dl, device):
    # images: list of PIL images

    with log_level(logging.ERROR):
        model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    embs = []
    for ds in tqdm(dl, leave=False):
        inputs = {k: v.to(device) for k, v in ds[0].items()}
        with torch.no_grad():
            embs_ = model.vision_model(**inputs, return_dict=True)['last_hidden_state'] 
            embs.append(embs_.detach().cpu().numpy())
    embs = np.concatenate(embs, axis=0)
    return embs

def decode_token_embeddings(embs, device, **generate_kwargs):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    decoder = MyBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    embs = torch.from_numpy(embs).unsqueeze(dim=1).to(device)
    output_ids = decoder.generate_image_embeddings(embs, return_dict=False, **generate_kwargs)
    nls = [l.strip() for l in processor.batch_decode(output_ids, skip_special_tokens=True)]

    return nls

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("save_embs_path", type=str)
    parser.add_argument("save_clusters_path", type=str)
    parser.add_argument("save_decoded_path", type=str)

    parser.add_argument("--save_nn_path", type=str, default=None)

    parser.add_argument("--bsize", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_root_dir", type=str, default="/home/andrewbai/data")
    parser.add_argument("--n_samples", type=int, default=None)

    parser.add_argument("--nc", type=int, default=64)
    parser.add_argument("--kmeans_iter", type=int, default=50)
    parser.add_argument("--topk", type=int, default=16)


    return parser.parse_args()

def main():
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.isfile(args.save_embs_path):
        embs = np.load(args.save_embs_path)
    else:
        processor = BlipImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        dm = LitAwA2DM(root_dir=args.data_root_dir,
                       bsize=args.bsize, 
                       num_workers=args.num_workers,
                       processor=processor,
                       attr_binarize=True)

        if args.n_samples is not None:
            dl = dm.val_dataloader(subset_indices=np.arange(args.n_samples))
        else:
            dl = dm.val_dataloader()
        # extract all embedding tokens (N * N_PATCH^2)
        embs = extract_blip_image_embeddings(dl, device=device)
        np.save(args.save_embs_path, embs)
        embs = embs[:, 1:] # remove CLS token

    embs = np.reshape(embs, (-1, embs.shape[-1]))
    print(embs.shape)

    # run kmeans on tokens
    topk_nn_idx, centroids = find_kmeans_prototype(embs, args.nc, args.kmeans_iter, args.topk)
    print(topk_nn_idx.shape, centroids.shape)
    np.save(args.save_clusters_path, centroids)

    # save results (tokens, mappings of tokens, cluster centers)

    # decode cluster center + topk nearest neighbor in cluster
    generate_kwargs = dict(max_new_tokens=5, repetition_penalty=1.0)
    nls = decode_token_embeddings(centroids, device, **generate_kwargs)

    with open(args.save_decoded_path, 'w') as f:
        f.write('\n'.join(nls))

    nn_nls = []
    for idx in topk_nn_idx:
        embs_ = embs[idx]
        nls_ = decode_token_embeddings(embs_, device, **generate_kwargs)
        nn_nls.append(nls_)

    if args.save_nn_path is not None:
        with open(args.save_nn_path, 'w') as f:
            for nl, nn_nls_ in zip(nls, nn_nls):
                f.write(f"cluster: {nl}\n")
                for i, nn_nl in enumerate(nn_nls_):
                    f.write(f"{i + 1} {nn_nl}\n")

if __name__ == '__main__':
    main()

