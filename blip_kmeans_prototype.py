import os
import numpy as np
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from tqdm.auto import tqdm, trange

import torch
import faiss
from transformers import BlipProcessor, BlipModel, logging

from dataset import LitAwA2DM
from blip_cbm import LitBlipCBM
from decode_cbm import BlipCBMDecoder
from utils import log_level

def find_kmeans_prototype(xs, ncentroids, niter, k=1):
    xs = xs.astype(np.float32)
    d = xs.shape[1]

    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=True, verbose=True, 
                          max_points_per_centroid=(xs.shape[0] // ncentroids) + 1)
    kmeans.train(xs)

    index_flat = faiss.IndexFlatL2(d)
    # res = faiss.StandardGpuResources()
    # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    co = faiss.GpuMultipleClonerOptions() # github.com/facebookresearch/faiss/blob/1dc992bf268ad4b14f47461922b0f70aee791f8f/faiss/gpu/test/test_multi_gpu.py#L30 
    co.shard = True
    gpu_index_flat = faiss.index_cpu_to_all_gpus(index_flat, co=co)

    gpu_index_flat.add(xs)
    D, I = gpu_index_flat.search(kmeans.centroids, k)
    return I, kmeans.centroids

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("data_src", type=str, choices=['awa2'])
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--data_root_dir", type=str, default='/home/andrewbai/data')
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--n_centroids", type=int, default=85)
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--flatten_emb", action="store_true")

    return parser.parse_args()

def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    if args.data_src == 'awa2':
        dm = LitAwA2DM(bsize=args.bsize, processor=LitBlipCBM.get_processor(), 
                       root_dir=args.data_root_dir, num_workers=args.num_workers)
        dl = dm.train_dataloader()
        # dl = dm.test_dataloader()
    else:
        raise NotImplementedError

    # load model
    with log_level(logging.ERROR):
        model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device, torch.float16)

    # encode data
    embs = []
    with torch.no_grad():
        for xs, cs, ys in tqdm(dl):
            xs = xs.to(device, torch.float16)
            embs_ = model.vision_model(**xs)[0].cpu().numpy()
            embs.append(embs_)
    embs = np.concatenate(embs, axis=0)
    if args.flatten_emb:
        embs = np.reshape(embs, (embs.shape[0], -1))

    # get kmeans centroid and nearest prototypes
    proto_iis, centroids = find_kmeans_prototype(embs, ncentroids=args.n_centroids, niter=args.n_iter)
    protos = embs[proto_iis[:, 0]]

    # decode to text
    with log_level(logging.ERROR):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        decoder = BlipCBMDecoder.from_pretrained("Salesforce/blip-image-captioning-base").to(device, torch.float16)

    with torch.no_grad():
        outputs = decoder.generate(torch.from_numpy(protos).to(device, torch.float16), return_dict=True, output_score=True)

    scores = []
    for i in range(len(outputs.sequences)):
        score = 1
        for j, id in enumerate(outputs.sequences[i][1:].cpu()):
            score *= outputs.scores[j][i].softmax(dim=0)[id].item()
        scores.append(score)

    nls = [l.strip() for l in processor.batch_decode(outputs.sequences, skip_special_tokens=True)]
    for score, nl in zip(scores, nls):
        print(nl, f"{score:.3e}")

    # save prototypes (weights in `npy`, images in folder, decoded nl descriptions in `txt`)
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        np.save(os.path.join(args.save_dir, 'weights.npy'), protos.astype(np.float32))
        with open(os.path.join(args.save_dir, 'decoded_prompts.txt'), 'w') as f:
            f.write('\n'.join(nls))

if __name__ == '__main__':
    main()

