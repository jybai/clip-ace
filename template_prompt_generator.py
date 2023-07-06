import os
import numpy as np
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("template", type=str)
    parser.add_argument("load_keywords_txt", type=str)
    parser.add_argument("save_prompts_txt", type=str)
    parser.add_argument("--keyword", type=str, default="KEYWORD")

    return parser.parse_args()

def main():
    args = parse_arguments()
    assert args.keyword in args.template

    with open(args.load_keywords_txt, 'r') as f:
        keywords = f.read().splitlines()
        keywords = np.array([k.split('\t')[-1] for k in keywords])

    prompts = []
    for keyword in keywords:
        prompt = args.template.replace(args.keyword, keyword)
        prompts.append(prompt)

    with open(args.save_prompts_txt, 'w') as f:
        f.write('\n'.join(prompts))

if __name__ == '__main__':
    main()

