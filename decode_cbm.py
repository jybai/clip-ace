import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    BlipProcessor, 
    BlipForConditionalGeneration
)

# from clip_cbm import LitClassifier
from clip_cbm import LitClipCBM
from blip_cbm import LitBlipCBM, LitBlipPrefixCBM
from resnet50_cbm import LitResNet50CBM

class MLP(nn.Module):
# {{{
    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
# }}}

class ClipCaptionModel(nn.Module):
# {{{
    def __init__(self, prefix_length: int, prefix_size: int = 512):# {{{
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length
            )
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )

    def get_dummy_token(self, batch_size, device):
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(self, tokens, prefix, mask=None, labels=None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out
# }}}

def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):
    # {{{
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts
    # }}}

class BlipCBMDecoder(BlipForConditionalGeneration):
# {{{
    def decode(self, cavs, input_ids=None):
        batch_size = cavs.shape[0]
        cavs = torch.reshape(cavs, (batch_size, -1, self.config.vision_config.hidden_size))
        image_attention_mask = torch.ones(cavs.shape[:-1], dtype=torch.long).to(cavs.device)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.decoder_input_ids, self.config.text_config.eos_token_id]])
                .repeat(batch_size, 1)
                .to(cavs.device)
            )
        input_ids[:, 0] = self.config.text_config.bos_token_id


        outputs = self.text_decoder(
            input_ids=input_ids[:, :-1],
            encoder_hidden_states=cavs,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )

        return outputs

    # https://github.com/huggingface/transformers/blob/7bce8042606b01d0dba3b83d5d606f47529d6ba4/src/transformers/models/blip/modeling_blip.py#L1034
    @torch.no_grad()
    def generate(self, cavs, input_ids=None, attention_mask=None, **generate_kwargs,):
        batch_size = cavs.shape[0]
        cavs = torch.reshape(cavs, (batch_size, -1, self.config.vision_config.hidden_size))

        image_attention_mask = torch.ones(cavs.shape[:-1], dtype=torch.long).to(cavs.device)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.decoder_input_ids, self.config.text_config.eos_token_id]])
                .repeat(batch_size, 1)
                .to(cavs.device)
            )

        input_ids[:, 0] = self.config.text_config.bos_token_id
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=attention_mask,
            encoder_hidden_states=cavs,
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        return outputs
# }}}

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("clipcbm_path", type=str)
    parser.add_argument("caption_arch", type=str, choices=['clipcap', 'blip'])
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--normalize", type=bool, default=True)

    # use default or decode
    parser.add_argument("--prompts_txt_path", type=str, default=None)

    # clipcap args
    parser.add_argument("--caption_model_path", type=str, default=None)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--temperature", type=float, default=1.0)

    # per-class printing args
    parser.add_argument("--class_names_txt_path", type=str, default=None)
    parser.add_argument("--topk", type=int, default=5)

    return parser.parse_args()

def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backbone is None:
        args.backbone = torch.load(args.clipcbm_path)['hyper_parameters']['backbone']

    if args.backbone == 'resnet50':
        LitModel = LitResNet50CBM
    elif args.backbone == 'clip':
        LitModel = LitClipCBM
    elif args.backbone == 'blip':
        LitModel = LitBlipCBM
    elif args.backbone == "blip_prefix":
        LitModel = LitBlipPrefixCBM
    else:
        raise NotImplementedError
    clipcbm = LitModel.load_checkpoint_as_model(args.clipcbm_path)
    clipcbm.eval().to(device)

    with torch.no_grad():
        concept_cavs = clipcbm.linear.weight
        if args.normalize:
            concept_cavs /= concept_cavs.norm(-1, keepdim=True)

    if args.caption_arch == 'clipcap':
        processor = GPT2Tokenizer.from_pretrained("gpt2")
        caption_model = ClipCaptionModel(prefix_length=args.prefix_length)
        caption_model.load_state_dict(torch.load(args.caption_model_path))
    elif args.caption_arch == 'blip':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        caption_model = BlipCBMDecoder.from_pretrained(
                "Salesforce/blip-image-captioning-base") #, torch_dtype=torch.float16)
    else:
        raise NotImplementedError
    caption_model.eval().to(device)

    if args.prompts_txt_path is not None:
        with open(args.prompts_txt_path, 'r') as f:
            concept_nls = np.array(f.read().splitlines())
    else:
        concept_nls, scores = [], []
        if args.caption_arch == 'clipcap':
            with torch.no_grad():
                for i, concept_cav in enumerate(concept_cavs):
                    concept_prefix = caption_model.clip_project(concept_cav).reshape(1, args.prefix_length, -1)
                    concept_nls_ = generate_beam(caption_model, processor, embed=concept_prefix, 
                                                 beam_size=args.beam_size, entry_length=args.entry_length, 
                                                 temperature=args.temperature)
                    concept_nls_ = [nl.strip() for nl in concept_nls_] # remove leading/trailing white spaces.
                    concept_nls.append(concept_nls_[0])
        elif args.caption_arch == 'blip':
            outputs = caption_model.generate(concept_cavs, output_scores=True, return_dict_in_generate=True)
            '''
            print(outputs.sequences.shape)
            print(len(outputs.scores), outputs.scores[0].shape)
            for i in range(len(outputs.sequences)):
                if not torch.equal(outputs.sequences[i][1:].cpu(), torch.tensor([s[i].argmax() for s in outputs.scores])):
                    print(outputs.sequences[i][1:].cpu())
                    print(torch.tensor([s[i].argmax() for s in outputs.scores]))
            print(caption_model.config.text_config.sep_token_id, 
                  caption_model.config.text_config.pad_token_id,
                  caption_model.config.text_config.bos_token_id,
                  caption_model.config.text_config.decoder_start_token_id)
            '''
            concept_nls = [l.strip() for l in processor.batch_decode(outputs.sequences, skip_special_tokens=True)]
            for i in range(len(outputs.sequences)):
                score = 1
                for j, id in enumerate(outputs.sequences[i][1:].cpu()):
                    score *= outputs.scores[j][i].softmax(dim=0)[id].item()
                scores.append(score)
        else:
            raise NotImplementedError
        concept_nls = np.array(concept_nls)
        scores = np.array(scores)

    # print out the decoded natural language descriptions
    for concept_nl, score in zip(concept_nls, scores):
        print(concept_nl, f"{score:.2e}")

    # print out topk decoded natural language descriptions for each class
    if args.class_names_txt_path is not None:
        with open(args.class_names_txt_path, 'r') as f:
            class_names = f.read().splitlines()
            class_names = np.array([c.split('\t')[-1] for c in class_names])

        class_weights = clipcbm.classification.weight.detach().cpu().numpy()
        for i, class_name in enumerate(class_names):
            class_weight = class_weights[i]
            order = np.argsort(-class_weight)
            print(class_name)
            for j in range(args.topk):
                print(j, concept_nls[order][j], class_weight[order][j])

if __name__ == '__main__':
    main()

