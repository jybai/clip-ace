import numpy as np
from functools import partial
from matplotlib import pyplot as plt

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

class MyBlipForConditionalGeneration(BlipForConditionalGeneration):
    # https://github.com/huggingface/transformers/blob/7bce8042606b01d0dba3b83d5d606f47529d6ba4/src/transformers/models/blip/modeling_blip.py#L1034
    @torch.no_grad()
    def generate_image_embeddings(self, image_embeds, input_ids=None, attention_mask=None, **generate_kwargs,):
        batch_size = image_embeds.shape[0]
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

