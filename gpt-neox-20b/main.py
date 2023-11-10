from typing import Optional

import torch
from pydantic import BaseModel, HttpUrl
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast


class Item(BaseModel):
    prompt: str
    temperature: float = 0.9
    max_length: int = 100
    webhook_endpoint: Optional[HttpUrl] = None


model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/gpt-neox-20b", torch_dtype=torch.float16
).cuda()
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")


def predict(item, run_id, logger):
    params = Item(**item)
    prompt = params.prompt
    temperature = params.temperature
    max_length = params.max_length

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        max_length=max_length,
    )
    result = tokenizer.batch_decode(gen_tokens)[0]

    return result
