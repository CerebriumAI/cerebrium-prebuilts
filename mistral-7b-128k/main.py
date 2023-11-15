from typing import Optional

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Yarn-Mistral-7b-128k",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")


class Item(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.75
    top_k: Optional[float] = 40
    max_length: Optional[int] = 256
    repetition_penalty: Optional[float] = 1.3
    no_repeat_ngram_size: Optional[int] = 5


def predict(item, run_id, logger):
    item = Item(**item)
    inputs = tokenizer(item.prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids.cuda(),
        max_length=item.max_length,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=inputs.attention_mask.cuda(),
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        repetition_penalty=item.repetition_penalty,
        no_repeat_ngram_size=item.no_repeat_ngram_size,
        temperature=item.temperature,
        top_k=item.top_k,
        top_p=item.top_p,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"result": result}
