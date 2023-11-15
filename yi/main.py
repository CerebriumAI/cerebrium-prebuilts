from typing import Optional

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class Item(BaseModel):
    prompt: str
    max_length: Optional[int] = 256
    repetition_penalty: Optional[float] = 1.3
    no_repeat_ngram_size: Optional[int] = 5
    temperature: Optional[float] = 5
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.8


model = AutoModelForCausalLM.from_pretrained(
    "01-ai/Yi-6B-200K",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-6B-200K", trust_remote_code=True)


def predict(item, run_id, logger):
    item = Item(**item)

    inputs = tokenizer(item.prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids.cuda(),
        max_length=item.max_length,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        repetition_penalty=item.repetition_penalty,
        no_repeat_ngram_size=item.no_repeat_ngram_size,
        temperature=item.temperature,
        top_k=item.top_k,
        top_p=item.top_p,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"result": result}
