from transformers import AutoTokenizer
import transformers
import torch
from pydantic import BaseModel, HttpUrl
from typing import Optional

class Item(BaseModel):
    prompt: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 10
    repetition_penalty: Optional[float] = 1.0
    num_return_sequences: Optional[int] = 1
    webhook_endpoint: Optional[HttpUrl] = None


model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def predict(item, run_id, logger):
    params = Item(**item)
    result = pipeline(
        params.prompt,
        top_k=params.top_k,
        num_return_sequences=params.num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        max_length=params.max_length,
        do_sample=True,
        temperature=params.temperature,
        top_p=params.top_p,
        repetition_penalty=params.repetition_penalty,
    )
    return result
