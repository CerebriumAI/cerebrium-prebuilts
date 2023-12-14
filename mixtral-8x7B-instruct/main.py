
from typing import Optional
from pydantic import BaseModel
from vllm import LLM
import torch

class Item(BaseModel):
    prompt: str

llm = LLM("mistralai/Mixtral-8x7B-Instruct-v0.1",tensor_parallel_size=torch.cuda.device_count(),trust_remote_code=True)

def predict(item, run_id, logger):
    item = Item(**item)

    output = llm.generate(item.prompt)

    return {"result": output}