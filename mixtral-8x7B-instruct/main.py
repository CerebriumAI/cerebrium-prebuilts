
from typing import Optional
from pydantic import BaseModel
from vllm import LLM
import torch

class Item(BaseModel):
    prompt: str

llm = LLM("mistralai/Mixtral-8x7B-Instruct-v0.1",dtype="bfloat16",tensor_parallel_size=torch.cuda.device_count())

def predict(item, run_id, logger):
    item = Item(**item)

    output = llm.generate(item.prompt)

    return {"result": output}