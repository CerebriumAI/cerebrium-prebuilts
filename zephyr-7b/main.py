
from typing import Optional
from pydantic import BaseModel
import torch
from transformers import pipeline

class Item(BaseModel):
    # Add your input parameters here
    prompt: str
    messages: list
    max_length: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")

def predict(item, run_id, logger):
    item = Item(**item)
    
    messages = item.messages
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=item.max_length, do_sample=True, temperature=item.temperature, top_k=item.top_k, top_p=item.top_p)

    return {"result": outputs[0]["generated_text"]} # return your results 
