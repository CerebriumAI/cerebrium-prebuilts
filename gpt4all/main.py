from typing import Optional
from pydantic import BaseModel
from gpt4all import GPT4All


class Item(BaseModel):
    # Add your input parameters here
    prompt: str
    max_tokens: Optional[int] = 250  # an example optional parameter
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.4
    repeat_penalty: Optional[float] = 1.18


model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")


def predict(item, run_id, logger):
    item = Item(**item)

    output = model.generate(
        item.prompt,
        max_tokens=item.max_tokens,
        temp=item.temperature,
        top_k=item.top_k,
        top_p=item.top_p,
        repeat_penalty=item.repeat_penalty,
    )
    return {"result": output}
