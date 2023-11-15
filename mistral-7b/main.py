from typing import Optional

from pydantic import BaseModel
from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mistral-7B-v0.1",
    dtype="bfloat16",
)


class Item(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.75
    top_k: Optional[float] = 40
    max_tokens: Optional[int] = 256
    frequency_penalty: Optional[int] = 1


def predict(item, run_id, logger):
    item = Item(**item)
    sampling_params = SamplingParams(
        temperature=item.temperature,
        top_p=item.top_p,
        top_k=item.top_k,
        max_tokens=item.max_tokens,
        frequency_penalty=item.frequency_penalty,
    )
    outputs = llm.generate([item.prompt], sampling_params)

    generated_text = []
    for output in outputs:
        generated_text.append(output.outputs[0].text)
    return {"result": generated_text}
