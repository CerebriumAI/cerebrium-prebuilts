from typing import Optional
from pydantic import BaseModel, HttpUrl
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Item(BaseModel):
    prompt: str
    webhook_endpoint: Optional[HttpUrl] = None


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-xl", device_map="auto"
)


def predict(item, run_id, logger):
    params = Item(**item)
    prompt = params.prompt

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result
