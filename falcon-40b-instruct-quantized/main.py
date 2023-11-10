from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from pydantic import BaseModel, HttpUrl
from typing import Optional

model_name_or_path = "TheBloke/falcon-40b-instruct-GPTQ"
model_basename = "model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
use_triton = False
model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    model_basename=model_basename,
    use_safetensors=True,
    trust_remote_code=True,
    device="cuda:0",
    use_triton=use_triton,
    quantize_config=None,
)



class Item(BaseModel):
    prompt: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 10
    repetition_penalty: Optional[float] = 1.0
    num_return_sequences: Optional[int] = 1
    webhook_endpoint: Optional[HttpUrl] = None


def predict(item, run_id, logger):
    params = Item(**item)

    prompt = params.prompt
    prompt_template = f"""A helpful assistant who helps the user with any questions asked.\n User: {prompt}\n Assistant:"""

    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.cuda()
    output = model.generate(
        inputs=input_ids,
        temperature=params.temperature,
        max_new_tokens=params.max_length,
        top_p=params.top_p,
        top_k=params.top_k,
        repetition_penalty=params.repetition_penalty,
    )
    result = tokenizer.decode(output[0])

    result = result.replace(prompt_template, "")

    return result
