
from typing import Optional
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
import base64
import io


class Item(BaseModel):
    # Add your input parameters here
    prompt: str
    guidance_scale: Optional[float] = 3.0 # an example optional parameter

pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2-1024px-aesthetic",
    torch_dtype=torch.float16,
    use_safetensors=True,
    add_watermarker=False,
    variant="fp16"
)
pipe.to("cuda")

def predict(item, run_id, logger):
    item = Item(**item)

    image  = pipe(prompt=item.prompt, guidance_scale=item.guidance_scale).images[0]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")

    return {"image": base64.b64encode(buffered.getvalue()).decode("utf-8")} 
