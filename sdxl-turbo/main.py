
from typing import Optional
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image
import torch
import base64
import io

class Item(BaseModel):
    # Add your input parameters here
    prompt: str
    num_inference_steps: Optional[int] = 25
    guidance_scale: Optional[float] = 3.0

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

def predict(item, run_id, logger):
    item = Item(**item)

    image = pipe(prompt=item.prompt, num_inference_steps=item.num_inference_steps, guidance_scale=item.guidance_scale).images[0]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")

    return {"image": base64.b64encode(buffered.getvalue()).decode("utf-8")} 
