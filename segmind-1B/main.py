from typing import Optional
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline
import torch
import io
import base64


class Item(BaseModel):
    prompt: str
    height: Optional[int] = 250
    width: Optional[int] = 250
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    negative_prompt: Optional[str]
    num_images_per_prompt: Optional[int] = 1


pipe = StableDiffusionXLPipeline.from_pretrained(
    "segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
pipe.to("cuda")


def predict(item, run_id, logger):
    item = Item(**item)

    pipe(
        prompt=item.prompt,
        negative_prompt=item.negative_prompt,
        height=item.height,
        width=item.width,
        guidance_scale=item.guidance_scale,
        num_images_per_prompt=item.num_images_per_prompt,
        num_inference_steps=item.num_inference_steps,
    ).images

    images = pipe(prompt=item.prompt, negative_prompt=item.negative_prompt).images
    finished_images = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return finished_images
