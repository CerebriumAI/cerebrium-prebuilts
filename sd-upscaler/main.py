import base64
import io
from io import BytesIO
from typing import Optional

import requests
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionUpscalePipeline
from pydantic import BaseModel, HttpUrl


class Item(BaseModel):
    image: Optional[str] = None
    file_url: Optional[str] = None
    prompt: str = ""
    height: int = 250
    width: int = 250
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    num_images_per_prompt: int = 1
    webhook_endpoint: Optional[HttpUrl] = None


hf_model_path = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    hf_model_path, torch_dtype=torch.float16, revision="fp16"
)
pipeline.set_use_memory_efficient_attention_xformers(True)
pipeline = pipeline.to("cuda")


# Downloads a file from a given URL and saves it to a given filename
def download_file_from_url(logger, url: str, filename: str):
    logger.info(f"Downloading file {url}")
    response = requests.get(url)
    if response.status_code == 200:
        logger.info("Download was successful")

        with open(filename, "wb") as f:
            f.write(response.content)

        return filename

    else:
        logger.info(response)
        raise Exception("Download failed")


def predict(item, run_id, logger):
    params = Item(**item)
    if params.file_url is not None:
        input_filename = f"{run_id}"
        image = download_file_from_url(logger, params.file_url, input_filename)
        init_image = Image.open(image)
    else:
        init_image = Image.open(BytesIO(base64.b64decode(params.image)))

    if params.width != 250 or params.height != 250:
        print("Image being resized")
        print(
            "NB: Please note the maximum width and height the model can take is 510x510"
        )
        init_image = ImageOps.contain(init_image, (params.width, params.height))

    images = pipeline(
        prompt=params.prompt,
        image=init_image
        # num_inference_steps=params.num_inference_steps,
        # guidance_scale=params.guidance_scale,
        # num_images_per_prompt=params.num_images_per_prompt,
        # negative_prompt=params.negative_prompt
    ).images

    finished_images = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return finished_images
