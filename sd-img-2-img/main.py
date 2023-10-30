import base64
import io
from io import BytesIO
from typing import Optional

import requests
import torch
from cerebrium import get_secret
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageOps
from pydantic import BaseModel, HttpUrl


#######################################
# User-facing API Parameters
#######################################
class Item(BaseModel):
    prompt: str
    image: Optional[str] = None
    hf_token: Optional[str] = None
    hf_model_path: Optional[str] = "stabilityai/stable-diffusion-2-1"
    guidance_scale: float = 7.5
    height: int = 512
    negative_prompt: str = ""
    num_images_per_prompt: int = 1
    num_inference_steps: int = 50
    seed: int = 0
    width: int = 512
    file_url: Optional[str] = None
    webhook_endpoint: Optional[HttpUrl] = None


# Downloads a file from a given URL and saves it to a given filename
def download_file_from_url(logger, url: str, filename: str):
    logger.info("Downloading file...")

    response = requests.get(url)
    if response.status_code == 200:
        logger.info("Download was successful")

        with open(filename, "wb") as f:
            f.write(response.content)

        return filename

    else:
        logger.info(response)
        raise Exception("Download failed")


#######################################
# Prediction
#######################################
def predict(item, run_id, logger):
    params = Item(**item)
    if params.file_url is not None:
        input_filename = f"{run_id}"
        image = download_file_from_url(logger, params.file_url, input_filename)
        init_image = Image.open(image)
    elif params.image is not None:
        init_image = Image.open(BytesIO(base64.b64decode(params.image)))
    else:
        raise Exception("No image or file_url provided")

    contained_image = ImageOps.contain(init_image, (params.width, params.height))

    hf_model_path = (
        params.hf_model_path
        if bool(params.hf_model_path)
        else "runwayml/stable-diffusion-v1-5"
    )

    generator = torch.Generator("cuda").manual_seed(params.seed)
    auth_token = params.hf_token if params.hf_token else False
    if not auth_token:
        print("No hf_auth_token provided, looking for secret")
        try:
            auth_token = get_secret("hf_auth_token")
        except Exception:
            print(
                "No hf_auth_token secret found in account. Setting auth_token to False."
            )

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        hf_model_path, torch_dtype=torch.float16, use_auth_token=auth_token
    )

    pipe = pipe.to("cuda")

    images = pipe(
        params.prompt,
        image=contained_image,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        num_images_per_prompt=params.num_images_per_prompt,
        negative_prompt=params.negative_prompt,
        generator=generator,
    ).images

    finished_images = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return finished_images
