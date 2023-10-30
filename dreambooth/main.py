import base64
import io
from typing import Optional

import torch
from cerebrium import get_secret
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from pydantic import BaseModel, HttpUrl


#######################################
# User-facing API Parameters
#######################################
class Item(BaseModel):
    prompt: str
    hf_token: Optional[str] = None
    hf_model_path: Optional[str] = None
    guidance_scale: float = 7.5
    height: int = 512
    negative_prompt: str = ""
    num_images_per_prompt: int = 1
    num_inference_steps: int = 50
    seed: Optional[int] = 1
    width: int = 512
    webhook_endpoint: Optional[HttpUrl] = None


#######################################
# Model Setup
#######################################

old_hf_model_path = "stabilityai/stable-diffusion-2-1"
scheduler = EulerDiscreteScheduler.from_pretrained(
    old_hf_model_path, subfolder="scheduler"
)
pipe = StableDiffusionPipeline.from_pretrained(
    old_hf_model_path, scheduler=scheduler, torch_dtype=torch.float16
)


def run_model(pipe, params, logger):
    if params.seed is not None and int(params.seed) > 0:
        logger.info("Manual seed")
        generator = torch.Generator("cuda").manual_seed(int(params.seed))
    else:
        logger.info("Random seed")
        generator = torch.Generator("cuda")

    logger.info("Seed: {}".format(params.seed))

    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to("cuda")

    images = pipe(
        params.prompt,
        height=params.height,
        width=params.width,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        num_images_per_prompt=params.num_images_per_prompt,
        negative_prompt=params.negative_prompt,
        generator=generator,
    ).images

    return images


########################################
# Prediction
########################################
def predict(item, run_id, logger):
    params = Item(**item)
    hf_model_path = (
        params.hf_model_path
        if bool(params.hf_model_path)
        else "stabilityai/stable-diffusion-2-1"
    )
    global scheduler
    global pipe
    global old_hf_model_path

    if hf_model_path == old_hf_model_path:
        logger.info("No change in model path. Using existing model...")
        images = run_model(pipe=pipe, params=params, logger=logger)
    else:
        auth_token = params.hf_token if params.hf_token else False
        if not auth_token:
            print("No hf_auth_token provided, looking for secret")
            try:
                auth_token = get_secret("hf_auth_token")
            except Exception:
                print(
                    "No hf_auth_token secret found in account. Setting auth_token to False."
                )
        scheduler = EulerDiscreteScheduler.from_pretrained(
            hf_model_path, subfolder="scheduler", use_auth_token=auth_token
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            hf_model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
        )
        images = run_model(pipe=pipe, params=params, logger=logger)
        old_hf_model_path = hf_model_path

    finished_images = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return finished_images
