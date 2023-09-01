import base64
import io
from typing import Optional
from pydantic import BaseModel
import torch
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import PidiNetDetector, HEDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from io import BytesIO

# Special thanks to the authors of https://huggingface.co/blog/controlnet for their work on the HF implementation of ControlNet model and code.

class Item(BaseModel):
    prompt: str
    hf_token: Optional[str]
    model_id: str = "runwayml/stable-diffusion-v1-5"
    checkpoint: Optional[str] = "lllyasviel/control_v11p_sd15_softedge"
    preprocessor_name: Optional[str] = "HED"
    image: Optional[str] = None
    image_url: Optional[str] = None
    num_inference_steps: Optional[int] = 20
    height: Optional[int] = 512
    width: Optional[int] = 512
    guidance_scale: Optional[float] = 7.5
    negative_prompt: Optional[str]
    num_images_per_prompt: Optional[str] = 1
    scale: Optional[float] = 9.0
    seed: Optional[int] = 1

def download_image(image_url):
    image = Image.open(BytesIO(base64.b64decode(image_url)))
    image = image.convert("RGB")
    return image


# Some initial setup. This is run once when the model is first loaded.
# we'll change the checkpoint to the one we want to use later if needed.

checkpoint = "lllyasviel/control_v11p_sd15_softedge"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = None
controlnet = None

def setup(checkpoint, model_id):
    # if the checkpoint is different, we need to load a new model
    # first delete any existing model if it exists
    global pipe
    global controlnet
    if pipe is not None:
        del pipe
    if controlnet is not None:
        del controlnet

    # Redeclare the global variables
    global controlnet
    global pipe
    
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map="auto")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id , controlnet=controlnet, torch_dtype=torch.float16, device_map="auto"
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")

setup(checkpoint, model_id)


def predict(item, run_id, logger):
    params = Item(**item)
    if params.image is not None:
        image = Image.open(BytesIO(base64.b64decode(item.image)))
        image = image.convert("RGB")
    elif params.image_url is not None:
        image = download_image(item.image_url)
    else:
        raise Exception("No image or image_url provided")
    
    global checkpoint
    global model_id
    if((params.checkpoint is not None) and (params.checkpoint != checkpoint)) or ((params.model_id is not None) and (params.model_id != model_id)):
        logger.info("Checkpoint is different! Loading new checkpoint!")
        setup(params.checkpoint, params.model_id)
        checkpoint = params.checkpoint
        model_id = params.model_id


    image_width, image_height = image.size

    # load in the pre-processors 
    if(item.preprocessor_name=="HED"):
        processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    elif(item.preprocessor_name=="PidiNet"):
        processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
    else:
        processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    

    control_image = processor(image, safe=True)


    images = pipe(
        prompt=item.prompt,
        image=control_image,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        num_images_per_prompt=params.num_images_per_prompt,
        negative_prompt=params.negative_prompt,
        generator=torch.manual_seed(params.seed),
    ).images
    
    finished_images = []
    for image in images:
        #image = image.resize((image_width, image_height))
        buffered = io.BytesIO()
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    
    return finished_images