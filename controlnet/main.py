import base64
import io
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import torch
from cerebrium import get_secret
from controlnet_aux import HEDdetector, MLSDdetector, OpenposeDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image
from pydantic import BaseModel, HttpUrl
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, pipeline


class Item(BaseModel):
    prompt: str
    hf_token: Optional[str] = None
    hf_model_path: Optional[str] = None
    num_inference_steps: Optional[int] = 20
    height: Optional[int] = 512
    width: Optional[int] = 512
    guidance_scale: Optional[float] = 7.5
    negative_prompt: Optional[str] = None
    num_images_per_prompt: Optional[int] = 1
    low_threshold: Optional[int] = 100
    high_threshold: Optional[int] = 200
    scale: Optional[float] = 9.0
    seed: Optional[int] = 1
    model: str = "normal"
    image: Optional[str] = None
    file_url: Optional[str] = None
    webhook_endpoint: Optional[HttpUrl] = None


# Downloads a file from a given URL and saves it to a given filename
def download_file_from_url(logger, url: str, filename: str):
    logger.info("Downloading file...")

    import requests

    response = requests.get(url)
    if response.status_code == 200:
        logger.info("Download was successful")

        with open(filename, "wb") as f:
            f.write(response.content)

        return filename

    else:
        logger.info(response)
        raise Exception("Download failed")


openposeDetector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
    "openmmlab/upernet-convnext-small"
)


def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]


#######################################
# Prediction
#######################################
def predict(item, run_id, logger):
    params = Item(**item)
    logger.info("Downloading file...")
    if params.file_url is not None:
        input_filename = f"{run_id}"
        image = download_file_from_url(logger, params.file_url, input_filename)
        init_image = Image.open(image)
    elif params.image is not None:
        init_image = Image.open(BytesIO(base64.b64decode(params.image)))
    else:
        raise Exception("No image or file_url provided")

    logger.info("Running ControlNet...")

    if params.model == "canny":
        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-canny",
            torch_dtype=torch.float16,
            cache_dir="/persistent-storage",
        )

        image = np.array(init_image)

        image = cv2.Canny(image, params.low_threshold, params.high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

    elif params.model == "openpose":
        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-openpose",
            torch_dtype=torch.float16,
            cache_dir="/persistent-storage",
        )
        image = openposeDetector(init_image)

    elif params.model == "depth":
        depth_estimator = pipeline("depth-estimation")
        image = depth_estimator(init_image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-depth",
            torch_dtype=torch.float16,
            cache_dir="/persistent-storage",
        )

    elif params.model == "hed":
        image = hed(init_image)
        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-hed",
            torch_dtype=torch.float16,
            cache_dir="/persistent-storage",
        )

    elif params.model == "mlsd":
        image = mlsd(init_image)
        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-mlsd",
            torch_dtype=torch.float16,
            cache_dir="/persistent-storage",
        )

    elif params.model == "normal":
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        image = depth_estimator(init_image)["predicted_depth"][0]

        image = image.numpy()

        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)

        bg_threshold = 0.4

        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threshold] = 0

        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threshold] = 0

        z = np.ones_like(x) * np.pi * 2.0

        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image**2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-normal",
            torch_dtype=torch.float16,
            cache_dir="/persistent-storage",
        )

    elif params.model == "scribble":
        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-scribble",
            torch_dtype=torch.float16,
            cache_dir="/persistent-storage",
        )

        image = hed(init_image, scribble=True)

    elif params.model == "seg":
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
        )

        pixel_values = image_processor(
            init_image.convert("RGB"), return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            outputs = image_segmentor(pixel_values)

        seg = image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[init_image.size[::-1]]
        )[0]

        color_seg = np.zeros(
            (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
        )  # height, width, 3

        palette = np.array(ade_palette())

        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        color_seg = color_seg.astype(np.uint8)

        image = Image.fromarray(color_seg)

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

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        hf_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_auth_token=auth_token,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    images = pipe(
        params.prompt,
        image,
        height=params.height,
        width=params.width,
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
