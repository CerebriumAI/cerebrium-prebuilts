import base64
import os
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import requests
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from pydantic import BaseModel, HttpUrl
from realesrgan import RealESRGANer

SOURCE = os.environ.get("SOURCE", "cortex")
APP_NAME = os.environ.get("APP_NAME", "eseragan")


#######################################
# User-facing API Parameters
#######################################
class Item(BaseModel):
    image: Optional[str] = None
    file_url: Optional[str] = None
    upscale: Optional[int] = 4
    face_enhance: Optional[bool] = False
    webhook_endpoint: Optional[HttpUrl] = None


#######################################
# Model Setup
#######################################


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
        logger.info(response.text)
        raise Exception(f"Download failed. Response from URL was: \nStatus Code:{response.status_code}\nText: {response.text}\nContent: {response.content}")


# check if model is already in /persistent-storage
model_dir = f"/persistent-storage/.resources/{SOURCE}/{APP_NAME}/"
model_path_realESRGAN = os.path.join(model_dir, "RealESRGAN_x4plus.pth")
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)
if not os.path.exists(model_path_realESRGAN):
    response = requests.get(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    )
    with open(model_path_realESRGAN, "wb") as f:
        f.write(response.content)
    print("Download complete")

model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
)
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path_realESRGAN,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half="fp16",
    gpu_id=None,
)
model_path_GFPGANv1 = os.path.join(model_dir, "GFPGANv1.3.pth")
if not os.path.exists(model_path_GFPGANv1):
    response = requests.get(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    )
    with open(model_path_GFPGANv1, "wb") as f:
        f.write(response.content)
    print("Download complete")


#######################################
# Prediction
#######################################
def predict(item, run_id, logger):
    params = Item(**item)
    if params.file_url is not None:
        input_filename = "".join(x for x in params.file_url if x.isalnum())
        image = download_file_from_url(logger, params.file_url, input_filename)
        init_image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    elif params.image is not None:
        image_array = np.asarray(
            bytearray(BytesIO(base64.b64decode(params.image)).read()), dtype=np.uint8
        )
        init_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    else:
        raise Exception("No image or file_url provided")

    if params.face_enhance:
        face_enhancer = GFPGANer(
            model_path=model_path_GFPGANv1,
            upscale=params.upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )
        _, _, output = face_enhancer.enhance(
            init_image, has_aligned=False, only_center_face=False, paste_back=True
        )
    else:
        output, _ = upsampler.enhance(init_image, outscale=params.upscale)

    _, buffer = cv2.imencode(".png", output)
    # Convert the encoded image to a byte stream
    byte_stream = BytesIO(buffer)
    # Encode the byte stream to base64
    encoded_image = base64.b64encode(byte_stream.getvalue()).decode("utf-8")

    return {"result": encoded_image}
