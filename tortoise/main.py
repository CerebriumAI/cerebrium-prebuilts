import base64
import io
import os
import site
from typing import Optional

import pydub
import torch
from pydantic import BaseModel, HttpUrl
from scipy.io import wavfile
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice


#######################################
# User-facing API Parameters
#######################################
class Item(BaseModel):
    prompt: str
    voice: Optional[str] = "random"
    preset: Optional[str] = "fast"
    file_url: Optional[HttpUrl] = None
    webhook_endpoint: Optional[HttpUrl] = None


#######################################
# Model Setup
#######################################

device = 0 if torch.cuda.is_available() else -1
model = TextToSpeech()


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


def mp3_bytes_from_wav_bytes(wav_bytes: io.BytesIO) -> io.BytesIO:
    mp3_bytes = io.BytesIO()
    sound = pydub.AudioSegment.from_wav(wav_bytes)
    sound.export(mp3_bytes, format="mp3")
    mp3_bytes.seek(0)
    return mp3_bytes


def base64_encode(buffer: io.BytesIO) -> str:
    """
    Encode the given buffer as base64.
    """
    return base64.encodebytes(buffer.getvalue()).decode("ascii")


def copy_package_tortoise(filename):
    for directory in site.getsitepackages():
        package_path = os.path.join(directory, "tortoise")
        if os.path.exists(package_path):
            custom_voices_path = os.path.join(package_path, "voices", "custom")
            os.makedirs(custom_voices_path, exist_ok=True)
            destination_wav_path = os.path.join(custom_voices_path, filename)
            with open(f"models/{filename}", "rb") as source_file, open(
                destination_wav_path, "wb"
            ) as destination_file:
                content = source_file.read()
                destination_file.write(content)
            print(f"File 'recording.wav' copied to {custom_voices_path}")
            return custom_voices_path
    else:
        print("Package 'tortoise' not found.")


def inference(params, input_filename: str):
    text = params.get(
        "prompt",
        "You need to send in text with the model in order to get a response back",
    )
    voice = params.get("voice", "random")
    preset = params.get("preset", "fast")

    if voice == "custom":
        custom_path = copy_package_tortoise(input_filename)

    voice_samples, conditioning_latents = load_voice(voice)

    # Run the model
    gen = model.tts_with_preset(
        text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset=preset,
    )

    if voice == "custom":
        os.remove(os.path.join(custom_path, input_filename))

    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, 24000, gen.squeeze().cpu().numpy())
    wav_bytes.seek(0)
    mp3_bytes = mp3_bytes_from_wav_bytes(wav_bytes)

    return {"audio": "data:audio/mpeg;base64," + base64_encode(mp3_bytes)}


#######################################
# Prediction
#######################################
def predict(item, run_id, logger):
    params = Item(**item)
    input_filename = f"{run_id}.wav"

    if params.file_url is not None:
        download_file_from_url(logger, params.file_url, input_filename)
    else:
        raise Exception("No file URL provided")

    logger.info("Converting file...")
    result = inference(params, input_filename)

    return result
