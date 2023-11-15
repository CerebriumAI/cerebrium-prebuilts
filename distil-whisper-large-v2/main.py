import base64
import uuid
from typing import Optional

from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from pydantic import HttpUrl
from whisper import load_model, transcribe


class Item(BaseModel):
    audio: Optional[str] = None
    file_url: Optional[HttpUrl] = None
    webhook_endpoint: Optional[HttpUrl] = None


distil_large_v2 = hf_hub_download(
    repo_id="distil-whisper/distil-large-v2", filename="original-model.bin"
)
model = load_model(distil_large_v2)

DOWNLOAD_ROOT = "/tmp/"  # Change this to /persistent-storage/ if you want to save files to the persistent storage


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


# Saves a base64 encoded file string to a local file
def save_base64_string_to_file(logger, audio: str):
    logger.info("Converting file...")

    decoded_data = base64.b64decode(audio)

    filename = f"{DOWNLOAD_ROOT}/{uuid.uuid4()}"

    with open(filename, "wb") as file:
        file.write(decoded_data)

    logger.info("Decoding base64 to file was successful")
    return filename


def predict(item, run_id, logger):
    item = Item(**item)
    input_filename = f"{run_id}.mp3"

    if item.audio is not None:
        file = save_base64_string_to_file(logger, item.audio)
    elif item.file_url is not None:
        file = download_file_from_url(logger, item.file_url, input_filename)
    logger.info("Transcribing file...")

    result = transcribe(model, audio=file)

    return result
