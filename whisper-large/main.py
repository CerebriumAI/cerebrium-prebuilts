import base64
import uuid
from typing import Literal, Optional

import whisper
from pydantic import BaseModel, HttpUrl


########################################
# User-facing API Parameters
########################################
class Item(BaseModel):
    mode: Optional[Literal["transcribe", "translate"]] = "transcribe"
    language: Optional[str] = None
    audio: Optional[str] = None
    file_url: Optional[HttpUrl] = None
    webhook_endpoint: Optional[HttpUrl] = None


########################################
# Initialize the model
########################################
model = whisper.load_model("large-v2")
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


#######################################
# Prediction
#######################################
def predict(item, run_id, logger):
    params = Item(**item)
    input_filename = f"{run_id}.mp3"

    if params.audio is not None:
        file = save_base64_string_to_file(logger, params.audio)
    elif params.file_url is not None:
        file = download_file_from_url(logger, params.file_url, input_filename)
    logger.info("Transcribing file...")

    if params.mode == "translate":
        result = model.transcribe(file, task="translate")
    else:
        result = model.transcribe(file, language=params.language)

    return result
