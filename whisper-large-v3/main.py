import base64
import uuid
from typing import Literal, Optional
import torch
from pydantic import BaseModel, HttpUrl
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class Item(BaseModel):
    mode: Optional[Literal["transcribe", "translate"]] = "transcribe"
    language: Optional[str] = None
    audio: Optional[str] = None
    file_url: Optional[HttpUrl] = None
    max_new_tokens: Optional[int] = 128
    chunk_length_s: Optional[int] = 30
    batch_size: Optional[int] = 16
    webhook_endpoint: Optional[HttpUrl] = None


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=True)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
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
    params = Item(**item)
    input_filename = f"{run_id}.mp3"

    if params.audio is not None:
        file = save_base64_string_to_file(logger, params.audio)
    elif params.file_url is not None:
        file = download_file_from_url(logger, params.file_url, input_filename)
    logger.info("Transcribing file...")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=item.max_new_tokens,
        chunk_length_s=item.chunk_length_s,
        batch_size=item.batch_size,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    if params.mode == "translate":
        result = pipe(file, generate_kwargs={"task": "translate"})
    else:
        result = pipe(file, generate_kwargs={"language": params.language})

    return result
