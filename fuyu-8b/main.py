from typing import Optional
from pydantic import BaseModel
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests
import base64
from io import BytesIO


class Item(BaseModel):
    # Add your input parameters here
    prompt: str
    file_url: Optional[str] = None
    image: Optional[str] = None


model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0")


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
        raise Exception(
            f"Download failed. Response from URL was: \nStatus Code:{response.status_code}\nText: {response.text}\nContent: {response.content}"
        )


def predict(item, run_id, logger):
    item = Item(**item)

    if item.file_url:
        init_image = Image.open(download_file_from_url(logger, item.file_url, run_id))
    elif item.image:
        init_image = Image.open(BytesIO(base64.b64decode(item.image)))

    inputs = processor(text=item.prompt, images=init_image, return_tensors="pt").to(
        "cuda:0"
    )
    generation_output = model.generate(**inputs, max_new_tokens=7)
    generation_text = processor.batch_decode(
        generation_output[:, -7:], skip_special_tokens=True
    )

    return {"result": generation_text}
