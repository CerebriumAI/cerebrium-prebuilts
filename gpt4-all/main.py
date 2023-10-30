import os
import sys
from pathlib import Path
from typing import Optional

from nomic.gpt4all.gpt4all import GPT4AllGPU
from pydantic import BaseModel, HttpUrl
import huggingface_hub
from cerebrium import get_secret


#######################################
# User-facing API Parameters
#######################################
class Item(BaseModel):
    prompt: str
    num_beams: Optional[int] = 2
    min_new_tokens: Optional[int] = 10
    max_length: Optional[int] = 100
    repetition_penalty: Optional[float] = 2.0
    webhook_endpoint: Optional[HttpUrl] = None


#######################################
# Load the model
#######################################

ENV = os.getenv("ENV", "prod")
MODEL_NAME = os.getenv("MODEL_NAME")

models_path = Path("..") / "models"
sys.path.append(str(models_path.resolve()))
sys.path.remove(str(models_path.resolve()))

try:
    hf_auth_token = get_secret("hf_auth_token")
    if hf_auth_token == "":
        raise Exception(
            "hf_auth_token is empty. You need a hf_auth_token secret added to your account to access this model."
        )
except Exception as e:
    print("\n\n")
    print("=" * 60)
    print("Error: ", e)
    print("=" * 60)
    raise e

huggingface_hub.login(token=hf_auth_token)
print("Loading hf model... this could take a while. Sit tight!")

model = GPT4AllGPU("decapoda-research/llama-7b-hf")


#######################################
# Prediction
#######################################
def predict(item, run_id, logger):
    params = Item(**item)
    config = {
        "num_beams": params.num_beams,
        "min_new_tokens": params.min_new_tokens,
        "max_length": params.max_length,
        "repetition_penalty": params.repetition_penalty,
    }
    out = model.generate(params.prompt, config)

    return out
