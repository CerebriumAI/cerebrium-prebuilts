
from typing import Optional
from pydantic import BaseModel
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoConfig, AutoTokenizer
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
import base64

MODEL_NAME = "SkunkworksAI/BakLLaVA-1"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

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


class Item(BaseModel):
    image:  Optional[str] = None
    file_url: Optional[str] = None
    prompt: str
    max_sequence: Optional[int] = 512

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
    cache_dir=TOKEN_CACHE,
)
model = LlavaMistralForCausalLM.from_pretrained(
    MODEL_CACHE
).to("cuda")

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    print("Loading vision tower")
    vision_tower.load_model()
vision_tower.to(device='cuda', dtype=torch.float16)

def predict(item, run_id, logger):
    item = Item(**item)

    if item.file_url:
        init_image = Image.open(download_file_from_url(logger, item.file_url, run_id))
    elif item.image:
        init_image = Image.open(BytesIO(base64.b64decode(item.image)))

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    image_processor = vision_tower.image_processor

    if model.config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + item.prompt
    else:
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + item.prompt

    conv = conv_templates["llava_v1"].copy()

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    image_tensor = image_processor.preprocess(init_image, return_tensors='pt')['pixel_values'].cuda()

    output_ids = model.generate(
        input_ids=input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=item.max_sequence,
        use_cache=True,
        stopping_criteria=[stopping_criteria]
    )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs