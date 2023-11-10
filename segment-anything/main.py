import base64
import heapq
import os
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel, HttpUrl
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class Item(BaseModel):
    cursor: list
    image: Optional[str] = None
    points_per_side: Optional[int] = 32
    pred_iou_thresh: Optional[float] = 0.96
    stability_score_thresh: Optional[float] = 0.92
    crop_n_layers: Optional[int] = 1
    crop_n_points_downscale_factor: Optional[int] = 2
    min_mask_region_area: Optional[int] = 100
    webhook_endpoint: Optional[HttpUrl]
    file_url: Optional[HttpUrl] = None
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


def distance(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def find_top_4_annotations_by_coordinates(annotations, x, y):
    distances = []
    for ann in annotations:
        bbox_x, bbox_y, bbox_w, bbox_h = ann["bbox"]
        center_x, center_y = bbox_x + bbox_w / 2, bbox_y + bbox_h / 2
        dist = distance(x, y, center_x, center_y)
        heapq.heappush(distances, (dist, ann))

    smallest = heapq.nsmallest(4, distances)
    top_4_distances = [item[1] for item in smallest]

    top_4_annotations = {
        f"segmentation_{i}": d["segmentation"].tolist()
        for i, d in enumerate(top_4_distances)
    }
    return top_4_annotations


print("Downloading file...")
if not os.path.exists("/persistent-storage/segment-anything/sam_vit_h_4b8939.pth"):
    response = requests.get(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    )
    with open("sam_vit_h_4b8939.pth", "wb") as f:
        f.write(response.content)
    print("Download complete")
else:
    print("File already exists")

sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")
sam.to("cuda")


def find_annotation_by_coordinates(annotations, x, y):
    for ann in annotations:
        bbox_x, bbox_y, bbox_w, bbox_h = ann["bbox"]
        if bbox_x <= x <= bbox_x + bbox_w and bbox_y <= y <= bbox_y + bbox_h:
            return ann
    return None


def predict(item, run_id, logger):
    params = Item(**item)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=params.points_per_side,
        pred_iou_thresh=params.pred_iou_thresh,
        stability_score_thresh=params.stability_score_thresh,
        crop_n_layers=params.crop_n_layers,
        crop_n_points_downscale_factor=params.crop_n_points_downscale_factor,
        min_mask_region_area=params.min_mask_region_area,  # Requires open-cv to run post-processing
    )

    if params.image:
        image = cv2.cvtColor(
            np.array(Image.open(BytesIO(base64.b64decode(params.image)))),
            cv2.COLOR_BGR2RGB,
        )
    elif params.file_url:
        image = download_file_from_url(logger, params.file_url, run_id)
        image_bin = cv2.cvtColor(np.array(Image.open(image)), cv2.COLOR_BGR2RGB)
    else:
        raise Exception("No image or file_url provided")

    masks = mask_generator.generate(image_bin)
    selected_annotations = find_top_4_annotations_by_coordinates(
        masks, params.cursor[0], params.cursor[1]
    )

    return selected_annotations
