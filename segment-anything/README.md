# Segment Anything

## Description

Segment Anything

## Usage

Deploy this model using the command: cerebrium deploy <NAME>

- Parameters

| cursor | list | List of cursor data | *None specified—it's a required field* |
| image | Optional[str] | Base64 encoded string of the image to process | None |
| points_per_side | Optional[int] | Number of points per side | 32 |
| pred_iou_thresh | Optional[float] | Threshold for predicted intersection over union | 0.96 |
| stability_score_thresh | Optional[float] | Threshold for stability score | 0.92 |
| crop_n_layers | Optional[int] | Number of layers to crop | 1 |
| crop_n_points_downscale_factor | Optional[int] | Downscale factor for number of points to crop | 2 |
| min_mask_region_area | Optional[int] | Minimum mask region area | 100 |
| file_url | Optional[HttpUrl] | A publicly accessible file URL of the image to process | None |
| webhook_endpoint | Optional[HttpUrl] | Endpoint for sending the result when processing is done | None |