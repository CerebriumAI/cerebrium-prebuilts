# SDXL Turbo

## Description

SDXL-Turbo is a fast generative text-to-image model that can synthesize photorealistic images from a text prompt in a single network evaluation.

## Usage

Deploy this model using the command: cerebrium deploy

- Parameters

| prompt | str | The input text prompt for the SDXL model | *None specified—it's a required field* |
| num_inference_steps | Optional[int] | The number of steps to take to generate the image | 25 |
| guidance_scale | Optional[float] | Controls how much the image generation followers the prompt. The higher the better.| 3.0 |
