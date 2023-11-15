# Stable Diffusion Upscaler

## Description

Stable Diffusion Upscaler

## Usage

Deploy this model using the command: cerebrium deploy <NAME>

- Parameters

| Parameter | Type | Description | Default |
| image | Optional[str] | Base64 encoded string of the image to process | None |
| file_url | Optional[str] | A publicly accessible file URL of the image to process | None |
| prompt | str | The input text prompt for the generation | "" |
| height | int | Height of the image | 250 |
| width | int | Width of the image | 250 |
| num_inference_steps | int | Number of inference steps | 50 |
| guidance_scale | float | Scale value for guidance | 7.5 |
| negative_prompt | str | Negative prompt for guidance | "" |
| num_images_per_prompt | int | Number of images per prompt | 1 |
| webhook_endpoint | Optional[HttpUrl] | Endpoint for sending the result when processing is done | None |