# Stable Diffusion Img2Img

## Description

Stable Diffusion Img2Img

## Usage

Deploy this model using the command: cerebrium deploy <NAME>

- Parameters

| Parameter | Type | Description | Default |
| prompt | str | The input text prompt for the generation | *None specified—it's a required field* |
| image | Optional[str] | Base64 encoded string of the image to process | None |
| hf_token | Optional[str] | Hugging Face API token | None |
| hf_model_path | Optional[str] | Hugging Face model path | "stabilityai/stable-diffusion-2-1" |
| guidance_scale | float | Scale value for guidance | 7.5 |
| height | int | Height of the image | 512 |
| negative_prompt | str | Negative prompt for guidance | "" |
| num_images_per_prompt | int | Number of images per prompt | 1 |
| num_inference_steps | int | Number of inference steps | 50 |
| seed | int | Seed value | 0 |
| width | int | Width of the image | 512 |
| file_url | Optional[str] | A publicly accessible file URL of the image to process | None |
| webhook_endpoint | Optional[HttpUrl] | Endpoint for sending the result when processing is done | None |