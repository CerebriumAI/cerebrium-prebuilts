# Dreambooth

## Description
Allows you to run the default Stable Diffusion unless you provide a Dreambooth
model on Huggingface - it will run that

## Usage
Use `cerebrium deploy` command in this directory to deploy this model.

- Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| prompt | str | The input prompt | None |
| hf_token | Optional[str] | Hugging Face token | None |
| hf_model_path | Optional[str] | Path to the Hugging Face model | None |
| guidance_scale | float | Guidance scaling factor | 7.5 |
| height | int | The height to resize the image to | 512 |
| negative_prompt | str | Negative prompt to direct the model's output | "" |
| num_images_per_prompt | int | Number of images to generate per prompt | 1 |
| num_inference_steps | int | Number of inference steps | 50 |
| seed | Optional[int] | Seed for random number generator | 1 |
| width | int | The width to resize the image to | 512 |
| webhook_endpoint | Optional[HttpUrl] | Endpoint for sending the result when processing is done | None |
