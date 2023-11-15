# ControlNet

## Description

ControlNet provides a minimal interface allowing users to customize the generation process up to a great extent. With
ControlNet, users can easily condition the generation with different spatial contexts such as a depth map, a
segmentation map, a scribble, keypoints, and so on!

## Usage

Use `cerebrium deploy` command in this directory to deploy this model.

- Parameters

| Parameter             | Type              | Description                                             | Default  |
|-----------------------|-------------------|---------------------------------------------------------|----------|
| prompt                | str               | The input prompt                                        | None     |
| hf_token              | Optional[str]     | Hugging Face token                                      | None     |
| hf_model_path         | Optional[str]     | Path to the Hugging Face model                          | None     |
| num_inference_steps   | Optional[int]     | Number of inference steps                               | 20       |
| height                | Optional[int]     | The height to resize the image to                       | 512      |
| width                 | Optional[int]     | The width to resize the image to                        | 512      |
| guidance_scale        | Optional[float]   | Guidance scaling factor                                 | 7.5      |
| negative_prompt       | Optional[str]     | Negative prompt to direct the model's output            | None     |
| num_images_per_prompt | Optional[int]     | Number of images to generate per prompt                 | 1        |
| low_threshold         | Optional[int]     | Lower threshold for Canny edge detection                | 100      |
| high_threshold        | Optional[int]     | Higher threshold for Canny edge detection               | 200      |
| scale                 | Optional[float]   | Scale factor for the generated image                    | 9.0      |
| seed                  | Optional[int]     | Seed for random number generator                        | 1        |
| model                 | str               | The type of the model                                   | "normal" |
| image                 | Optional[str]     | Base64 encode string of the image to process            | None     |
| file_url              | Optional[str]     | A publicly accessible file URL of the image to process  | None     |
| webhook_endpoint      | Optional[HttpUrl] | Endpoint for sending the result when processing is done | None     |
