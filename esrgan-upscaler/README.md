# ESRGAN Upscaler

## Description

Allows you to upscale images to a much better quality.

## Usage

Use `cerebrium deploy` command in this directory to deploy this model.

- Parameters

| Parameter        | Type              | Description                                             | Default |
|------------------|-------------------|---------------------------------------------------------|---------|
| image            | Optional[str]     | Base64 encoded string of the image to process           | None    |
| file_url         | Optional[str]     | A publicly accessible file URL of the image to process  | None    |
| upscale          | Optional[int]     | Factor by which to upscale the image                    | 4       |
| face_enhance     | Optional[bool]    | Whether to enhance faces in the image                   | False   |
| webhook_endpoint | Optional[HttpUrl] | Endpoint for sending the result when processing is done | None    |