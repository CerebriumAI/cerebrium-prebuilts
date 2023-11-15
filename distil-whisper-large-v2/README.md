# Distill Whisper

## Description

It is a distilled version of the Whisper model that is 6 times faster, 49% smaller, and performs within 1% WER on
out-of-distribution evaluation sets. This is the repository for distil-large-v2, a distilled variant of Whisper
large-v2.

## Usage

Use `cerebrium deploy` command in this directory to deploy this model.

- Parameters

| Parameter        | Type              | Description                                             | Default |
|------------------|-------------------|---------------------------------------------------------|---------|
| audio            | Optional[str]     | Base64 encoded string of the audio to process           | None    |
| file_url         | Optional[HttpUrl] | A publicly accessible file URL of the audio to process  | None    |
| webhook_endpoint | Optional[HttpUrl] | Endpoint for sending the result when processing is done | None    |
