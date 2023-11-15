# Whisper Large 2

## Description

Whisper large 2

## Usage

Deploy this model using the command: cerebrium deploy <NAME>

- Parameters

| mode | Optional[Literal["transcribe", "translate"]] | The mode of operation, either "transcribe" or "translate" | "
transcribe" |
| language | Optional[str] | The language for translation or transcription | None |
| audio | Optional[str] | Base64 encoded string of the audio file | None |
| file_url | Optional[HttpUrl] | A publicly accessible file URL of the audio file | None |
| webhook_endpoint | Optional[HttpUrl] | Endpoint for sending the result when processing is done | None |