# Whisper Large 3

## Description
Whisper large 3

## Usage
Deploy this model using the command: cerebrium deploy <NAME>

- Parameters

| mode | Optional[Literal["transcribe", "translate"]] | The mode of operation, either "transcribe" or "translate" | "transcribe" |
| language | Optional[str] | The language for translation or transcription | None |
| audio | Optional[str] | Base64 encoded string of the audio file | None |
| file_url | Optional[HttpUrl] | A publicly accessible file URL of the audio file | None |
| max_new_tokens | Optional[int] | Maximum number of tokens to generate | 128 |
| chunk_length_s | Optional[int] | Length of audio chunks in seconds | 30 |
| batch_size | Optional[int] | Batch size for processing | 16 |
| webhook_endpoint | Optional[HttpUrl] | Endpoint for sending the result when processing is done | None |