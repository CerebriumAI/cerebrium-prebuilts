# Meta Seamless

## Description

Meta Seamless

## Usage

1. Git clone the seamless-communication repo into this folder using the command: git
   clone https://github.com/facebookresearch/seamless_communication.git
2. Make sure you have added your AWS access key and secret to your secrets page on Cerebrium under the names:
   aws-access-key, aws-secret-key and meta-seamless-bucket. This
   is to upload the audio files to a s3 bucket
3. Deploy this model using the command: cerebrium deploy <NAME>

- Parameters

| Parameter   | Type          | Description                                       | Default                                |
|-------------|---------------|---------------------------------------------------|----------------------------------------|
| task        | str           | The task to perform, e.g., "s2st", "s2st"         | *None specified—it's a required field* |
| target_lang | str           | The target language for the translation           | *None specified—it's a required field* |
| src_lang    | str           | The source language of the input text             | *None specified—it's a required field* |
| url         | Optional[str] | A publicly accessible URL of the audio to process | None                                   |
| text        | Optional[str] | The input text to be translated                   | None                                   |