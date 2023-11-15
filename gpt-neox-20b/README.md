# GPT NEOX 20B

## Description

GPT NEOX 20B

## Usage

Use `cerebrium deploy` command in this directory to deploy this model.

- Parameters

| Parameter        | Type              | Description                                                                        | Default                 |
|------------------|-------------------|------------------------------------------------------------------------------------|-------------------------|
| prompt           | str               | The input text prompt to guide the generation                                      | *it's a required field* |
| max_length       | Optional[int]     | The maximum length of the generated text output                                    | 100                     |
| temperature      | Optional[float]   | Controls randomness in generation; lower values lead to more deterministic outputs | 0.9                     |
| webhook_endpoint | Optional[HttpUrl] | The HTTP URL endpoint to send the result to once generation is complete            | None                    |