# Falcon 40B Instruct Quantized

## Description
The Falcon 7B LLM

## Usage
Use `cerebrium deploy` command in this directory to deploy this model.

- Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| prompt | str | The input text prompt to guide the generation | *None specified—it's a required field* |
| max_length | Optional[int] | The maximum length of the generated text output | 200 |
| temperature | Optional[float] | Controls randomness in generation; lower values lead to more deterministic outputs | 0.5 |
| top_p | Optional[float] | Probability threshold for nucleus sampling; tokens with cumulative probability above top_p are considered | 1.0 |
| top_k | Optional[int] | Number of top tokens considered for generation at each step | 10 |
| repetition_penalty | Optional[float] | Multiplier for penalizing repetition of words or phrases in generation | 1.0 |
| num_return_sequences | Optional[int] | The number of generation sequences to return | 1 |
| webhook_endpoint | Optional[HttpUrl] | The HTTP URL endpoint to send the result to once generation is complete | None |