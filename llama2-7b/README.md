# Llama 2 7B

## Description
Llama 2 7B

## Usage
Use `cerebrium deploy` command in this directory to deploy this model.

- Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| prompt | str | The input text prompt to guide the generation | *None specified—it's a required field* |
| cutoff_len | Optional[int] | The maximum length of the generated text output | 256 |
| temperature | Optional[float] | Controls randomness in generation; lower values lead to more deterministic outputs | 0.1 |
| top_p | Optional[float] | Probability threshold for nucleus sampling; tokens with cumulative probability above top_p are considered | 0.75 |
| top_k | Optional[float] | Number of top tokens considered for generation at each step | 40 |
| num_beams | Optional[int] | Number of beams for beam search, used to control the diversity of the generated text | 4 |
| max_new_tokens | Optional[int] | The maximum number of new tokens to generate | 256 |