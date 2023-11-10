# Mistral 7B Instruct

## Description
Mistral 7B Instruct

## Usage
Deploy this model using the command: cerebrium deploy <NAME>

- Parameters

| Parameter | Type | Description | Default |
| prompt | str | The input text prompt to guide the generation | *None specified—it's a required field* |
| temperature | Optional[float] | Controls randomness in generation; lower values lead to more deterministic outputs | 0.8 |
| top_p | Optional[float] | Probability threshold for nucleus sampling; tokens with cumulative probability above top_p are considered | 0.75 |
| top_k | Optional[float] | Number of top tokens considered for generation at each step | 40 |
| max_tokens | Optional[int] | The maximum number of tokens in the generated text output | 256 |
| frequency_penalty | Optional[int] | Determines how much to avoid repeating the same words | 1 |