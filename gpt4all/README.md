# GPT4ALL

## Description

GPT4ALL

## Usage

Use `cerebrium deploy` command in this directory to deploy this model.

- Parameters

| Parameter      | Type            | Description                                                                                               | Default                                |
|----------------|-----------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------|
| prompt         | str             | The input text prompt to guide the generation                                                             | *None specified—it's a required field* |
| max_tokens     | Optional[int]   | The maximum number of tokens in the generated text output                                                 | 250                                    |
| temperature    | Optional[float] | Controls randomness in generation; lower values lead to more deterministic outputs                        | 0.7                                    |
| top_k          | Optional[int]   | Number of top tokens considered for generation at each step                                               | 40                                     |
| top_p          | Optional[float] | Probability threshold for nucleus sampling; tokens with cumulative probability above top_p are considered | 0.4                                    |
| repeat_penalty | Optional[float] | Multiplier for penalizing repetition of words or phrases in generation                                    | 1.18                                   |