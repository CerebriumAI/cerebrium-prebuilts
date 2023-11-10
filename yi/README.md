# YI 6B 200k

## Description
The Yi series models are large language models trained from scratch by developers at 01.AI. The first public release contains two bilingual(English/Chinese) base models with the parameter sizes of 6B(Yi-6B) and 34B(Yi-34B). Both of them are trained with 4K sequence length and can be extended to 32K during inference time. The Yi-6B-200K and Yi-34B-200K are base model with 200K context length.

## Usage
Deploy this model using the command: cerebrium deploy <NAME>

- Parameters

| prompt | str | The input text prompt for the generation | *None specified—it's a required field* |
| max_length | Optional[int] | The maximum length of the generated text output | 256 |
| repetition_penalty | Optional[float] | Controls the degree of repetition in the generated text | 1.3 |
| no_repeat_ngram_size | Optional[int] | The size of n-grams to avoid repeating in the output | 5 |
| temperature | Optional[float] | Controls the degree of randomness in the generation process | 5 |
| top_k | Optional[int] | The number of highest probability vocabulary tokens to keep | 40 |
| top_p | Optional[float] | The cumulative probability cutoff for nucleus sampling | 0.8 |