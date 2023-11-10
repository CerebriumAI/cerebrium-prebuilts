# Zephr 7B

## Description
Zephyr is a series of language models that are trained to act as helpful assistants. Zephyr-7B-β is the second model in the series, and is a fine-tuned version of mistralai/Mistral-7B-v0.1 that was trained on on a mix of publicly available, synthetic datasets using Direct Preference Optimization (DPO). We found that removing the in-built alignment of these datasets boosted performance on MT Bench and made the model more helpful. However, this means that model is likely to generate problematic text when prompted to do so and should only be used for educational and research purposes. You can find more details in the technical report.

## Usage
Deploy this model using the command: cerebrium deploy <NAME>

- Parameters

| prompt | str | The input text prompt for the generation | *None specified—it's a required field* |
| messages | list | List of messages | *None specified—it's a required field* |
| max_length | Optional[int] | The maximum length of the generated text output | 256 |
| temperature | Optional[float] | Controls the randomness of the output | 0.7 |
| top_k | Optional[int] | Number of highest probability vocabulary tokens to keep for top-k-filtering | 50 |
| top_p | Optional[float] | Nucleus sampling: picks the next word from the smallest possible set of words whose cumulative probability exceeds the value of top_p | 0.95 |