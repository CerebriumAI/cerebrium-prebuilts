# Pygmalion 2.7B

## Description
Pygmalion 2.7B

## Usage
Deploy this model using the command: cerebrium deploy <NAME>

- Parameters

| Parameter | Type | Description | Default |
| prompt | str | The input text prompt to guide the generation | *None specified—it's a required field* |
| max_new_tokens | Optional[int] | The maximum number of new tokens to generate | 50 |
| temperature | Optional[float] | Controls randomness in generation; lower values lead to more deterministic outputs | 0.9 |
| top_p | Optional[float] | Probability threshold for nucleus sampling; tokens with cumulative probability above top_p are considered | 0.9 |
| typical_p | Optional[float] | Typical value for nucleus sampling | 0.9 |
| repetition_penalty | Optional[float]  | Penalty for repeating the same words or phrases | 0.5 |
| top_k | Optional[int] | Number of top tokens considered for generation at each step | None |
| stopping_criteria | Optional[list] | Criteria to stop the generation process | [] |
| pad_token_id | Optional[str] | ID of the padding token | None |
| webhook_endpoint | Optional[HttpUrl] | Endpoint for sending the result once generation is complete | None |