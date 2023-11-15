# Falcon 7B

## Description

The Falcon 7B LLM

## Usage

Use `cerebrium deploy` command in this directory to deploy this model.

- Parameters

| Parameter            | Type              | Description                                                                                                                           | Default |
|----------------------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------|---------|
| prompt               | str               | The text prompt to guide the image processing                                                                                         | None    |
| max_length           | Optional[int]     | The maximum length of the returned text                                                                                               | 200     |
| temperature          | Optional[float]   | Controls the randomness of the output (lower is more deterministic)                                                                   | 0.5     |
| top_p                | Optional[float]   | Nucleus sampling: picks the next word from the smallest possible set of words whose cumulative probability exceeds the value of top_p | 1.0     |
| top_k                | Optional[int]     | The number of highest probability vocabulary tokens to keep for top-k-filtering                                                       | 10      |
| repetition_penalty   | Optional[float]   | Penalty applied to repeated words to discourage repetition                                                                            | 1.0     |
| num_return_sequences | Optional[int]     | Number of different sequences to generate                                                                                             | 1       |
| webhook_endpoint     | Optional[HttpUrl] | Endpoint for sending the result when processing is completed                                                                          | None    |