
from typing import Optional
from pydantic import BaseModel
import csv
import sys
import time

import numpy as np
import torch

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
from utils import load_tokenizer, read_model_name

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp
# logger.set_level('error')

# TensorRT params
engine_dir = "/app/trt-engine"
tokenizer_dir = "/app/gemma_pytorch/tokenizer/tokenizer.model"

STOP_WORDS_LIST = None
BAD_WORDS_LIST = None
PROMPT_TEMPLATE = None


class Item(BaseModel):
    prompt: str
    num_tokens = Optional[int] = 100
    temperature:  Optional[float] = 0.95
    top_k: Optional[int] = 100
    top_p: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.05, 
    prompt_template: Optional[str] = "<start_of_turn>user\n{input_text}<end_of_turn>\n<start_of_turn>model\n"

def load_model():
    runtime_rank = tensorrt_llm.mpi_rank()

    use_py_session = False

    self.model_name, self.model_version = read_model_name(engine_dir)

    self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
        vocab_file=tokenizer_dir,
        model_name=self.model_name,
        model_version=self.model_version,
    )

    if not PYTHON_BINDINGS:
        logger.warning(
            "Python bindings of C++ session is unavailable, fallback to Python session."
        )
        use_py_session = True

    runner_cls = ModelRunner if use_py_session else ModelRunnerCpp
    self.runner = runner_cls.from_dir(engine_dir=engine_dir, rank=runtime_rank)

    print("Model loaded!")

def parse_input(
        tokenizer,
        input_text=None,
        prompt_template=None,
        input_file=None,
        add_special_tokens=True,
        max_input_length=923,
        pad_id=None,
        num_prepend_vtokens=[],
        model_name=None,
        model_version=None,
    ):
        if pad_id is None:
            pad_id = tokenizer.pad_token_id

        batch_input_ids = []
        if input_file is None:
            for curr_text in input_text:
                if prompt_template is not None:
                    curr_text = prompt_template.format(input_text=curr_text)
                input_ids = tokenizer.encode(
                    curr_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length,
                )
                batch_input_ids.append(input_ids)
        else:
            if input_file.endswith(".csv"):
                with open(input_file, "r") as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    for line in csv_reader:
                        input_ids = np.array(line, dtype="int32")
                        batch_input_ids.append(input_ids[-max_input_length:])
            elif input_file.endswith(".npy"):
                inputs = np.load(input_file)
                for row in inputs:
                    input_ids = row[row != pad_id]
                    batch_input_ids.append(input_ids[-max_input_length:])
            elif input_file.endswith(".txt"):
                with open(
                    input_file, "r", encoding="utf-8", errors="replace"
                ) as txt_file:
                    input_text = txt_file.read()
                    input_ids = tokenizer.encode(
                        input_text,
                        add_special_tokens=add_special_tokens,
                        truncation=True,
                        max_length=max_input_length,
                    )
                    batch_input_ids.append(input_ids)
            else:
                print("Input file format not supported.")
                raise SystemExit
        if model_name == "GemmaForCausalLM":
            batch_input_ids[0] = [tokenizer.bos_token_id] + batch_input_ids[0]

        if num_prepend_vtokens:
            assert len(num_prepend_vtokens) == len(batch_input_ids)
            base_vocab_size = tokenizer.vocab_size - len(
                tokenizer.special_tokens_map.get("additional_special_tokens", [])
            )
            for i, length in enumerate(num_prepend_vtokens):
                batch_input_ids[i] = (
                    list(range(base_vocab_size, base_vocab_size + length))
                    + batch_input_ids[i]
                )

        if model_name == "ChatGLMForCausalLM" and model_version == "glm":
            for ids in batch_input_ids:
                ids.append(tokenizer.sop_token_id)

        batch_input_ids = [torch.tensor(x, dtype=torch.int32) for x in batch_input_ids]
        return batch_input_ids


def predict(item, run_id, logger):
    item = Item(**item)

    print("Predicting...")

    if len(prompt) > 8:
        raise ValueError(
            "The number of batched prompts must be less than or equal to 8."
        )

    # # An example to stop generation when the model generate " London" on first sentence, " eventually became" on second sentence
    # stop_words_list = [[" London"], ["eventually became"]]
    # stop_words_list = tensorrt_llm.runtime.to_word_list_format(stop_words_list, tokenizer)
    # stop_words_list = torch.Tensor(stop_words_list).to(torch.int32).to("cuda").contiguous()
    stop_words_list = None

    # # An example to prevent generating " chef" on first sentence, " eventually" and " chef before" on second sentence
    # bad_words_list = [[" chef"], [" eventually, chef before"]]
    # bad_words_list = tensorrt_llm.runtime.to_word_list_format(bad_words_list, tokenizer)
    # bad_words_list = torch.Tensor(bad_words_list).to(torch.int32).to("cuda").contiguous()
    bad_words_list = None

    batch_input_ids = self.parse_input(
        tokenizer=self.tokenizer,
        input_text=prompt,
        prompt_template=kwargs.prompt_template,
        pad_id=self.pad_id,
        model_name=self.model_name,
        model_version=self.model_version,
    )
    input_lengths = [x.size(0) for x in batch_input_ids]

    time_begin = time.time()
    with torch.no_grad():
        outputs = self.runner.generate(
            batch_input_ids,
            max_new_tokens=num_tokens,
            max_attention_window_size=None,
            sink_token_length=None,
            end_id=self.end_id,
            pad_id=self.pad_id,
            temperature=kwargs.temperature,
            top_k=kwargs.top_k,
            top_p=kwargs.top_p,
            num_beams=1,
            repetition_penalty=kwargs.repetition_penalty,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            output_sequence_lengths=True,
            return_dict=True,
        )
        torch.cuda.synchronize()

    time_total = time.time() - time_begin

    output_ids = outputs["output_ids"]
    sequence_lengths = outputs["sequence_lengths"]

    batch_size, num_beams, _ = output_ids.size()
    generated_texts = []
    generated_ids = []
    for batch_idx in range(batch_size):
        inputs = output_ids[batch_idx][0][: input_lengths[batch_idx]].tolist()
        input_text = self.tokenizer.decode(inputs)
        print(f'Input [Text {batch_idx}]: "{input_text}"')
        for beam in range(num_beams):
            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[batch_idx][beam]
            outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()
            output_text = self.tokenizer.decode(outputs)
            generated_ids.append(outputs)
            generated_texts.append(output_text)

    generated_tokens = sum([len(x) for x in generated_ids])
    print(
        f"Response generated in {time_total:.2f} seconds, {generated_tokens} tokens, {generated_tokens / time_total:.2f} tokens/second"
    )
    print(f"Batch size: {batch_size}")

    print("Prediction complete!")

    return {
        "response_txt": generated_texts,
        "throughput_tok_s": generated_tokens / time_total,
        "latency_s": time_total,
    }
