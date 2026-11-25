# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset, load_dataset
from datasets import Dataset as HFDataset
from transformers import ImageProcessingMixin, PreTrainedTokenizer

from legospec.data.parse import GeneralParser, HarmonyParser, ThinkingParser
from legospec.data.template import TEMPLATE_REGISTRY, ChatTemplate

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    process_vision_info = None


# define a type called conversation
Conversation = List[Dict[str, str]]


# ==============================
# This file is for preprocessing the data
# ==============================


def _apply_loss_mask_from_chat_template(
    text: str,
    offsets: torch.Tensor,
    chat_template: ChatTemplate,
) -> torch.Tensor:
    """
    Apply loss mask to identify assistant response spans using chat template.

    Args:
        text: The formatted conversation text.
        offsets: Token offset mapping from tokenizer.
        chat_template: The chat template to use for identifying assistant spans.

    Returns:
        A tensor indicating which tokens should contribute to the loss (1) or not (0).
    """
    loss_mask = torch.zeros(len(offsets), dtype=torch.long)

    user_message_separator = (
        f"{chat_template.end_of_turn_token}{chat_template.user_header}"
    )
    assistant_message_separator = (
        f"{chat_template.end_of_turn_token}{chat_template.assistant_header}"
    )

    # Find spans of assistant responses using regex
    assistant_pattern = (
        re.escape(assistant_message_separator)
        + r"(.*?)(?="
        + re.escape(user_message_separator)
        + "|$)"
    )

    matches_found = 0

    for match in re.finditer(assistant_pattern, text, re.DOTALL):
        matches_found += 1
        # Assistant response text span (excluding assistant_header itself)
        assistant_start_char = match.start(1)
        assistant_end_char = match.end(1)

        # Mark tokens overlapping with assistant response
        for idx, (token_start, token_end) in enumerate(offsets):
            # Token is part of the assistant response span
            if token_end <= assistant_start_char:
                continue  # token before assistant text
            if token_start > assistant_end_char:
                continue  # token after assistant text
            loss_mask[idx] = 1

    if matches_found == 0:
        print("WARNING: No assistant response spans found in the conversation text.")

    return loss_mask


# Copied from https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py
def preprocess_conversations(
    tokenizer: PreTrainedTokenizer,
    conversations: Union[List[Conversation], List[str]],
    chat_template: ChatTemplate,
    max_length: int = 2048,
    is_preformatted: bool = False,
    train_only_last_turn: bool = False,
    **kwargs,
) -> Dict[str, List[torch.Tensor]]:
    """
    Preprocess a batch of ShareGPT style conversations or pre-formatted text.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        conversations: A list of conversations (if is_preformatted=False) or
                      a list of pre-formatted text strings (if is_preformatted=True).
        chat_template: The chat template to use for formatting/identifying spans.
        max_length: The maximum length of the tokenized input.
        is_preformatted: Whether the input is already formatted text strings.
        train_only_last_turn: If True, only the last assistant turn contributes to the loss.

    Returns:
        A dictionary containing:
            - input_ids: List of tokenized input IDs.
            - loss_mask: List of loss masks indicating which tokens should contribute to the loss.
            - attention_mask: List of attention masks.
    """

    # prepare result
    results = {"input_ids": [], "loss_mask": [], "attention_mask": []}

    if chat_template.parser_type == "general":
        parser = GeneralParser(tokenizer, chat_template)
    elif chat_template.parser_type == "thinking":
        parser = ThinkingParser(tokenizer, chat_template)
    elif chat_template.parser_type == "openai-harmony":
        parser = HarmonyParser(tokenizer, chat_template)
    else:
        raise ValueError(f"Invalid parser type: {chat_template.parser_type}")

    kwargs_list = [{} for _ in range(len(conversations))]
    for key, value_list in kwargs.items():
        for i, value in enumerate(value_list):
            kwargs_list[i][key] = value
    for source, kwargs_item in zip(conversations, kwargs_list):
        if not source:
            # if the source is None, skip it
            continue
        input_ids, loss_mask = parser.parse(
            source,
            max_length,
            preformatted=is_preformatted,
            train_only_last_turn=train_only_last_turn,
            **kwargs_item,
        )
        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
    return results


def preprocess_vlm_conversations(
    processor: ImageProcessingMixin,
    examples: List[Conversation],
    chat_template: ChatTemplate,
    max_length: int = 2048,
) -> Dict[str, List[torch.Tensor]]:
    """
    Preprocess a batch of ShareGPT style conversations.

    Args:
        processor: The image processor to use for processing images.
        examples: A list of examples, where each example is a dictionary containing:
            - image: The image in the conversation.
            - conversations: A list of conversations, where each conversation is a list of messages.
        chat_template: The chat template to use for formatting the conversations.
        max_length: The maximum length of the tokenized input.

    Returns:
        A dictionary containing:
            - input_ids: List of tokenized input IDs.
            - loss_mask: List of loss masks indicating which tokens should contribute to the loss.
            - attention_mask: List of attention masks.
            - pixel_values: List of pixel values for images in the examples.
            - image_grid_thw: List of image grid tensors.
    """
    system_prompt = chat_template.system_prompt

    # prepare result
    results = {
        "input_ids": [],
        "loss_mask": [],
        "attention_mask": [],
        "pixel_values": [],
        "image_grid_thw": [],
    }

    # Note: currently, we assume that each example has only one image
    for i, image in enumerate(examples["image"]):
        source = examples["conversations"][i]
        messages = [{"role": "system", "content": system_prompt}]
        if not source:
            # if the source is None, skip it
            continue

        if source[0]["role"] != "user":
            # if the first message is not from user, skip it
            source = source[1:]

        convroles = ["user", "assistant"]
        for j, sentence in enumerate(source):
            role = sentence["role"]
            assert role == convroles[j % 2], f"unexpected role {role}"
            if role == "user":
                # if the message is from user and has image, process the image
                messages.append(
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": sentence["content"]},
                        ],
                    }
                )
            else:
                messages.append({"role": role, "content": sentence["content"]})

        conversation = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # get vision infor use qwen_vl_utils
        if not HAS_QWEN_VL_UTILS:
            raise ImportError(
                "qwen_vl_utils is required for VLM preprocessing but is not installed. "
                "Please install it to use VLM features."
            )
        image_inputs, video_inputs = process_vision_info(messages)
        assert image_inputs is not None, "image_inputs must not be None"

        encoding = processor(
            text=[conversation],
            images=image_inputs,
            videos=video_inputs,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        pixel_values = encoding.pixel_values
        image_grid_thw = encoding.image_grid_thw[0]

        # get conversation with image info for loss mask generation
        decoded_conversation = processor.tokenizer.decode(
            encoding.input_ids[0], skip_special_tokens=False
        )

        # Apply loss mask
        loss_mask = _apply_loss_mask_from_chat_template(
            decoded_conversation, offsets, chat_template
        )

        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
        results["pixel_values"].append(pixel_values)
        results["image_grid_thw"].append(image_grid_thw[None, :])
    return results


def build_training_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: Optional[str] = None,
    max_length: Optional[int] = 2048,
    shuffle_seed: Optional[int] = 42,
    num_proc: Optional[int] = 8,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
    is_vlm: Optional[bool] = False,
    processor: Optional[ImageProcessingMixin] = None,
    is_preformatted: Optional[bool] = False,
    train_only_last_turn: Optional[bool] = False,
    capture_errors: bool = False,
) -> HFDataset:
    """
    build training dataset

    Args:
        dataset: HF dataset to process.
        tokenizer: The tokenizer to use for tokenization.
        chat_template: The chat template to use for formatting conversations.
                        This includes the system prompt and user/assistant tokens
                        required to delineate different parts of the conversation
                        for loss mask generation.
        max_length: The maximum length of the tokenized input.
        shuffle_seed: The seed for shuffling the dataset.
        num_proc: The number of processes to use for multiprocessing.
        cache_dir: The directory to use for caching the processed dataset.
        cache_key: The key to use for caching the processed dataset.
        is_vlm: Whether the dataset is for VLM models.
        processor: The image processor to use for processing images.
        is_preformatted: Whether the dataset contains preformatted text of the conversation
                        (e.g. includes system prompt, user and assistant start and end tokens)
                        and doesn't need to have the chat template applied.
                        Note that the chat_template still needs to be specified to determine
                        the assistant spans for loss mask generation.
                        If True, expects "text" column with ready-to-train text.
                        If False, expects "conversations" column with ShareGPT format.
        train_only_last_turn: If True, only the last assistant turn contributes to the loss.
                             Useful for thinking models where history may not contain thoughts.
        capture_errors: If True, preserve one output row per input row, add
                        ``source_index`` and ``preprocessing_error``, and replace
                        failed tensor values with empty tensors. The default
                        remains strict for training callers.

    Returns:
        The processed HF dataset.
    """
    if is_vlm:
        assert processor is not None, "processor must be provided when is_vlm is True"

    # Validate chat_template requirement
    if chat_template is None:
        raise ValueError("chat_template must be provided for all dataset types")

    assert chat_template in TEMPLATE_REGISTRY.get_all_template_names(), (
        f"Chat template {chat_template} not found in TEMPLATE_REGISTRY, you may need to register it first"
    )

    template: ChatTemplate = TEMPLATE_REGISTRY.get(chat_template)

    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    original_cols = dataset.column_names

    def preprocess_function(examples):
        # Handle different dataset formats
        if is_vlm:
            processed = preprocess_vlm_conversations(
                processor,
                examples,
                template,
                max_length,
            )
        elif is_preformatted:
            # Handle pre-formatted text (should be in "text" column)
            if "text" not in examples:
                raise ValueError(
                    f"Expected 'text' column for is_preformatted=True, but found columns: {list(examples.keys())}"
                )
            processed = preprocess_conversations(
                tokenizer,
                examples["text"],
                template,
                max_length,
                is_preformatted=True,
                train_only_last_turn=train_only_last_turn,
            )
        else:
            # Handle ShareGPT conversations
            if "conversations" not in examples:
                raise ValueError(
                    f"Expected 'conversations' column for is_preformatted=False, but found columns: {list(examples.keys())}"
                )
            conversations = examples.pop("conversations")
            if "id" in examples:
                examples.pop("id")
            processed = preprocess_conversations(
                tokenizer,
                conversations,
                template,
                max_length,
                is_preformatted=False,
                train_only_last_turn=train_only_last_turn,
                **examples,
            )

        return processed

    # Process dataset only once
    if cache_dir and cache_key:
        load_from_cache_file = True
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = os.path.join(cache_dir, f"{cache_key}.pkl")
        print(f"dataset is cached at {cache_file_name}")
    elif cache_dir is None and cache_key is None:
        load_from_cache_file = False
        cache_file_name = None
        print("dataset is not cached")
    else:
        warnings.warn(
            "cache_dir and cache_key must be provided together to make caching work"
        )

    # Disable tokenizers internal parallelism when using multiprocessing to avoid
    # deadlocks caused by forked Rust threads (see huggingface/tokenizers#1391).
    if num_proc is not None and num_proc > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # adjust batch size based on dataset type
    if is_vlm:
        batch_size = (
            200  # reduce batch size for VLM datasets to avoid PyArrow offset overflow
        )
    else:
        batch_size = 1000  # default for conversations
    map_function = preprocess_function
    map_kwargs = {}
    if capture_errors:

        def preprocess_with_errors(examples, indices):
            results = {
                "input_ids": [],
                "loss_mask": [],
                "attention_mask": [],
                "source_index": [],
                "preprocessing_error": [],
            }
            for offset, source_index in enumerate(indices):
                single = {key: [values[offset]] for key, values in examples.items()}
                try:
                    processed = preprocess_function(single)
                    if len(processed["input_ids"]) != 1:
                        raise ValueError(
                            "preprocessing did not return exactly one sample"
                        )
                    for key in ("input_ids", "loss_mask", "attention_mask"):
                        results[key].append(processed[key][0])
                    results["preprocessing_error"].append("")
                except Exception as exc:
                    empty = torch.empty((1, 0), dtype=torch.long)
                    for key in ("input_ids", "loss_mask", "attention_mask"):
                        results[key].append(empty.clone())
                    results["preprocessing_error"].append(
                        f"{type(exc).__name__}: {exc}"
                    )
                results["source_index"].append(source_index)
            return results

        map_function = preprocess_with_errors
        map_kwargs["with_indices"] = True

    dataset = dataset.map(
        map_function,
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
        remove_columns=original_cols,
        # keep_in_memory=True,
        load_from_cache_file=load_from_cache_file,
        cache_file_name=cache_file_name,
        **map_kwargs,
    )

    dataset.set_format(type="torch")
    return dataset


def _project_jsonl(path, columns, source_identity):
    # source_identity participates in the generator cache fingerprint. A source
    # edited in place must not silently reuse an older Arrow dataset.
    import json

    try:
        import orjson

        decode = orjson.loads
    except ImportError:
        decode = json.loads
    with open(path, "rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = decode(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            missing = set(columns) - record.keys()
            if missing:
                raise ValueError(
                    f"Missing training columns {sorted(missing)} at {path}:{line_number}"
                )
            yield {column: record[column] for column in columns}


def load_training_json(data_files, *, columns=None, cache_dir=None):
    """Keep the standard loader unless explicit JSONL projection is requested.

    Projection preserves selected values and source order, including all turns.
    Auxiliary inputs such as tools can be selected alongside conversations.
    """
    if columns is None:
        if isinstance(data_files, Path):
            data_files = str(data_files)
        return load_dataset("json", data_files=data_files)["train"]
    # HF treats list-valued gen_kwargs as input shards. Column names are one
    # atomic projection, rather than independent shards to split or iterate.
    columns = tuple(dict.fromkeys(columns))
    if not columns:
        raise ValueError("At least one training data column must be selected")
    path = Path(data_files).expanduser().resolve(strict=True)
    stat = path.stat()
    return Dataset.from_generator(
        _project_jsonl,
        gen_kwargs={
            "path": str(path),
            "columns": columns,
            "source_identity": (stat.st_size, stat.st_mtime_ns),
        },
        cache_dir=cache_dir,
    )


class DataCollatorWithPadding:
    """
    Datacollator that will dynamically pad the inputs for batching.
    """

    def __init__(
        self,
        pad_to_length: Optional[int] = None,
        *,
        sp_degree: int = 1,
        ulysses_degree: int = 1,
    ):
        self.sp_degree = sp_degree
        self.ulysses_degree = ulysses_degree
        self.pad_to_length = pad_to_length

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad to the longest sequence in the batch.

        Args:
            intensors: (B, n, S)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N, S)
        """
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, S, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad 2D tensor to the longest sequence in the batch.

        Args:
            intensors: (B, n)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N)
        """
        B, n = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of features.

        Args:
            features: A list of features, where each feature is a dictionary containing:
                - input_ids: torch.Tensor of shape (n,)
                - attention_mask: torch.Tensor of shape (n,)
                - loss_mask: torch.Tensor of shape (n,)

        Returns:
            A dictionary containing:
                - input_ids: torch.Tensor of shape (B, N)
                - attention_mask: torch.Tensor of shape (B, N)
                - loss_mask: torch.Tensor of shape (B, N)
        """
        max_length = max(item["input_ids"].shape[1] for item in features)
        if self.pad_to_length is not None:
            if max_length > self.pad_to_length:
                raise ValueError(
                    f"Batch length {max_length} exceeds fixed padding length "
                    f"{self.pad_to_length}."
                )
            max_length = self.pad_to_length

        # pad for sequence parrel
        max_length = (
            (max_length + self.sp_degree - 1) // self.sp_degree
        ) * self.sp_degree
        # position max len, ulysses do not need chuck position ids
        position_max_len = max_length * self.ulysses_degree

        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        if "position_ids" in features[0]:
            batch_position_ids = torch.cat(
                [
                    self.paddingtensor2D(item["position_ids"], position_max_len)
                    for item in features
                ]
            )
        else:
            batch_position_ids = None
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_state": None,
            "target": None,
        }
        if batch_position_ids is not None:
            batch["position_ids"] = batch_position_ids
        if all("hidden_state" in item for item in features):
            assert all("target" in item for item in features), (
                "target is required when hidden_state is provided"
            )
            if self.sp_degree > 1:  # USP mode
                batch["hidden_state"] = torch.cat(
                    [item["hidden_state"] for item in features]
                )
            else:
                batch["hidden_state"] = torch.cat(
                    [
                        self.paddingtensor(item["hidden_state"], max_length)
                        for item in features
                    ]
                )
            batch["target"] = torch.cat(
                [self.paddingtensor(item["target"], max_length) for item in features]
            )
        return batch


class VlmDataCollatorWithPadding:
    """
    Datacollator that will dynamically pad the inputs for batching.
    """

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad to the longest sequence in the batch.

        Args:
            intensors: (B, n, S)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N, S)
        """
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad 2D tensor to the longest sequence in the batch.

        Args:
            intensors: (B, n)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N)
        """
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of features.

        Args:
            features: A list of features, where each feature is a dictionary containing:
                - input_ids: torch.Tensor of shape (n,)
                - attention_mask: torch.Tensor of shape (n,)
                - loss_mask: torch.Tensor of shape (n,)
                - pixel_values: torch.Tensor of shape (grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
                - image_grid_thw: torch.Tensor of shape (3,)

        Returns:
            A dictionary containing:
                - input_ids: torch.Tensor of shape (B, N)
                - attention_mask: torch.Tensor of shape (B, N)
                - loss_mask: torch.Tensor of shape (B, N)
        """
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        batch_pixel_values = torch.cat(
            [item["pixel_values"] for item in features], dim=0
        )
        batch_image_grid_thw = torch.cat(
            [item["image_grid_thw"] for item in features], dim=0
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw,
            "hidden_state": None,
            "target": None,
        }
        if all("hidden_state" in item for item in features):
            assert all("target" in item for item in features), (
                "target is required when hidden_state is provided"
            )
            batch["hidden_state"] = torch.cat(
                [
                    self.paddingtensor(item["hidden_state"], max_length)
                    for item in features
                ]
            )
            batch["target"] = torch.cat(
                [self.paddingtensor(item["target"], max_length) for item in features]
            )
        return batch
