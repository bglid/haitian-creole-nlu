import argparse
import os
import csv
import json
import random
from _pytest.config import argparsing
import numpy as np
import transformers
from transformers import AutoTokenizer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict, load_dataset
from dataclasses import dataclass
from transformers.generation import candidate_generator
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)
from typing import Optional, Union
import torch

# we won't be using the Tensorboard

from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from tqdm import tqdm, trange
import evaluate

# setting our eval and tokenizer
accuracy = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

# setting debugging logging incase we need it
transformers.logging.set_verbosity_debug()


# defining our parsing function for CL interaction
def parse_args():
    parser = argparse.ArgumentParser()

    # args for Data

    parser.add_argument(
        "--data_dir", type=str, default="", help="path to directory with mctest files"
    )
    # we may not need the one below
    # parser.add_argument(
    #     "--split", type=str, default="mc500.train", help=""
    # )

    # Output & logging

    # parser.add_argument(
    #     "--tb_dir`", type=str, default="", help="for connecting to tensorboard"
    # )
    # parser.add_argument("--output_dir", type=str, default="", help="output results")

    # Model

    parser.add_argument(
        "--tokenizer", type=str, default="bert-base-multilingual-uncased"
    )
    parser.add_argument(
        "--from_pretrained", type=str, default="bert-base-multilingual-uncased"
    )
    parser.add_argument("--from_checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # Training
    parser.add_argument(
        "--action", type=str, default="train", choices=["train", "evaluate", "test"]
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="former default was 5e-5"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)

    return parser.parse_args()


# function for computing evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# defining our preprocessing for this model
def preprocess_function(examples):
    story = [[context] * 4 for context in examples["story"]]
    story = sum(story, [])
    choices = []

    q2a_lut = {}
    for q, a in zip(examples["questions"], examples["text_answer"]):
        q2a_lut[q] = a

    # seperating tokens
    sep_token = tokenizer.sep_token

    # getting question answer
    for q, candidates in zip(examples["question"], examples["choices"]):
        answer = q2a_lut[q]
        # getting choices to tokenize
        for option in candidates:
            choices.append(f"{q} {sep_token} {option}")

    tokenized_examples = tokenizer(story, choices, truncation=False)
    bb = {
        k: [v[i : i + 4] for i in range(0, len(v), 4)]
        for k, v in tokenized_examples.items()
    }
    return bb


# Data collator to be called for multiple choice
@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    max_len = 512

    # function that will be called upon class creation in running model
    def __call__(self, features):
        label_name = "label"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = 4
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        # encoding the inputs
        flattened_features = sum(flattened_features, [])
        truncated_features = []

        for encoded_input in flattened_features:
            if len(encoded_input["input_ids"]) < self.max_len:
                truncated_features.append(encoded_input)
            else:
                trunc_input_ids = encoded_input["input_ids"][: self.max_len]
                trunc_token_type_ids = encoded_input["token_type_ids"][: self.max_len]
                trunc_attention_mask = encoded_input["attention_mask"][: self.max_len]
                truncated_encoded_input = {
                    "input_ids": trunc_input_ids,
                    "token_type_ids": trunc_token_type_ids,
                    "attention_mask": trunc_attention_mask,
                }
                truncated_features.append(truncated_encoded_input)

        # Padding the input
        batch = self.tokenizer.pad(
            truncated_features,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        labels_as_ints = [int(label) for label in labels]
        batch["labels"] = torch.tensor(labels_as_ints, dtype=torch.int64)
        input_ids = batch["input_ids"]
        return batch


# main function to train/eval the model
def main():
    pass


if __name__ == "__main__":
    main()
