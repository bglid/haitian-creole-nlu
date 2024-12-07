import argparse
import os
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

import wandb
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from tqdm import tqdm
import evaluate


# setting our eval and tokenizer
accuracy = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


# defining our parsing function for CL interaction
def parse_args():
    parser = argparse.ArgumentParser()

    # args for Data

    parser.add_argument(
        "--data_dir", type=str, default="", help="path to directory with mctest files"
    )
    # we may not need the one below
    # parser.add_argument("--split", type=str, default=".train", help="")

    # Output & logging

    # parser.add_argument(
    #     "--tb_dir`", type=str, default="", help="for connecting to tensorboard"
    # )

    parser.add_argument(
        "--output_dir", type=str, default="./outputs/", help="output results"
    )

    # Model

    parser.add_argument("--tokenizer", type=str, default="xlm-roberta-base")
    parser.add_argument("--from_pretrained", type=str, default="xlm-roberta-base")
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

    # Evaluation
    parser.add_argument("--eval_batch_size", type=int, default=1)

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
    for q, a in zip(examples["question"], examples["text_answer"]):
        q2a_lut[q] = a

    # seperating tokens
    sep_token = tokenizer.sep_token

    # getting question answer
    for q, candidates in zip(examples["question"], examples["choices"]):
        answer = q2a_lut[q]
        # getting choices to tokenize
        for option in candidates:
            choices.append(f"{q} {sep_token} {option}")

    tokenized_examples = tokenizer(story, choices, truncation=True)
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
                # trunc_token_type_ids = encoded_input["token_type_ids"][: self.max_len]
                trunc_attention_mask = encoded_input["attention_mask"][: self.max_len]
                truncated_encoded_input = {
                    "input_ids": trunc_input_ids,
                    # "token_type_ids": trunc_token_type_ids,
                    "attention_mask": trunc_attention_mask,
                }
                truncated_features.append(truncated_encoded_input)

        # Padding the input
        batch = self.tokenizer.pad(
            truncated_features,
            padding="max_length",
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
    # setting the args and seed
    args = parse_args()
    seed = args.seed

    # setting the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # setting the device to run on gpu if avaialable
    device = torch.device(args.device)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)
        print(f"% of GPUs available: {torch.cuda.device_count()}")
        print(f"The GPU that will be used: {torch.cuda.get_device_name(0)}")
    else:
        # Here they kill the job if there isn't any gpus
        print("Running on CPU")
        device = torch.device("cpu")
        # they kill the job, but we will see if running on CPU is possible
        # print("Killing this job...")
        # exit(333)

    data_path = args.data_dir
    # split = args.split

    # getting examples
    examples = load_dataset(f"{data_path}")
    print("***EXAMPLES***")
    print(examples)

    tokenized_mct = examples.map(preprocess_function, batched=True)

    # setting up the model for training
    if args.action == "train":
        model = AutoModelForMultipleChoice.from_pretrained(args.from_pretrained).to(
            device
        )

        # reminder to set up Weights and Biases here~

        # early stopping for when the model converges early:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=10, early_stopping_threshold=0.001
        )

        # initializing wandb for tracking
        wandb.init(project="hc_nlu")

        # setting up training parameters from our parse_args
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir),  # took out experiment directory
            report_to="wandb",
            save_strategy="epoch",
            # evaluation_strategy="epoch",
            eval_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
            logging_steps=100,
            logging_first_step=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_mct["train"],
            eval_dataset=tokenized_mct["validation"],
            processing_class=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
            callbacks=[early_stopping],
            compute_metrics=compute_metrics,
        )

        print(f"DEVICE: {device}")
        if device == "cuda":
            model.cuda()

        trainer.train()

    elif args.action == "evaluate":
        # load the trained model
        model = AutoModelForMultipleChoice.from_pretrained(args.from_checkpoint).to(
            device
        )

        # initializing wandb for tracking
        wandb.init(project="hc_nlu")

        # omitting experimental sub directory for now
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir),  # took out experiment directory
            report_to="wandb",
            save_strategy="epoch",
            # evaluation_strategy="epoch",
            eval_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_mct["train"],
            eval_dataset=tokenized_mct["validation"],
            processing_class=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        )

        if device == "cuda":
            model.cuda()

        # Setting model to Eval
        model.eval()

        eval_loss = 0
        eval_accuracy = 0
        nb_eval_steps = 0
        nb_eval_examples = 0
        # nb_eval_batches = 0 #same as eval steps

        # preds_list = []
        # true_list = []

        # opening dataloader for dev/validation
        dev_dataloader = trainer.get_eval_dataloader()

        for batch in tqdm(dev_dataloader, desc=f"Evaluating: {data_path}"):
            # getting our input to evaluate for each batch
            with torch.no_grad():
                inputs = {
                    "input_ids": batch["input_ids"].to(args.device),
                    "attention_mask": batch["attention_mask"].to(args.device),
                    # "token_type_ids": batch["token_type_ids"].to(args.device),
                    "labels": batch["labels"].to(args.device),
                }

                # calculating output and loss:
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            # print(f"Logits: {logits}")
            # print(f"[eval] predictions: {preds}")
            # [preds_list.append(p) for p in preds]
            label_ids = inputs["labels"].to("cpu").numpy()
            print(f"[eval] labels: {label_ids}")
            # [true_list.append(label) for label in label_ids]

            tmp_eval_accuracy = (preds == label_ids).astype(np.float32).mean().item()

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            nb_eval_examples += inputs["input_ids"].size(0)
            # nb_eval_batches += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        result = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
        print(result)

    elif args.action == "test":
        model = AutoModelForMultipleChoice.from_pretrained(args.from_checkpoint).to(
            device
        )

        # not adding sub directory for time being

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir),  # took out experiment directory
            save_strategy="epoch",
            # evaluation_strategy="epoch",
            eval_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_mct["train"],
            eval_dataset=tokenized_mct["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        )

        if device == "cuda":
            model.cuda()

        # Setting model to Eval
        model.eval()

        test_loss = 0
        test_accuracy = 0
        nb_test_steps = 0
        nb_test_examples = 0
        # nb_test_batches = 0

        # The following two variables don't get utilized
        # preds_list = []
        # true_list = []

        test_dataloader = trainer.get_eval_dataloader()

        for batch in tqdm(test_dataloader, desc=f"Evaluating TEST data: {data_path}"):
            # getting our input to evaluate for each batch
            with torch.no_grad():
                inputs = {
                    "input_ids": batch["input_ids"].to(args.device),
                    "attention_mask": batch["attention_mask"].to(args.device),
                    # "token_type_ids": batch["token_type_ids"].to(args.device),
                    "labels": batch["labels"].to(args.device),
                }

                # calculating output and loss:
                outputs = model(**inputs)
                tmp_test_loss, logits = outputs[:2]
                test_loss += tmp_test_loss.mean().item()

            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            print(f"[test] predictions: {preds}")
            # [preds_list.append(p) for p in preds]
            label_ids = inputs["labels"].to("cpu").numpy()
            print(f"[test] labels: {label_ids}")
            # [true_list.append(label) for label in label_ids]

            tmp_test_accuracy = (preds == label_ids).astype(np.float32).mean().item()

            test_accuracy += tmp_test_accuracy
            nb_test_steps += 1  # num batches
            nb_test_examples += inputs["input_ids"].size(0)
            # nb_test_batches += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_steps
        result = {"test_loss": test_loss, "test_accuracy": test_accuracy}
        print(result)

    else:
        print("Supported actions are only 'train', 'evaluate', or 'test'")


if __name__ == "__main__":
    main()
