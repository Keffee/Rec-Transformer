import logging
import os

from functools import partial
from RecModel import RecModel
from RecDataset import create_dataset
from RecTokenizer import create_rec_tokenizer
from tokenizers import Tokenizer
from transformers import TrainingArguments, HfArgumentParser, Trainer, Qwen3ForCausalLM, Qwen3Config
from dataclasses import dataclass, field
from typing import Optional
from utils import update_config_by_default_data_config

@dataclass
class CustomTrainingArguments(TrainingArguments):
    dataset_name: Optional[str] = field(
        default="amazon-beauty",
        metadata={"help": "The name of the dataset to use."}
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the dataset."}
    )
    max_length: Optional[int] = field(
        default=100,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    k_core: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of interactions per user/item for k-core filtering."}
    )
    user_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the user column in the dataset."}
    )
    item_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the item column in the dataset."}
    )
    timestamp_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the timestamp column in the dataset."}
    )

def seq_to_string(sample):
    sample['item_sequence'] = ' '.join(map(str, sample['item_sequence']))
    return sample

def tokenize_function(examples, tokenizer : Tokenizer, max_length: int = 256):
    tokenized_data = tokenizer(
        examples["item_sequence"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
        padding_side="left",
    )
    labels = tokenized_data["input_ids"].clone()  # Clone to avoid modifying the original input_ids
    labels[labels == tokenizer.pad_token_id] = -100  # Set padding tokens to -100 for loss calculation
    return {
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "labels": labels,
    }

if __name__ == "__main__":
    logging.info("Initializing the recommendation dataset and tokenizer...")
    parser = HfArgumentParser(CustomTrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    training_args = update_config_by_default_data_config(training_args)
    # Load the dataset
    dataset = create_dataset(
        training_args.dataset_path,
        max_seq_len=training_args.max_length,
        user_column=training_args.user_column,
        item_column=training_args.item_column,
        timestamp_column=training_args.timestamp_column,
        k_core=training_args.k_core,
    )
    # Load the tokenizer and model
    tokenizer = create_rec_tokenizer(dataset, training_args)
    tokenizer.save_pretrained(os.path.join("outputs", "tokenizers", f"{training_args.dataset_name}"))
    # Tokenize the dataset
    dataset = dataset.map(seq_to_string, num_proc=32)
    dataset.to_parquet(os.path.join("outputs", "datasets", f"{training_args.dataset_name}.parquet"))
