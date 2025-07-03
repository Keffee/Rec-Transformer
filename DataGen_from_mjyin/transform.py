import logging
import os

from functools import partial
from RecModel import RecModel
from RecDataset import create_dataset
from RecTokenizer import create_rec_tokenizer
from tokenizers import Tokenizer
from transformers import TrainingArguments, HfArgumentParser, Trainer, Qwen3ForCausalLM, Qwen3Config, AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional
from utils import update_config_by_default_data_config
from tqdm import tqdm
from datasets import load_dataset, Dataset

@dataclass
class CustomTrainingArguments(TrainingArguments):
    dataset_name: Optional[str] = field(
        default="amazon-software",
        metadata={"help": "The name of the dataset to use."}
    )
    max_length: Optional[int] = field(
        default=100,
        metadata={"help": "The maximum total input sequence length after tokenization."}
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

def seq_to_HSTU_format(sample, indices, tokenizer, split='train'):
    "index,user_id,sequence_item_ids,sequence_ratings,sequence_timestamps,sex,age_group,occupation,zip_code"
    def list_to_string(x):
        return ','.join([str(_) for _ in x])
    sample['index'] = indices
    sample['user_id'] = indices + 1
    sample['sequence_item_ids'] = list_to_string([tokenizer.vocab[_] for _ in (sample['item_sequence']).split()])
    sample['sequence_ratings'] = list_to_string([5] * len(sample['item_sequence'].split()))
    sample['sequence_timestamps'] = list_to_string(sample['timestamp_sequence'])
    sample['sex'] = 1
    sample['age_group'] = 2
    sample['occupation'] = 20
    sample['zip_code'] = 3090
    return sample

def tokenize(sample, tokenizer):
    "user_id:token,item_id:token,rating:float,timestamp:float"
    user_seq = tokenizer.encode(sample['item_sequence'], add_special_tokens=False)
    user_seq = [_ - 4 + 1 for _ in user_seq] # Hard-coded -3 is for the 4 special tokens in the tokenizer
    sample['tokenized_item_sequence'] = user_seq
    return sample

def seq_to_RecBole_format(sample, indices):
    "user_id:token,item_id:token,rating:float,timestamp:float"
    user_seq = sample['tokenized_item_sequence']
    user_seq_str = ' '.join([str(_) for _ in user_seq[:-1]])
    sample['user_id:token'] = indices + 1
    sample['item_id_list:token_seq'] = user_seq_str
    sample['item_id:token'] = user_seq[-1]
    return sample

def drop_last_n_items(sample, drop_last_n):
    for k, v in sample.items():
        if k.endswith('sequence'):
            assert drop_last_n != 0
            if type(v) is list:
                sample[k] = v[:-drop_last_n]
            elif type(v) is str:
                sample[k] = ' '.join(v.split()[:-drop_last_n])
    return sample

if __name__ == "__main__":
    logging.info("Initializing the recommendation dataset and tokenizer...")
    parser = HfArgumentParser(CustomTrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    # training_args = update_config_by_default_data_config(training_args)
    # Load the dataset
    dataset = load_dataset("parquet", data_files=os.path.join("outputs", "datasets", f"{training_args.dataset_name}.parquet"), split='train')
    logging.info("Datasets loaded")
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("outputs", "tokenizers", f"{training_args.dataset_name}"))
    # dataset = dataset.map(partial(add_new_seq, tokenizer=tokenizer), load_from_cache_file=True, num_proc=32)
    # Generating RecBole data files
    logging.warning("!!! We hard-coded the number of special tokens in the data transform process.")
    tokenized_dataset = dataset.map(
        partial(tokenize, tokenizer=tokenizer),
        num_proc=64,
        load_from_cache_file=True,
    )
    train_dataset = tokenized_dataset.map(
        partial(drop_last_n_items, drop_last_n=2),
        num_proc=64,
        load_from_cache_file=True,
    ).map(
        seq_to_RecBole_format,
        with_indices=True,
        num_proc=64,
        load_from_cache_file=True,
        remove_columns=tokenized_dataset.column_names,
    )
    train_dataset.to_csv(f'data_transformed/{training_args.dataset_name}/{training_args.dataset_name}.train.inter')

    valid_dataset = tokenized_dataset.map(
        partial(drop_last_n_items, drop_last_n=1),
        num_proc=64,
        load_from_cache_file=True,
    ).map(
        seq_to_RecBole_format,
        with_indices=True,
        num_proc=64,
        load_from_cache_file=True,
        remove_columns=tokenized_dataset.column_names,
    )
    valid_dataset.to_csv(f'data_transformed/{training_args.dataset_name}/{training_args.dataset_name}.valid.inter')

    test_dataset = tokenized_dataset.map(
        seq_to_RecBole_format,
        with_indices=True,
        num_proc=64,
        load_from_cache_file=True,
        remove_columns=tokenized_dataset.column_names,
    )
    test_dataset.to_csv(f'data_transformed/{training_args.dataset_name}/{training_args.dataset_name}.test.inter')
    logging.info("Done!")