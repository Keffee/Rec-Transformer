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
from utils import update_config_by_default_data_config, condense_seq_data
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

if __name__ == "__main__":
    logging.info("Initializing the recommendation dataset and tokenizer...")
    parser = HfArgumentParser(CustomTrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    # training_args = update_config_by_default_data_config(training_args)
    # Load the dataset
    dataset = load_dataset("csv", data_files=os.path.join("generated_data", f"{training_args.dataset_name}", f"{training_args.dataset_name}.train.inter"), split='train')
    logging.info("Datasets loaded")
    # Load the tokenizer and model
    # dataset = dataset.map(partial(add_new_seq, tokenizer=tokenizer), load_from_cache_file=True, num_proc=32)
    # Generating RecBole data files
    logging.warning("!!! We hard-coded the number of special tokens in the data transform process.")
    train_dataset = condense_seq_data(dataset, training_args.max_length)
    train_dataset.to_csv(f'generated_data/{training_args.dataset_name}/{training_args.dataset_name}.condense.inter')
