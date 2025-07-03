import os
import logging

from functools import partial
from RecModel import RecModel
from RecDataset import create_dataset
from RecTokenizer import create_rec_tokenizer
from tokenizers import Tokenizer
from transformers import TrainingArguments, HfArgumentParser, Trainer, Qwen3ForCausalLM, Qwen3Config, AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional
from utils import update_config_by_default_data_config, drop_last_n_items
from datasets import load_dataset

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
    """
    In pre-training, we do not need to add the EOS token, since the behavior sequence of the user is not supposed to end.
    """
    tokenized_data = tokenizer(
        examples["item_sequence"],
        truncation=True,
        max_length=max_length + 1, # +1 for the removal of EOS token
        padding="max_length",
        return_tensors="pt",
        padding_side="left",
    )
    tokenized_data = {k: v[:, :-1] for k, v in tokenized_data.items()}  # Remove EOS token from all sequences
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
    # Load the tokenizer and model
    # tokenizer = create_rec_tokenizer(dataset, training_args)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("outputs", "tokenizers", f"{training_args.dataset_name}"))
    model = RecModel(
        config=Qwen3Config(
            vocab_size=tokenizer.vocab_size,  # 使用你的Tokenizer词汇表大小
            hidden_size=64,                  # 隐藏层维度 (e.g., Qwen3-0.5B is 1024)
            intermediate_size=256,           # FFN中间层维度
            num_hidden_layers=2,             # Transformer层数 (e.g., Qwen3-0.5B is 24)
            num_attention_heads=4,           # 注意力头数
            num_key_value_heads=4,           # GQA/MQA的KV头数 (保持与attention_heads一致以使用MHA)
            max_position_embeddings=training_args.max_length,      # 与数据处理中的max_length保持一致
            rope_theta=10000.0,
            use_cache=False,                  # 训练时关闭cache
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attn_implementation="flash_attention_2",
            torch_dtype="bfloat16",
        )
    )
    # Load the dataset
    dataset = load_dataset("parquet", data_files=os.path.join("outputs", "datasets", f"{training_args.dataset_name}.parquet"), split='train')
    tokenized_datasets = dataset.map(
        partial(drop_last_n_items, drop_last_n=2),
        num_proc=32,
        load_from_cache_file=True,
    ).map(
        partial(tokenize_function, tokenizer=tokenizer, max_length=training_args.max_length),
        remove_columns=dataset.column_names,
        batched=True,
        load_from_cache_file=True
    )
    tokenized_datasets.set_format("torch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer, # 用于 DataCollatorWithPadding
    )

    logging.info("Training the model...")
    logging.warning("!!! We hard-coded the number of special tokens in the negative sampling process.")
    trainer.train()