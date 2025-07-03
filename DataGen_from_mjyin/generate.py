import logging
import os
import torch

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
from accelerate import Accelerator

@dataclass
class CustomTrainingArguments(TrainingArguments):
    ckpt_path: str = field(
        default="outputs/amazon-software/checkpoint-18000",
        metadata={"help": "The path of the pre-trained data generator."}
    )
    dataset_name: Optional[str] = field(
        default="amazon-software",
        metadata={"help": "The name of the dataset to use."}
    )
    max_length: Optional[int] = field(
        default=100,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    max_new_tokens: Optional[int] = field(
        default=1,
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

def add_new_seq(sample, model, tokenizer, max_new_tokens):
    input_ids = tokenizer.encode(sample['item_sequence'], return_tensors="pt")[:, :-1]
    generated_tensors = model.generate(
        input_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    ) # Remove Bos token
    new_seq = generated_tensors[0, 1:].tolist()
    # new_seq = tokenizer.decode(generated_tensors, skip_special_tokens=True)
    new_seq = [_ - 4 + 1 for _ in new_seq] # Hard-coded -3 is for the 4 special tokens in the tokenizer
    sample['new_seq'] = new_seq
    return sample

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
    user_seq = tokenizer.encode(sample['item_sequence'], add_special_tokens=True, return_tensors="pt",)[:, :-1]
    # user_seq = [_ - 4 + 1 for _ in user_seq] # Hard-coded -3 is for the 4 special tokens in the tokenizer
    sample['tokenized_item_sequence'] = user_seq
    return sample

def seq_to_RecBole_format(sample, indices):
    "user_id:token,item_id:token,rating:float,timestamp:float"
    user_seq = sample['new_seq']
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

    # Initialize Accelerator
    # This will automatically handle device placement and distributed setup
    accelerator = Accelerator()
    device = accelerator.device # Get the device assigned by Accelerator

    # Load the dataset
    dataset = load_dataset("parquet", data_files=os.path.join("outputs", "datasets", f"{training_args.dataset_name}.parquet"), split='train')
    logging.info("Datasets loaded")
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("outputs", "tokenizers", f"{training_args.dataset_name}"))

    ckpt_path = "outputs/amazon-software/checkpoint-18000"
    model = AutoModelForCausalLM.from_pretrained(training_args.ckpt_path)

    # Prepare model and tokenizer with Accelerator
    # This moves them to the correct device and sets up distributed training/inference if applicable
    model, tokenizer = accelerator.prepare(model, tokenizer)

    model.eval() # Ensure model is in evaluation mode

    # Pre-process the dataset: drop_last_n_items first
    dataset = dataset.map(
        partial(drop_last_n_items, drop_last_n=2),
        num_proc=32 if accelerator.is_main_process else 1, # Only main process needs full dataset for map
        load_from_cache_file=True,
    )

    # Convert to PyTorch Dataset to use DataLoader
    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset):
            self.hf_dataset = hf_dataset

        def __len__(self):
            return len(self.hf_dataset)

        def __getitem__(self, idx):
            return self.hf_dataset[idx]

    inference_hf_dataset = InferenceDataset(dataset)

    def custom_collate_fn(batch_samples):
        item_sequences = [sample['item_sequence'] for sample in batch_samples]
        tokenized_inputs = tokenizer(
            item_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        )
        return tokenized_inputs

    inference_dataloader = torch.utils.data.DataLoader(
        inference_hf_dataset,
        batch_size=2048,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )

    # Prepare the DataLoader with Accelerator
    inference_dataloader = accelerator.prepare(inference_dataloader)
    
    print(f"Starting generation on {accelerator.num_processes} GPU(s)...")
    all_new_seqs = []
    with torch.no_grad():
        for batch in tqdm(inference_dataloader, desc="Generating sequences", disable=not accelerator.is_main_process):
            # Inputs are already on the correct device thanks to accelerator.prepare(dataloader)
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask

            generated_tensors = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=training_args.max_new_tokens,
            )

            # Gather results from all processes if running in multi-GPU setup
            # This is important for ensuring all results are collected
            generated_tensors = accelerator.gather(generated_tensors)


            # Process generated sequences from the batch
            for gen_seq_single in generated_tensors:
                new_seq = gen_seq_single.tolist()
                num_special_token = 4
                new_seq = [token - num_special_token + 1 for token in new_seq if token not in range(num_special_token)]
                all_new_seqs.append({'new_seq': new_seq})

    # Only the main process saves the combined results
    if accelerator.is_main_process:
        new_seq_dataset = Dataset.from_list(all_new_seqs)
        dataset_with_new_seq = dataset.add_column("new_seq", [d['new_seq'] for d in all_new_seqs])

        train_dataset = dataset_with_new_seq.map(
            seq_to_RecBole_format,
            with_indices=True,
            num_proc=32,
            load_from_cache_file=True,
            remove_columns=dataset.column_names + ['new_seq'],
        )
        train_dataset.to_csv(f'data_generated/{training_args.dataset_name}-gen/{training_args.dataset_name}-gen.train.inter')

    logging.info("Done!")