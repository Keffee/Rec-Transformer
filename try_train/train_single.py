# train.py (最终版本)

import os
import logging
from typing import Dict, List, Union, Optional, Any
from datasets import load_dataset, Dataset
from functools import partial
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse

# 导入你的自定义代码
from transformers.models.llama_rec.tokenization_llamarec import create_rq_code_tokenizer, MockTrainingArguments 
from transformers.models.llama_rec.modeling_llamarec import LlamaRecForCausalLM, LlamaRecConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
# --- 1. 注册模型 ---
MODEL_TYPE = "llama-rec"
# AutoConfig.register(MODEL_TYPE, LlamaRecConfig)
# AutoModelForCausalLM.register(LlamaRecConfig, LlamaForRec)


# --- 2. 预处理函数 ---
# --- [最终版] 预处理函数 ---
def final_preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
    """将文本序列 '1,2,3' 转换为整数列表 [1, 2, 3]"""
    all_sequences = []
    for text in examples["text"]:
        item_ids = [int(i.strip()) for i in text.split(',') if i.strip()]
        if len(item_ids) > 1: # 确保序列至少有两个 item，才能用于留一法
            all_sequences.append(item_ids)
    return {"sequence": all_sequences}

# --- [最终版] 训练数据整理器 ---
class TrainDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        sequences = [e["text"] for e in examples]
        # sequences_as_str = [[str(item_id) for item_id in seq] for seq in sequences_as_int]

        batch_dict = self.tokenizer(
            sequences,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        batch_dict['labels'] = batch_dict['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            batch_dict["labels"][batch_dict["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch_dict

# --- [最终版] 评估数据整理器 ---

class EvalDataCollator: 
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        sequences = [e["text"] for e in examples]
        # sequences_as_str = [[str(item_id) for item_id in seq] for seq in sequences_as_int]

        batch_dict = self.tokenizer(
            sequences,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        batch_dict['labels'] = batch_dict['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            batch_dict["labels"][batch_dict["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch_dict

# 流式指标
class StreamingMetricsCalculator:   # 这里也用了默认3的设定，看到3要谨慎
    def __init__(self, k_values: List[int] = [1, 5, 10, 20, 50]):
        """
        初始化计算器。

        Args:
            k_values (List[int]): 用于计算 HR@k 和 NDCG@k 的 k 值列表。
        """
        self.k_values = k_values
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_ranks: List[torch.Tensor] = []

    def __call__(self, eval_preds: EvalPrediction, compute_result: bool) -> Dict[str, float]:
        logits, labels_matrix = eval_preds.predictions, eval_preds.label_ids

        num_eval_steps = 3 # 留一法，最后3个token是eval的

        last_step_logits = logits[0][-num_eval_steps-1:-1, :]
        labels = labels_matrix.view(-1)[-num_eval_steps:]
        
        valid_mask = labels != -100
        labels = labels[valid_mask]
        last_step_logits = last_step_logits[valid_mask]

        # 如果这个批次没有有效标签，则直接跳过
        if labels.numel() > 0:
            sorted_indices = torch.argsort(last_step_logits, descending=True, dim=-1)
            ranks = (sorted_indices == labels.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1
            
            self.all_ranks.append(ranks.cpu())
        #print('compute_result: ', compute_result)
        if compute_result:
            #print('---------comp')
            if not self.all_ranks:
                return {} # 如果整个评估过程都没有有效标签

            final_ranks = torch.cat(self.all_ranks).float()
            
            metrics = {}
            for k in self.k_values:
                in_top_k = final_ranks <= k
                hr_k = in_top_k.float().mean().item()
                metrics[f"HR@{k}"] = round(hr_k, 4)
                
                # 计算 NDCG
                ndcg_k = (1.0 / torch.log2(final_ranks + 1.0)).where(in_top_k, 0.0).mean().item()
                metrics[f"NDCG@{k}"] = round(ndcg_k, 4)

            metrics["MRR"] = round((1.0 / final_ranks).mean().item(), 4)
            print(metrics)
            self.all_ranks = []
            
            return metrics
        
        return {}

# --- 6. 自定义 Trainer ---
# 这个 Trainer 可以确保评估时使用我们自定义的 EvalDataCollator
class CustomTrainer(Trainer):
    def __init__(self, *args, eval_collator, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.custom_eval_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# --- 7. Main 函数 (从 YAML 读取配置的最终版本) ---
def main():
    # <<< 新增: 解析命令行参数以获取配置文件路径 >>>
    parser = argparse.ArgumentParser(description="Train a LlamaRec model using a YAML config file.")
    parser.add_argument("--config", type=str, required=True, help="Name of the config file to use. For example: pantry")
    cli_args = parser.parse_args()

    # <<< 新增: 读取并解析 YAML 配置文件 >>>
    print(f"Loading configuration from: {cli_args.config}")
    current_dir_name = os.path.dirname(os.path.abspath("__file__"))
    config_path = os.path.join(current_dir_name, "pretrain_config", cli_args.config+'.yaml')
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # <<< 新增: 从解析的数据中提取配置组 >>>
    paths_config = config_data['paths']
    model_params = config_data['model_params']
    training_args_dict = config_data['training_args']
    # dataset_split_config = config_data['dataset_split']

    # 使用从配置中读取的参数
    dataset_path = dict(
        train=paths_config['train_dataset_path'],
        test=paths_config['test_dataset_path']
    )
    output_dir = paths_config['output_dir']
    tokenizer_dir = paths_config['tokenizer_dir']
    max_seq_length = model_params['max_seq_length']

    # <<< MODIFIED: Tokenizer 创建逻辑现在使用配置中的路径 >>>
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    raw_dataset = load_dataset("json", data_files=dataset_path, split="test")
    # (这部分创建 tokenizer 的逻辑不变，只是使用了来自 config 的变量)
    if not os.path.exists(tokenizer_file):
        print("Tokenizer not found. Creating a new one from the RQ code dataset...")
        mock_args = MockTrainingArguments(output_dir=tokenizer_dir, max_length=max_seq_length)
        tokenizer = create_rq_code_tokenizer(dataset=raw_dataset, training_args=mock_args)
    else:
        print("Found existing tokenizer. Loading it...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    # (健壮性检查逻辑不变)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if tokenizer.bos_token_id is None: tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    if tokenizer.eos_token_id is None: tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    assert tokenizer.pad_token_id is not None and tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None
    print(f"Final check - pad_token_id: {tokenizer.pad_token_id}, bos_token_id: {tokenizer.bos_token_id}, eos_token_id: {tokenizer.eos_token_id}")

    # # <<< MODIFIED: 数据集划分现在使用配置中的参数 >>>
    # print("Preprocessing and splitting the dataset...")
    # processed_dataset = raw_dataset.map(
    #     final_preprocess_function,
    #     batched=True,
    #     remove_columns=raw_dataset.column_names,
    #     num_proc=4,
    # )
    # split_dataset = processed_dataset.train_test_split(
    #     test_size=dataset_split_config['test_size'], 
    #     seed=dataset_split_config['seed']
    # )
    train_dataset = load_dataset("json", data_files=dataset_path, split="train")
    test_dataset = load_dataset("json", data_files=dataset_path, split="test")
    # eval_dataset = split_dataset["test"]
    # print(f"Train dataset size: {len(train_dataset)}, Evaluation dataset size: {len(eval_dataset)}")

    # <<< MODIFIED: LlamaRecConfig 现在从字典动态构建 >>>
    print("Creating LlamaRecForCausalLM model from scratch...")
    config = LlamaRecConfig(
        # 从 model_params 读取架构参数
        hidden_size=model_params['hidden_size'],
        intermediate_size=model_params['intermediate_size'],
        num_hidden_layers=model_params['num_hidden_layers'],
        num_attention_heads=model_params['num_attention_heads'],
        max_position_embeddings=max_seq_length,
        rms_norm_eps=model_params['rms_norm_eps'],
        # 运行时确定的参数
        model_type=MODEL_TYPE,
        vocab_size=len(tokenizer),
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaRecForCausalLM(config)
    print(f"Model created with {model.num_parameters() / 1e6:.2f} M parameters.")

    # <<< MODIFIED: TrainingArguments 现在从字典动态构建 >>>
    # 将 YAML 中未包含的、依赖于路径的参数添加到字典中
    training_args_dict['output_dir'] = output_dir
    training_args_dict['logging_dir'] = os.path.join(output_dir, 'logs')
    # 使用字典解包来创建 TrainingArguments 实例
    training_args = TrainingArguments(**training_args_dict)

    # (DataCollator 和 Trainer 的实例化逻辑不变)
    train_collator = TrainDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    
    streaming_metrics_calculator = StreamingMetricsCalculator() 

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=train_collator,
        compute_metrics=streaming_metrics_calculator,
        eval_collator=eval_collator,
    )

    chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.chat_template = chat_template

    # (训练和保存逻辑不变)
    print("Starting training with proper Leave-One-Out evaluation...")
    trainer.train()

    final_model_path = os.path.join(output_dir, "final")
    print(f"Training complete. Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("All operations complete!")

if __name__ == "__main__":
    main()