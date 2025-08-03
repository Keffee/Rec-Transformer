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
from transformers.models.llama_rec.tokenization_llamarec_try import create_hybrid_item_tokenizer, MockTrainingArguments 
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
        sequences_as_int = [e["sequence"] for e in examples]
        sequences_as_str = [[str(item_id) for item_id in seq] for seq in sequences_as_int]

        batch_dict = self.tokenizer(
            sequences_as_str,
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
        # 1. 分离输入和标签 (原始 item ID)
        input_sequences_as_int = [e["sequence"][:-1] for e in examples]
        eval_labels_as_int = [e["sequence"][-1] for e in examples]
        
        # 2. 将输入序列的原始 ID 转换为字符串
        input_sequences_as_str = [[str(item_id) for item_id in seq] for seq in input_sequences_as_int]
        
        # 3. 使用 tokenizer 对输入序列进行编码、截断和填充
        batch = self.tokenizer(
            input_sequences_as_str,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 4. 将评估标签的原始 ID 转换为 Token ID
        eval_labels_as_token_ids = self.tokenizer.convert_tokens_to_ids(
            [str(item_id) for item_id in eval_labels_as_int]
        )
        
        # --- [核心修正] 创建一个与 input_ids 形状相同的 labels 张量 ---
        
        # # 首先，创建一个全是 -100 的张量
        # labels = torch.full_like(batch['input_ids'], -100)
        
        # # 然后，只在每个序列的最后一个有效位置（非 padding 的位置）填上真实标签
        # # 我们需要找到每个序列的长度
        # sequence_lengths = batch['attention_mask'].sum(dim=1)
        
        # labels[:, -1] = torch.tensor(eval_labels_as_token_ids)
        # # for i in range(len(examples)):
        # #     # 最后一个有效 token 的索引是 length - 1
        # #     last_token_idx = sequence_lengths[i] - 1
        # #     # 在该位置填上真实的目标 Token ID
        # #     labels[i, last_token_idx] = eval_labels_as_token_ids[i]
            
        batch['labels'] = torch.tensor(eval_labels_as_token_ids)
        
        return batch


import time
# --- 5. 评估指标计算函数 ---
# 这个目前先不用，之前一直是这个地方卡手了，改进一下
def compute_metrics(eval_preds: EvalPrediction):
    logits, labels_matrix = eval_preds
    
    # 检查是否有可用的 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 将数据转换为 PyTorch 张量并移动到 GPU
    # 我们只关心最后一个时间步的 logit
    last_step_logits = torch.from_numpy(logits[:, -1, :]).to(device)
    
    # 从 labels_matrix 中提取出有效的标签
    labels = torch.from_numpy(labels_matrix).view(-1).to(device)
    
    # 2. [健壮性检查] 过滤掉不需要计算的 -100 标签
    valid_mask = labels != -100
    labels = labels[valid_mask]
    last_step_logits = last_step_logits[valid_mask]
    
    # 如果过滤后没有有效标签，则直接返回空字典
    if labels.numel() == 0:
        return {}

    # 3. [核心优化] 在 GPU 上执行高效的排序和排名查找
    # torch.sort 在 GPU 上非常快
    sorted_indices = torch.argsort(last_step_logits, descending=True, dim=-1)
    
    # 使用 broadcast 和 a==b 的方式高效查找 rank
    ranks = (sorted_indices == labels.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1

    # 4. 计算指标 (可以将结果移回 CPU)
    ranks = ranks.float() # 转换为浮点数以进行后续计算
    
    metrics = {}
    for k in [1, 5, 10, 20, 50]:
        hr_k = (ranks <= k).float().mean().item()
        metrics[f"HR@{k}"] = round(hr_k, 4)
        
        in_top_k = (ranks <= k)
        ndcg_k = (1.0 / torch.log2(ranks + 1.0)).where(in_top_k, 0.0).mean().item()
        metrics[f"NDCG@{k}"] = round(ndcg_k, 4)

    mrr = (1.0 / ranks).mean().item()
    metrics["MRR"] = round(mrr, 4)
    
    return metrics

# 改进成下面这个：
class StreamingMetricsCalculator:
    """
    一个支持流式评估的指标计算器，专为推荐任务设计。

    该类通过在每个评估批次上计算并累积轻量级的“排名(ranks)”信息，
    避免了在内存中保留巨大的 logits 张量，从而解决了评估过程中的性能瓶颈和卡顿问题。
    它与 Hugging Face Trainer 的 `batch_eval_metrics=True` 参数协同工作。
    """
    def __init__(self, k_values: List[int] = [1, 5, 10, 20, 50]):
        """
        初始化计算器。

        Args:
            k_values (List[int]): 用于计算 HR@k 和 NDCG@k 的 k 值列表。
        """
        self.k_values = k_values
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.all_ranks 将在 CPU 上累积，以节省 GPU 显存
        self.all_ranks: List[torch.Tensor] = []

    def __call__(self, eval_preds: EvalPrediction, compute_result: bool) -> Dict[str, float]:
        """
        Trainer 在每个评估步骤调用的核心方法。

        Args:
            eval_preds (EvalPrediction): 包含当前批次的 predictions 和 label_ids 的对象。
            compute_result (bool): 由 Trainer 传入的标志。
                                   如果为 False，则只处理并累积当前批次。
                                   如果为 True (在最后一个批次)，则计算并返回最终指标。
        """
        # --- 1. 每个批次都会执行的部分：计算并累积 Ranks ---
        logits, labels_matrix = eval_preds.predictions, eval_preds.label_ids

        # 将当前批次的数据移动到GPU进行快速计算
        last_step_logits = logits[:, -1, :]
        labels = labels_matrix.view(-1)
        
        valid_mask = labels != -100
        labels = labels[valid_mask]
        last_step_logits = last_step_logits[valid_mask]

        # 如果这个批次没有有效标签，则直接跳过
        if labels.numel() > 0:
            # 在 GPU 上高效计算排名
            sorted_indices = torch.argsort(last_step_logits, descending=True, dim=-1)
            ranks = (sorted_indices == labels.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1
            
            # 将计算出的 ranks (小张量) 移回 CPU 并添加到列表中
            self.all_ranks.append(ranks.cpu())

        # --- 2. 仅在最后一个批次执行的部分：计算最终指标 ---
        if compute_result:
            if not self.all_ranks:
                return {} # 如果整个评估过程都没有有效标签

            # 将所有批次的 ranks 拼接成一个大张量
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
            
            # **非常重要**：清空状态，为下一次可能的评估做准备
            self.all_ranks = []
            
            return metrics
        
        # 如果不是最后一步，返回空字典
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

# --- 7. Main 函数 (DDP 多卡并行最终修正版 v2) ---
def main():
    # ... (参数解析和 YAML 加载部分保持不变) ...
    parser = argparse.ArgumentParser(description="Train a LlamaRec model using a YAML config file.")
    parser.add_argument("--config", type=str, required=True, help="Name of the config file. e.g., 'pantry'")
    cli_args = parser.parse_args()
    
    default_config_path = "/home/kfwang/20250613Rec-Factory/try_train/train_config/"
    with open(default_config_path + cli_args.config + '.yaml', 'r') as f:
        config_data = yaml.safe_load(f)

    paths_config = config_data['paths']
    model_params = config_data['model_params']
    training_args_dict = config_data['training_args']
    dataset_split_config = config_data['dataset_split']

    # 1. 提前创建 TrainingArguments 来初始化分布式环境
    output_dir = paths_config['output_dir']
    training_args_dict['output_dir'] = output_dir
    training_args_dict['logging_dir'] = os.path.join(output_dir, 'logs')
    training_args = TrainingArguments(**training_args_dict)

    # 2. 主进程执行一次性设置
    if training_args.local_rank <= 0:
        print(f"Main Process [Rank {training_args.local_rank}]: Running one-time setup...")
        
        raw_dataset = load_dataset("json", data_files=paths_config['dataset_path'], split="train")
        tokenizer_dir = paths_config['tokenizer_dir']
        tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")

        if not os.path.exists(tokenizer_file):
            print("Tokenizer not found. Creating a new one...")
            # ... (创建 tokenizer 的逻辑) ...
            def text_to_int_list(example): ...
            temp_dataset_with_list = raw_dataset.map(...)
            mock_args = MockTrainingArguments(output_dir=tokenizer_dir, max_length=model_params['max_seq_length'])
            # 这个函数会写入文件，所以必须在保护块内
            create_hybrid_item_tokenizer(dataset=temp_dataset_with_list, training_args=mock_args)
        
        print("Preprocessing and caching the dataset...")
        # .map() 会写入缓存，也必须在保护块内
        processed_dataset = raw_dataset.map(
            final_preprocess_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
            num_proc=4,
        )
        print("One-time setup complete.")

    # --- 3. 关键修复：添加同步屏障 ---
    # 这会强制所有子进程(rank > 0)在此处暂停，直到主进程(rank 0)也到达这里。
    # 此时，我们能确保主进程已经完成了上面所有的文件写入操作。
    if training_args.local_rank != -1:
        torch.distributed.barrier()
        print(f"Process [Rank {training_args.local_rank}]: Barrier passed. Files are ready.")

    # 4. 现在，所有进程都可以安全地加载文件和数据了
    print(f"Process [Rank {training_args.local_rank}]: Loading shared resources...")
    tokenizer_dir = paths_config['tokenizer_dir']
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    # 加载数据集 (所有进程都会执行，但会利用主进程创建的缓存)
    raw_dataset = load_dataset("json", data_files=paths_config['dataset_path'], split="train")
    processed_dataset = raw_dataset.map(
        final_preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )
    
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if tokenizer.bos_token_id is None: tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")

    split_dataset = processed_dataset.train_test_split(
        test_size=dataset_split_config['test_size'], seed=dataset_split_config['seed']
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # ... (模型创建、DataCollator 创建等逻辑完全不变)
    max_seq_length = model_params['max_seq_length']
    # <<< MODIFIED: LlamaRecConfig 现在从字典动态构建 >>>
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
    train_collator = TrainDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    streaming_metrics_calculator = StreamingMetricsCalculator()

    # --- Trainer 实例化 ---
    # 它会使用我们提前创建的 training_args 来完成分布式环境的最终设置
    trainer = CustomTrainer(
        model=model,
        args=training_args, # 使用已经创建好的 args
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=train_collator,
        compute_metrics=streaming_metrics_calculator,
        eval_collator=eval_collator,
    )

    # --- 训练和保存 ---
    print(f"Process [Rank {training_args.local_rank}]: Starting training...")
    trainer.train()

    # Trainer.save_model 和 is_world_process_zero 都是安全的
    final_model_path = os.path.join(paths_config['output_dir'], "final")
    trainer.save_model(final_model_path)
    if trainer.is_world_process_zero():
        print("Main Process: Saving final tokenizer.")
        tokenizer.save_pretrained(final_model_path)
    
    print(f"Process [Rank {training_args.local_rank}]: All operations complete!")

if __name__ == "__main__":
    main()