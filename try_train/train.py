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

# 导入你的自定义代码
from transformers.models.llama_rec.tokenization_llamarec_try import create_hybrid_item_tokenizer, MockTrainingArguments 
from transformers.models.llama_rec.modeling_llamarec_try import LlamaForRec, LlamaRecConfig
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


# --- 5. 评估指标计算函数 ---
def compute_metrics(eval_preds: EvalPrediction):    # 经过逐行比对，这个metric应该没问题，问题应该出在训练部分
    logits, labels_matrix = eval_preds
    
    # logits: [batch_size, seq_len, vocab_size]
    # labels_matrix: [batch_size, seq_len]
    
    # 我们只关心最后一个时间步的 logit
    last_step_logits = torch.from_numpy(logits)
    
    # 从 labels_matrix 中提取出有效的标签
    # 有效标签是不等于 -100 的值
    # 因为我们只在最后一个位置放了有效标签，所以可以直接取最后一列
    labels = torch.from_numpy(labels_matrix).view(-1)
    
    # [健壮性检查] 确保我们真的拿到了有效标签
    # 如果 labels 里还有 -100，说明我们的逻辑有误
    if (labels == -100).any():
        print("Warning: Found -100 in the final labels slice, something is wrong with collator alignment.")
        # 我们只对有效的进行计算
        valid_mask = labels != -100
        labels = labels[valid_mask]
        last_step_logits = last_step_logits[valid_mask]

    # [最终修正] 使用 torch.sort 进行稳健的排名
    sorted_logits, sorted_indices = torch.sort(last_step_logits, descending=True, dim=-1)
    ranks = (sorted_indices == labels.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1

    # --- 计算指标 ---
    metrics = {}
    for k in [1, 5, 10, 20, 50]:
        hr_k = (ranks <= k).float().mean().item()
        metrics[f"HR@{k}"] = round(hr_k, 4)
        
        in_top_k = (ranks <= k)
        ndcg_k = torch.where(
            in_top_k, 1.0 / torch.log2(ranks.float() + 1), torch.tensor(0.0)
        ).mean().item()
        metrics[f"NDCG@{k}"] = round(ndcg_k, 4)

    mrr = (1.0 / ranks.float()).mean().item()
    metrics["MRR"] = round(mrr, 4)
    return metrics

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

# --- 7. Main 函数 ---
def main():
    # 参数定义
    dataset_path = "/root/20250613Rec-Factory/data/amazon_SPIAO_from_qspan/llama_pt_format.json"
    output_dir = "/root/20250613Rec-Factory/try_train/llama-rec-checkpoints"
    tokenizer_dir = "/root/20250613Rec-Factory/try_train/hybrid_item_tokenizer_SPIAO"
    max_seq_length = 128
    
    # 创建/加载 Tokenizer (使用全量数据)
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    if not os.path.exists(tokenizer_file):
        print("Tokenizer not found. Creating a new one from the full dataset...")
        def text_to_int_list(example):
            example['item_sequence'] = [int(i.strip()) for i in example['text'].split(',') if i.strip()]
            return example
        temp_dataset_with_list = raw_dataset.map(text_to_int_list, remove_columns=["text"])
        mock_args = MockTrainingArguments(output_dir=tokenizer_dir, max_length=max_seq_length)
        tokenizer = create_hybrid_item_tokenizer(dataset=temp_dataset_with_list, training_args=mock_args)
    else:
        print("Found existing tokenizer. Loading it...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    # 健壮性检查
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if tokenizer.bos_token_id is None: tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    assert tokenizer.pad_token_id is not None and tokenizer.bos_token_id is not None
    print(f"Final check - pad_token_id: {tokenizer.pad_token_id}, bos_token_id: {tokenizer.bos_token_id}")

    # --- 5. 预处理数据集并划分 ---
    print("Preprocessing and splitting the dataset...")
    processed_dataset = raw_dataset.map(
        final_preprocess_function, # 确保是这个函数
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=4,
    )
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Train dataset size: {len(train_dataset)}, Evaluation dataset size: {len(eval_dataset)}")

    # 创建模型
    print("Creating LlamaForRec model from scratch...")
    config = LlamaRecConfig(
        model_type=MODEL_TYPE, vocab_size=len(tokenizer), hidden_size=256,
        intermediate_size=512, num_hidden_layers=4, num_attention_heads=4,
        max_position_embeddings=max_seq_length, rms_norm_eps=1e-6, use_cache=False,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForRec(config)
    print(f"Model created with {model.num_parameters() / 1e6:.2f} M parameters.")

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=output_dir, per_device_train_batch_size=96, per_device_eval_batch_size=256,
        eval_accumulation_steps=10, gradient_accumulation_steps=1, learning_rate=5e-4,
        num_train_epochs=100, lr_scheduler_type="cosine", warmup_ratio=0,
        logging_dir=f"{output_dir}/logs", logging_steps=100, save_strategy="epoch",
        eval_strategy="epoch", save_total_limit=20, fp16=True, report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    # 定义数据整理器
    train_collator = TrainDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)

    # 实例化 Trainer
    trainer = CustomTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=eval_dataset, tokenizer=tokenizer, data_collator=train_collator,
        compute_metrics=compute_metrics, eval_collator=eval_collator,
    )

    # 开始训练
    print("Starting training with proper Leave-One-Out evaluation...")
    trainer.train()

    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final")
    print(f"Training complete. Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("All operations complete!")

if __name__ == "__main__":
    main()