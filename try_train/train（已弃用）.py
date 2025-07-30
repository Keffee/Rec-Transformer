import os
import logging
from typing import Dict, List, Union, Optional
from datasets import load_dataset, Dataset
from functools import partial
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

# 你的 create_hybrid_item_tokenizer 脚本
from transformers.models.aaa_llama4rec.tokenization_llamarec_try import create_hybrid_item_tokenizer, MockTrainingArguments 

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
# 你的自定义模型
from transformers.models.aaa_llama4rec.modeling_llamarec_try import LlamaForRec, LlamaRecConfig
from transformers.tokenization_utils_base import BatchEncoding

# --- [最终修改] 自定义 Data Collator ---
# 这个版本更简单、更直接，也更符合 fast tokenizer 的设计
class FinalDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        # 1. 从输入的 dict list 中提取出 'text' 字段的 list
        # examples = [{'text': '1,2,3'}, {'text': '4,5,6,7'}]
        # texts = ['1,2,3', '4,5,6,7']
        texts = [e["text"] for e in examples]

        # 2. 使用 tokenizer 的 __call__ 方法一步到位处理所有事情
        batch = self.tokenizer(
            texts,
            padding=True,          # 动态填充到批次内的最大长度
            truncation=True,       # 截断到 max_length
            max_length=self.max_length,
            return_tensors="pt",   # 直接返回 PyTorch 张量
        )

        # 3. 创建 labels，并将 padding 部分设置为 -100
        # 这是 Causal LM 训练的标准做法
        batch["labels"] = batch["input_ids"].clone()
        # 获取 attention_mask，值为 0 的地方是 padding
        # 我们将这些地方的 labels 设置为 -100
        mask = batch["attention_mask"].bool()
        batch["labels"][~mask] = -100
        
        return batch

# # --- [最终修改] 预处理函数 ---
# # 这个函数现在变得极其简单，只负责确保数据格式正确
# def minimal_preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
#     # 我们什么都不做，只返回原始的 'text' 字段
#     # 因为所有复杂的处理都移到了 Data Collator 中
#     return {"text": examples["text"]}

# --- 1. 注册模型 (保持不变) ---
MODEL_TYPE = "llama-rec"
AutoConfig.register(MODEL_TYPE, LlamaRecConfig)
AutoModelForCausalLM.register(LlamaRecConfig, LlamaForRec)

# --- 2. 预处理函数 (稍微调整以适应 hybrid_tokenizer 的输出) ---

# --- [最终修改] 预处理函数 ---
# 这个函数现在只需要把文本分割成整数列表
def final_preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
    all_sequences = []
    for text in examples["text"]:
        # 将 "1,2,3" 转换为 [1, 2, 3]
        item_ids = [int(i.strip()) for i in text.split(',') if i.strip()]
        all_sequences.append(item_ids)
    return {"sequence": all_sequences} # 返回一个新的 'sequence' 列


# --- [新增] 智能的评估数据整理器 ---
class EvalDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # 1. 分离原始 ID
        input_sequences_as_int = [e["sequence"][:-1] for e in examples]
        eval_labels_as_int = [e["sequence"][-1] for e in examples]
        
        # 2. 编码输入序列 (tokenizer 自动处理 left-padding)
        input_sequences_as_str = [[str(i) for i in seq] for seq in input_sequences_as_int]
        batch = self.tokenizer(
            input_sequences_as_str, is_split_into_words=True, padding=True,
            truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        
        # 3. 编码标签和历史记录，但不 padding，作为普通的 list 传递
        batch['eval_labels_as_list'] = self.tokenizer.convert_tokens_to_ids(
            [str(i) for i in eval_labels_as_int]
        )
        batch['history_ids_as_list'] = [
            self.tokenizer.convert_tokens_to_ids([str(i) for i in seq])
            for seq in input_sequences_as_int
        ]
        
        return batch

# --- [新增] 训练数据整理器 ---
# 训练时，我们仍然使用 next-item prediction
class TrainDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.collator = DataCollatorForLanguageModeling(tokenizer, mlm=False) # 这行不再需要

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # 训练时，我们使用完整的序列进行 next-token prediction
        sequences_as_int = [e["sequence"] for e in examples]
        
        # --- [核心修改] 将整数ID列表转换为字符串ID列表 ---
        # [[1, 2, 3]] -> [['1', '2', '3']]
        sequences_as_str = [[str(item_id) for item_id in seq] for seq in sequences_as_int]

        # 手动转换为 tokenizer 期望的格式
        # 现在 sequences_as_str 的类型是 List[List[str]]，符合要求
        batch_dict = self.tokenizer(
            sequences_as_str,
            is_split_into_words=True, # 告诉 tokenizer 输入已经是 "词" (字符串形式的ID) 的列表
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 后续逻辑保持不变
        batch_dict['labels'] = batch_dict['input_ids'].clone()
        mask = batch_dict["attention_mask"].bool()
        batch_dict["labels"][~mask] = -100
        
        return batch_dict


# # [修改] 这个函数现在只做分词和截断，不处理 padding
# def preprocess_function(examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizerFast, max_length: int) -> Dict[str, List[int]]:
#     """
#     使用 hybrid tokenizer 对文本进行编码。
#     只进行分词和截断，Padding 的工作完全交给 DataCollator。
#     """
#     # 核心修改：移除 padding 参数，让 tokenizer 返回变长的 token ID 列表
#     # tokenizer 已经配置为自动添加 BOS/EOS
#     all_input_ids = tokenizer(
#         examples["text"], 
#         truncation=True, 
#         max_length=max_length,
#         # 移除 padding='max_length'
#     )["input_ids"]

#     # 注意：这里我们只返回一个字典，包含 input_ids 和 labels
#     # DataCollatorForLanguageModeling 会自动处理 attention_mask
#     return {
#         "input_ids": all_input_ids,
#         "labels": [ids.copy() for ids in all_input_ids],
#     }

def compute_metrics(eval_preds: "EvalPrediction"):
    """
    计算推荐指标。
    """
    filtered_logits, labels = eval_preds
    
    logits = torch.from_numpy(filtered_logits)
    labels = torch.from_numpy(labels).view(-1)
    
    # --- 计算排名 (现在非常简单) ---
    true_logits = logits.gather(1, labels.unsqueeze(-1)).squeeze(-1)
    ranks = (logits > true_logits.unsqueeze(-1)).sum(dim=1) + 1

    # --- 计算指标 ---
    metrics = {}
    for k in [1, 5, 10, 20, 50]:
        hr_k = (ranks <= k).float().mean().item()
        metrics[f"HR@{k}"] = round(hr_k * 100, 4)
        
        in_top_k = (ranks <= k)
        ndcg_k = torch.where(
            in_top_k, 1.0 / torch.log2(ranks.float() + 1), torch.tensor(0.0)
        ).mean().item()
        metrics[f"NDCG@{k}"] = round(ndcg_k, 4)

    mrr = (1.0 / ranks.float()).mean().item()
    metrics["MRR"] = round(mrr, 4)

    return metrics

# --- [新增] 自定义 Trainer 以便使用不同的 Data Collator ---
# 这是比之前 hack 更优雅、更健壮的方法
class CustomTrainer(Trainer):
    # [新增] 重写 __init__ 方法
    def __init__(self, *args, eval_collator, **kwargs):
        # 我们不再需要 max_length，直接传入 collator 实例
        super().__init__(*args, **kwargs)
        # 将我们自定义的 eval_collator 保存起来
        self.custom_eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> torch.utils.data.DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.custom_eval_collator, # <-- 使用我们保存的 collator
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def main():
    # --- 3. 定义路径和参数 ---
    dataset_path = "/home/kfwang/20250613Rec-Factory/data/amazon_pantry_from_qspan/llama_pt_format.json"
    output_dir = "/home/kfwang/20250613Rec-Factory/try_train/llama-rec-checkpoints"
    tokenizer_dir = "/home/kfwang/20250613Rec-Factory/try_train/hybrid_item_tokenizer" # 你的 tokenizer 保存路径
    max_seq_length = 128
    
    # 4.1. 先加载原始数据集
    print("Loading raw dataset to build tokenizer...")
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 4.2. 创建或加载 Tokenizer (使用全量数据)
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    
    if not os.path.exists(tokenizer_file):
        print(f"Tokenizer not found. Creating a new one from the full dataset...")
        
        def text_to_int_list(example):
            example['item_sequence'] = [int(i.strip()) for i in example['text'].split(',') if i.strip()]
            return example
            
        # 使用全量的 raw_dataset 来构建词汇表
        temp_dataset_with_list = raw_dataset.map(text_to_int_list)
        
        mock_args = MockTrainingArguments(output_dir=tokenizer_dir, max_length=max_seq_length)
        tokenizer = create_hybrid_item_tokenizer(dataset=temp_dataset_with_list, training_args=mock_args)
        print("Tokenizer created and saved.")
    else:
        print(f"Found existing tokenizer. Loading it...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        print("Tokenizer loaded.")

    # --- [新增] 健壮性检查和修复 ---
    # 确保特殊 token ID 被正确加载，避免 'NoneType' 错误
    if tokenizer.pad_token_id is None:
        print("Warning: tokenizer.pad_token_id is None. Setting it manually.")
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    
    if tokenizer.bos_token_id is None:
        print("Warning: tokenizer.bos_token_id is None. Setting it manually.")
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")

    print(f"Final check - pad_token_id: {tokenizer.pad_token_id}, bos_token_id: {tokenizer.bos_token_id}")
    # 确保它们现在不是 None
    assert tokenizer.pad_token_id is not None
    assert tokenizer.bos_token_id is not None

    # --- 5. 预处理数据集并划分 ---
    print("Preprocessing and splitting the dataset...")
    # 注意：现在 raw_dataset 已经被加载过了，直接使用即可
    processed_dataset = raw_dataset.map(
        final_preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=4,
    )
    
    # 现在可以安全地划分了，因为 tokenizer 知道所有的 item ID
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    # --- 6. 从头创建模型 (使用新 tokenizer 的词汇表大小) ---
    print("Creating LlamaForRec model from scratch...")
    config = LlamaRecConfig(
        model_type=MODEL_TYPE,
        vocab_size=len(tokenizer), # 关键：使用新 tokenizer 的大小
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=max_seq_length,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForRec(config)
    print(f"Model created with {model.num_parameters() / 1e6:.2f} M parameters.")

    # --- 7. 定义训练参数 ---
    # [修改] 确保评估参数已设置
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64, # 评估时的批次大小
        eval_accumulation_steps=40,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        num_train_epochs=20,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch", # 在每个 epoch 结束时进行评估
        save_total_limit=3,
        fp16=True,
        report_to="tensorboard",
        remove_unused_columns=False, # 确保为 False
        # label_names=["labels"],      # 告诉 Trainer 评估时的标签字段名
        # include_inputs_for_metrics=True, 
    )
    
    # --- 8. 定义数据整理器 ---
    # [修改] 为训练和评估使用不同的 Data Collator
    train_collator = TrainDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)

    # --- 9. 实例化 Trainer ---

    # --- [修改] 在实例化 Trainer 之前，准备好 compute_metrics ---
    # 使用 functools.partial 来将 tokenizer 对象 "绑定" 到 compute_metrics 函数上
    compute_metrics_with_tokenizer = partial(compute_metrics, tokenizer=tokenizer)

    # [修改] 使用 CustomTrainer，并将整个数据集同时作为训练集和评估集
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=train_collator, # 训练时使用
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        eval_collator=eval_collator, # <-- 将 eval_collator 传入
    )

    # --- 10. 开始训练！---
    print("Starting training with Leave-One-Out evaluation...")
    trainer.train()

    # --- 11. 保存最终模型和 tokenizer (保持不变) ---
    final_model_path = os.path.join(output_dir, "final")
    print(f"Training complete. Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    # 确保 tokenizer 也被保存到最终模型目录中，以便于分发
    tokenizer.save_pretrained(final_model_path)
    print("All operations complete!")

if __name__ == "__main__":
    main()