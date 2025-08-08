import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
from typing import Dict, List, Optional
import warnings

# --- 1. 复用你在 train.py 中定义好的核心组件 ---
#    (为了让脚本独立可运行，我们直接将它们复制过来)

# 导入 Transformers 和你的自定义模型代码
from transformers import PreTrainedTokenizerFast
from transformers.models.llama_rec.modeling_llamarec import LlamaRecForCausalLM
from transformers import EvalPrediction

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- [复用] 预处理函数 ---
def final_preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
    all_sequences = []
    for text in examples["text"]:
        item_ids = [int(i.strip()) for i in text.split(',') if i.strip()]
        if len(item_ids) > 1:
            all_sequences.append(item_ids)
    return {"sequence": all_sequences}

# --- [复用] 评估数据整理器 ---
class EvalDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_sequences_as_int = [e["sequence"][:-1] for e in examples]
        eval_labels_as_int = [e["sequence"][-1] for e in examples]
        input_sequences_as_str = [[str(item_id) for item_id in seq] for seq in input_sequences_as_int]
        
        batch = self.tokenizer(
            input_sequences_as_str,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        eval_labels_as_token_ids = self.tokenizer.convert_tokens_to_ids(
            [str(item_id) for item_id in eval_labels_as_int]
        )
        
        # 这个标签设计与你训练时使用的评估逻辑是匹配的
        batch['labels'] = torch.tensor(eval_labels_as_token_ids)
        return batch

# --- [复用] 流式指标计算器 ---
class StreamingMetricsCalculator:
    def __init__(self, k_values: List[int] = [1, 5, 10, 20, 50]):
        self.k_values = k_values
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_ranks: List[torch.Tensor] = []

    def accumulate(self, eval_preds: EvalPrediction):
        """仅累积当前批次的排名"""
        logits, labels_matrix = eval_preds.predictions, eval_preds.label_ids
        
        last_step_logits = logits[:, -1, :]
        labels = labels_matrix.view(-1)

        valid_mask = labels != -100
        labels = labels[valid_mask]
        last_step_logits = last_step_logits[valid_mask]

        if labels.numel() > 0:
            sorted_indices = torch.argsort(last_step_logits, descending=True, dim=-1)
            ranks = (sorted_indices == labels.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1
            self.all_ranks.append(ranks.cpu())

    def compute(self) -> Dict[str, float]:
        """计算并返回最终指标，然后重置状态"""
        if not self.all_ranks:
            return {"message": "No valid labels found during evaluation."}

        final_ranks = torch.cat(self.all_ranks).float()
        metrics = {}
        for k in self.k_values:
            in_top_k = final_ranks <= k
            hr_k = in_top_k.float().mean().item()
            metrics[f"HR@{k}"] = round(hr_k, 4)
            ndcg_k = (1.0 / torch.log2(final_ranks + 1.0)).where(in_top_k, 0.0).mean().item()
            metrics[f"NDCG@{k}"] = round(ndcg_k, 4)

        metrics["MRR"] = round((1.0 / final_ranks).mean().item(), 4)
        
        # 重置状态
        self.all_ranks = []
        return metrics

# --- 2. 评估主函数 ---
def evaluate_checkpoint():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="Evaluate a trained LlamaRec checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved model checkpoint directory.")
    parser.add_argument("--config", type=str, required=True, help="Name of the config file used during training (e.g., 'pantry').")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size for evaluation.")
    args = parser.parse_args()

    print("--- Evaluation Setup ---")
    print(f"Checkpoint Path: {args.checkpoint_path}")
    print(f"Config Name: {args.config}")
    
    # --- 加载与训练时相同的配置文件 ---
    default_config_path = "/home/kfwang/20250613Rec-Factory/try_train/train_config/" # 与你的训练脚本保持一致
    with open(os.path.join(default_config_path, args.config + '.yaml'), 'r') as f:
        config_data = yaml.safe_load(f)

    paths_config = config_data['paths']
    model_params = config_data['model_params']
    dataset_split_config = config_data['dataset_split']

    # --- 设备配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 加载模型和 Tokenizer ---
    print("\n--- Loading Model & Tokenizer ---")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.checkpoint_path)
    model = LlamaRecForCausalLM.from_pretrained(args.checkpoint_path)
    model.to(device)
    model.eval() # **非常重要**：切换到评估模式
    print(f"Model loaded with {model.num_parameters() / 1e6:.2f} M parameters.")

    # --- 加载和准备数据集 ---
    print("\n--- Loading and Preparing Dataset ---")
    raw_dataset = load_dataset("json", data_files=paths_config['dataset_path'], split="train")
    processed_dataset = raw_dataset.map(
        final_preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=4,
    )
    
    # **非常重要**：使用与训练时完全相同的种子和比例来切分数据集，确保得到的是同一个测试集
    split_dataset = processed_dataset.train_test_split(
        test_size=dataset_split_config['test_size'], 
        seed=dataset_split_config['seed']
    )
    eval_dataset = split_dataset["test"]
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    # --- 创建数据整理器和数据加载器 ---
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=model_params['max_seq_length'])
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=eval_collator,
        num_workers=4,
        pin_memory=True
    )

    # --- 初始化指标计算器 ---
    metrics_calculator = StreamingMetricsCalculator()
    
    # --- 手动评估循环 ---
    print("\n--- Starting Evaluation Loop ---")
    with torch.no_grad(): # **非常重要**：禁用梯度计算，节省显存和计算资源
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # 1. 将数据移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 2. 获取模型输出
            outputs = model(**batch)
            
            # 3. 准备评测函数需要的输入
            logits = outputs.logits
            labels = batch['labels']
            eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
            
            # 4. 累积指标
            metrics_calculator.accumulate(eval_pred)

    # --- 计算并打印最终结果 ---
    final_metrics = metrics_calculator.compute()
    
    print("\n--- Evaluation Results ---")
    # 使用漂亮的格式打印结果
    for metric, value in final_metrics.items():
        print(f"{metric:<10}: {value}")
    print("--------------------------")


if __name__ == "__main__":
    evaluate_checkpoint()