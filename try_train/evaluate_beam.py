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
        # 1. 分离输入和标签 (原始 item ID)
        #input_sequences_as_int = [e["sequence"][:-1] for e in examples]
        #eval_labels_as_int = [e["sequence"][-1] for e in examples]

        all_inputs = [e["text"].split(" ") for e in examples]
        all_labels = all_inputs
        
        all_inputs, all_labels = [], []
        self.sid_len = 3 # the number of semantic token ids for one item id
        self.tgt_pad_len = 6223*self.sid_len
        # padded_sid_seq = ['<a_194>', '<b_63>', '<c_39>']
        for e in examples:
            tokens = e["text"].split(" ") # ['<a_13>', '<b_76>', '<c_117>', '<a_95>', '<b_66>', '<c_182>', '<a_194>', '<b_63>', '<c_39>'...]
            hist_tokens = tokens[:-self.tgt_pad_len]
            tgt_tokens = tokens[-self.tgt_pad_len:] 
            
            all_inputs.append(hist_tokens)
            all_labels.append(tgt_tokens) 
        # 2. 将输入序列的原始 ID 转换为字符串
        #input_sequences_as_str = [[str(item_id) for item_id in seq] for seq in input_sequences_as_int]
        # 3. 使用 tokenizer 对输入序列进行编码、截断和填充
        batch = self.tokenizer(
            all_inputs,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        batch_labels = self.tokenizer(
            all_labels,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.tgt_pad_len,
            return_tensors="pt"
        )        
        # 4. 将评估标签的原始 ID 转换为 Token ID
        #label_ids = self.tokenizer.convert_tokens_to_ids(all_labels)
        batch["labels"] =  batch_labels['input_ids']
        if self.tokenizer.pad_token_id is not None:
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        
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
    # <<< 新增: 解析命令行参数以获取配置文件路径 >>>
    parser = argparse.ArgumentParser(description="Train a LlamaRec model using a YAML config file.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved model checkpoint directory.")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size for evaluation.")
    
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


    # --- 设备配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 加载模型和 Tokenizer ---
    print("\n--- Loading Model & Tokenizer ---")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cli_args.checkpoint_path)
    model = LlamaRecForCausalLM.from_pretrained(cli_args.checkpoint_path)
    model.to(device)
    model.eval() # **非常重要**：切换到评估模式
    print(f"Model loaded with {model.num_parameters() / 1e6:.2f} M parameters.")

    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=model_params['max_seq_length'])
    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=cli_args.eval_batch_size,
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
            batch.pop("token_type_ids", None)
            # 2. 获取模型输出
            len_sid = 3
            num_beam = 5
            num_return_sequences=5
            outputs_beam = model.generate(
                **batch,
                max_length=len_sid + int(model_params['max_seq_length']),
                num_beams=num_beam,
                num_return_sequences=num_return_sequences,   # return all beams
                early_stopping=True
            )
            # 3. 准备评测函数需要的输入
            generate_only = outputs_beam[:,-len_sid:]
            outputs = generate_only.view(cli_args.eval_batch_size, num_return_sequences, len_sid) # [B,nbeam,3]
            #logits = outputs.logits
            labels = batch['labels']
            labels = labels.view(cli_args.eval_batch_size,  int(labels.size(1)/3), len_sid) # [B,seq,3]

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