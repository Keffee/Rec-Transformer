import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
from typing import Dict, List, Optional,Iterable
import warnings
import json
from utils_metric import eval_from_beams
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
import time
# --- 1. 复用你在 train.py 中定义好的核心组件 ---
#    (为了让脚本独立可运行，我们直接将它们复制过来)

# 导入 Transformers 和你的自定义模型代码
from transformers import PreTrainedTokenizerFast
from transformers.models.llama_rec.modeling_llamarec import LlamaRecForCausalLM
from transformers import EvalPrediction

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

import torch.distributed as dist

def setup_device():
    if dist.is_available() and dist.is_nccl_available() and "RANK" in os.environ:
        # Multi-GPU distributed mode (torchrun/torch.distributed.launch)
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(f"[Rank {dist.get_rank()}] Using device: {device}")
    elif torch.cuda.is_available():
        # Single-GPU
        device = torch.device("cuda", 0)
        print(f"[Single GPU] Using device: {device}")
    else:
        # CPU fallback
        device = torch.device("cpu")
        print("[CPU] Using device: cpu")
    return device

device = setup_device()

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
        all_labels = [e["ground_truth"].split(" ") for e in examples]
        
        batch = self.tokenizer(
            all_inputs,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return batch, all_labels

'''
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
        self.tgt_pad_len = 5893*self.sid_len
        # padded_sid_seq = ['<a_194>', '<b_63>', '<c_39>']
        for idx, e in enumerate(examples):
            tokens = e["text"].split(" ") # ['<a_13>', '<b_76>', '<c_117>', '<a_95>', '<b_66>', '<c_182>', '<a_194>', '<b_63>', '<c_39>'...]
            hist_tokens = tokens[:-self.tgt_pad_len]
            tgt_tokens = tokens[-self.tgt_pad_len:] 
            
            if idx==0:
                print('hist_tokens firt 10: ', hist_tokens[:10])
                print('hist_tokens - 10: ', hist_tokens[-10:])

                print('tgt_tokens :10: ', tgt_tokens[:10] )
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
'''       

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

def map_triplets_outputs(outputs: torch.Tensor, tokenizer, tokens2iid: dict) -> torch.Tensor:
    """
    Map model outputs [B, nbeam, 3] to [B, nbeam] new ids.
    No ignore_index applied here.
    """
    B, nbeam, T = outputs.shape
    flat_ids = outputs.detach().cpu().reshape(-1).tolist()
    flat_toks = tokenizer.convert_ids_to_tokens(flat_ids)  # list[str], length = B*nbeam*3
    triplets = [tuple(flat_toks[i:i+T]) for i in range(0, len(flat_toks), T)]

    out = torch.full((B, nbeam), -1, dtype=torch.long)
    idx = 0
    for b in range(B):
        for j in range(nbeam):
            trip = triplets[idx]; idx += 1
            out[b, j] = tokens2iid.get(trip, -1)  # -1 if not found
    return out


def map_triplets_labels(labels: torch.Tensor, tokenizer, tokens2iid: dict, ignore_index: int = -100) -> torch.Tensor:
    """
    Map labels [B, seqlen, 3] to [B, seqlen] new ids.
    Any triplet containing ignore_index is skipped -> -1.
    """
    B, L, T = labels.shape
    flat_ids = labels.detach().cpu().view(-1).tolist()
    triplets = [tuple(flat_ids[i:i+T]) for i in range(0, len(flat_ids), T)]

    out = torch.full((B, L), -1, dtype=torch.long)
    idx = 0
    for b in range(B):
        for j in range(L):
            trip = triplets[idx]; idx += 1
            # if this label row contains ignore_index anywhere → drop
            if (labels[b, j] == ignore_index).any():
                continue
            # id to tokens 
            toks = tuple(tokenizer.convert_ids_to_tokens(trip))
            out[b, j] = tokens2iid.get(toks, -1)
    return out

# --- 2. 评估主函数 ---
def evaluate_checkpoint():
    # <<< 新增: 解析命令行参数以获取配置文件路径 >>>
    parser = argparse.ArgumentParser(description="Train a LlamaRec model using a YAML config file.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved model checkpoint directory.")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size for evaluation.")
    parser.add_argument('--rq_map_path', type=str, default='4_item_id_to_rq_code.json',
                        help="Path to the JSON file mapping original item IDs to RQ codes (e.g., '4_item_id_to_rq_code.json').")
                        
    parser.add_argument("--config", type=str, required=True, help="Name of the config file to use.")

    parser.add_argument("--lensid", type=int, default=3, help="the length of semantic tokens")
    parser.add_argument("--decode-strategy", type=str, choices=["beam", "topk-topp"], default="beam", help="Decoding strategy: 'beam' for beam search or 'topk-topp' for nucleus/top-k sampling")
    parser.add_argument("--num-beams", type=int, default=5, help="Number of beams for beam search. Also used as top-k cutoff if --decode-strategy=topk-topp")
    parser.add_argument("--num-return-sequences", type=int, default=1, help="Number of sequences to return for each input")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling probability, used if --decode-strategy=topk-topp")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff, used if --decode-strategy=topk-topp")

    parser.add_argument("--eval-ratio-last", type=float, default=1.0, help="Fraction of evaluation set to use (0 < ratio ≤ 1.0), the last part of the whole dataset")

    parser.add_argument(
        "--metrics", type=str, nargs="+",
        default=["HR", "NDCG", "MRR", "MAP"],
        help="List of metrics to compute, e.g. --metrics HR NDCG Recall"
    )
    # ["HR", "Recall", "Precision", "NDCG", "MRR", "MAP"],
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100, 200],
        help="List of cutoff values for evaluation metrics, e.g. --ks 20 50 100")

    args = parser.parse_args()
    print(args.ks)
    if args.decode_strategy == "beam":
        assert args.num_return_sequences <= args.num_beams, \
            f"num_return_sequences ({args.num_return_sequences}) must be <= num_beams ({args.num_beams}) when using beam search."
        run_name = f"{args.config}_beam{args.num_beams}_ret{args.num_return_sequences}"
    elif args.decode_strategy == "topk-topp":
        run_name = (
            f"{args.config}_topk{args.top_k}_topp{args.top_p}_ret{args.num_return_sequences}_evalratio{args.eval_ratio_last}"
        )

    with open(args.rq_map_path, "r") as f:
        iid2tokens = json.load(f)

    tokens2iid = {
        tuple(map(str, toks)): int(vid)  # assumes all vids are int
        for vid, toks in iid2tokens.items()
    }

    # <<< 新增: 读取并解析 YAML 配置文件 >>>
    print(f"Loading configuration from: {args.config}")
    current_dir_name = os.path.dirname(os.path.abspath("__file__"))
    config_path = os.path.join(current_dir_name, "pretrain_config", args.config+'.yaml')
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

    test_dataset_full = load_dataset("json", data_files=dataset_path, split="test")
    total = len(test_dataset_full)
    test_num = int(total * args.eval_ratio_last)
    test_dataset = test_dataset_full.select(range(len(test_dataset_full) - test_num, len(test_dataset_full)))
    print(f"Test on ratio_last {args.eval_ratio_last:.2f}, {test_num}/{total} samples ({test_num/total:.2%})")

    # --- 加载模型和 Tokenizer ---
    print("\n--- Loading Model & Tokenizer ---")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.checkpoint_path)
    model = LlamaRecForCausalLM.from_pretrained(args.checkpoint_path)
    model.to(device)
    model.eval() # **非常重要**：切换到评估模式
    print(f"Model loaded with {model.num_parameters() / 1e6:.2f} M parameters.")
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=model_params['max_seq_length'])
    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=eval_collator,
        num_workers=4,
        pin_memory=True
    )    

    total_batches = len(eval_dataloader)
    metrics_sum = defaultdict(lambda: defaultdict(float))
    num_batches = 0
    all_results = []  

    # --- 手动评估循环 ---
    print("\n--- Starting Evaluation Loop ---")
    
    #with open("eval_results.jsonl", "w", encoding="utf-8") as f:
    with torch.no_grad(): # **非常重要**：禁用梯度计算，节省显存和计算资源
        for batch, labels in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("token_type_ids", None)

            start_time = time.time() 
            if args.decode_strategy == "beam":
                outputs_beam = model.generate(
                    **batch,
                    max_length=args.lensid + int(model_params['max_seq_length']),
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_return_sequences,
                    early_stopping=True
                )

            elif args.decode_strategy == "topk-topp":
                outputs_beam = model.generate(
                    **batch,
                    max_length=args.lensid + int(model_params['max_seq_length']),
                    do_sample=True,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_return_sequences=args.num_return_sequences,
                    early_stopping=True
                )
            end_time = time.time() 

            elapsed_minutes = (end_time - start_time) / 60
            print(f"beam Time spent: {elapsed_minutes:.2f} minutes")
            generate_only = outputs_beam[:,-args.lensid:]
            outputs = generate_only.view(-1, args.num_return_sequences, args.lensid) # [B,nbeam,3]
            lab_ids = labels
            out_ids = map_triplets_outputs(outputs, tokenizer, tokens2iid=tokens2iid)
            # convert to list of tensors (int)
            lab_ids_int = [torch.tensor([int(x) for x in row], dtype=torch.long) for row in lab_ids]
            # pad them to the same length
            lab_tensor = pad_sequence(lab_ids_int, batch_first=True, padding_value=-100)
            eval_pred = eval_from_beams(out_ids, lab_tensor, ignore_index=-100, ks=args.ks, metrics=args.metrics)            
            
            for metric_name, ks_dict in eval_pred.items():
                for k, value in ks_dict.items():
                    metrics_sum[metric_name][k] += value
            '''
            #for i in range(len(out_ids)):
            for i in range(1):
                result = {
                    "input": tokenizer.decode(batch["input_ids"][i].tolist(), skip_special_tokens=True),
                    "output": out_ids[i].tolist() if torch.is_tensor(out_ids[i]) else out_ids[i],
                    "groundtruth": lab_ids[i]
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            '''
        metrics_avg = {}
        for metric_name, ks_dict in metrics_sum.items():
            metrics_avg[metric_name] = {k: v / num_batches for k, v in ks_dict.items()}

        formatted_metrics = {
            metric_name: {k: f"{v:.4f}" for k, v in ks_dict.items()}
            for metric_name, ks_dict in metrics_avg.items()
        }

        print("Average metrics:", formatted_metrics)

        output_file = f"{run_name}_metrics.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_metrics, f, indent=2, ensure_ascii=False)

        print(f"Saved metrics to {output_file}")

if __name__ == "__main__":
    evaluate_checkpoint()