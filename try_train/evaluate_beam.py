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

from transformers import PreTrainedTokenizerFast
from transformers.models.llama_rec.modeling_llamarec import LlamaRecForCausalLM
from transformers import EvalPrediction

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

def final_preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
    all_sequences = []
    for text in examples["text"]:
        item_ids = [int(i.strip()) for i in text.split(',') if i.strip()]
        if len(item_ids) > 1:
            all_sequences.append(item_ids)
    return {"sequence": all_sequences}

class EvalDataCollator: 
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:

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

def evaluate_checkpoint():
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

    print(f"Loading configuration from: {args.config}")
    current_dir_name = os.path.dirname(os.path.abspath("__file__"))
    config_path = os.path.join(current_dir_name, "pretrain_config", args.config+'.yaml')
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    paths_config = config_data['paths']
    model_params = config_data['model_params']
    training_args_dict = config_data['training_args']

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

    print("\n--- Loading Model & Tokenizer ---")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.checkpoint_path)
    model = LlamaRecForCausalLM.from_pretrained(args.checkpoint_path)
    model.to(device)
    model.eval() 
    print(f"Model loaded with {model.num_parameters() / 1e6:.2f} M parameters.")
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=model_params['max_seq_length'])
    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=eval_collator,
        num_workers=4,
        pin_memory=True
    )    

    metrics_sum = defaultdict(lambda: defaultdict(float))
    all_results = []  

    print("\n--- Starting Evaluation Loop ---")
    with torch.no_grad(): 
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

        metrics_avg = {}
        for metric_name, ks_dict in metrics_sum.items():
            metrics_avg[metric_name] = {k: v / test_num for k, v in ks_dict.items()}

        formatted_metrics = {
            metric_name: {k: f"{v:.4f}" for k, v in ks_dict.items()}
            for metric_name, ks_dict in metrics_avg.items()
        }

        print("Average metrics:", formatted_metrics)
        output_dir_eval = "output_eval"
        os.makedirs(output_dir_eval, exist_ok=True)
        output_file = os.path.join(output_dir_eval, f"{run_name}_metrics.json")
        output_file = f"{run_name}_metrics.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_metrics, f, indent=2, ensure_ascii=False)

        print(f"Saved metrics to {output_file}")

if __name__ == "__main__":
    evaluate_checkpoint()