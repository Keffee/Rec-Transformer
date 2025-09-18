import argparse
import random
import torch
import numpy as np
import json
import logging
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 确保这些模块可以被正确导入 ---
from models.rqvae import RQVAE
from trainer import Trainer # 假设您的 Trainer 类在这里
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
#
# --- 第1部分: 脚本设置和辅助函数 ---
#

def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-VAE and Generate Codes from Embeddings")

    parser.add_argument('--dataset', required=True,choices=["KuaiRand-27K", "KuaiRand-27K-0501","KuaiRand-27K-100krows"])
    
    # --- 输入/输出路径参数 ---
    parser.add_argument("--sasrec_emb_path", type=str,
                        help="Path to the item embeddings .npy file. Assumes index corresponds to original item ID.")
    parser.add_argument("--ckpt_dir", type=str, default="./rqvae_checkpoints",
                        help="Directory to save RQ-VAE model checkpoints.")
    parser.add_argument("--output_dir", type=str, default="./rqvae_output",
                        help="Directory to save the final code and embedding mappings.")

    # --- 训练超参数 ---
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--weight_decay", type=float, default=0.0, help='l2 regularization weight')
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument('--patience', default=5, type=int, help='Number of epochs to wait for improvement before early stopping.')


    # --- RQ-VAE 模型特定参数 ---
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128], help='hidden sizes of encoder/decoder layers')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", action='store_true', help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", action='store_true', help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")


    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleEmbeddingDataset(Dataset):
    """一个简单的数据集，用于包装嵌入向量的NumPy数组"""
    def __init__(self, embeddings_array):
        # 将数据转换为 torch.FloatTensor
        self.embeddings = torch.FloatTensor(embeddings_array)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

#
# --- 第2部分: 主执行逻辑 ---
#
if __name__ == '__main__':
    args = parse_args()
    set_seed(2024)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("====================== ARGS ======================")
    print(json.dumps(vars(args), indent=2))
    print("==================================================")

    # 准备输出目录
    #os.makedirs(args.output_dir, exist_ok=True)
    #os.makedirs(args.ckpt_dir, exist_ok=True)

    # Define finetuned parameters and show them in output folder name. 
    # If already select the best parameter, just remove it.
    num_emb_str = "-".join(map(str, args.num_emb_list))

    run_parts = [
        f"bs{args.batch_size}",
        f"d{args.e_dim}",
        f"lr{str(args.lr)}",
        f"emb{num_emb_str}",
    ]

    run_name = "_".join(run_parts)
    run_dir = os.path.join(f"output_{args.dataset}", run_name)
    os.makedirs(run_dir, exist_ok=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    # tensorboard folder
    tblog_dir = os.path.join(run_dir, 'tblog')
    if not os.path.exists(tblog_dir):
        os.makedirs(tblog_dir)    
    writer = SummaryWriter(log_dir=tblog_dir)

    args.ckpt_dir = os.path.join(run_dir, 'rqvae_checkpoints')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    args.output_dir = os.path.join(run_dir, 'rqvae_output')
    os.makedirs(args.output_dir, exist_ok=True) 
    base_output_dir = "../../output_"+args.dataset
    #args.sasrec_emb_path=os.path.join(base_output_dir,"item_embeddings_sasrec.npy")
    args.sasrec_emb_path=os.path.join(base_output_dir,"best_item_embeddings.npy")


    # --- 训练阶段 ---
    logging.info("--- Phase 1: Training RQ-VAE Model ---")

    # 1. 加载SASRec嵌入数据
    logging.info(f"Loading item embeddings from: {args.sasrec_emb_path}")
    # **新逻辑**: 直接加载整个 .npy 文件，不再假设第0行为padding
    item_embeddings = np.load(args.sasrec_emb_path)
    embedding_dim = item_embeddings.shape[1]
    logging.info(f"Loaded {item_embeddings.shape[0]} item embeddings with dimension {embedding_dim}.")

    # 2. 创建数据集和数据加载器
    train_dataset = SimpleEmbeddingDataset(item_embeddings)
    data_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)

    # 3. 初始化RQ-VAE模型
    model = RQVAE(in_dim=embedding_dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters)
    logging.info("RQ-VAE Model Structure:\n" + str(model))

    # 4. 训练模型
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader, writer)
    logging.info(f"Training finished. Best Loss: {best_loss:.6f}, Best Collision Rate: {best_collision_rate:.6f}")


    # --- 推断与映射阶段 ---
    logging.info("\n--- Phase 2: Generating Codes with the Best Model ---")
    
    # 1. 加载刚刚训练好的最佳模型
    best_model_path = os.path.join(trainer.ckpt_dir, 'best_collision_model.pth')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Could not find the best model at '{best_model_path}'. Please check your Trainer's save logic.")

    logging.info(f"Loading best trained model from: {best_model_path}")
    #ckpt = torch.load(best_model_path, map_location=torch.device('cpu'))
    ckpt = torch.load(best_model_path, map_location=torch.device('cpu'),weights_only=False)
     
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 2. 对所有嵌入进行推断
    # **注意**: 这里我们使用完整的 `train_dataset`，因为它包含了所有的物品嵌入
    inference_loader = DataLoader(train_dataset, batch_size=args.batch_size * 2, shuffle=False)
    all_rq_indices = []
    all_reconstructed_embs = []

    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Generating Codes"):
            batch = batch.to(device)
            indices_batch = model.get_indices(batch, use_sk=False)
            reconstructed_batch, _, _ = model(batch)
            all_rq_indices.append(indices_batch.cpu().numpy())
            all_reconstructed_embs.append(reconstructed_batch.cpu().numpy())
    
    all_rq_indices = np.vstack(all_rq_indices)
    all_reconstructed_embs = np.vstack(all_reconstructed_embs)

    # 3. 将结果映射回原始ID并保存
    logging.info("Mapping results to original item IDs and saving...")
    orig_id_to_rq_code = {}
    orig_id_to_recon_emb = {}
    # 根据你的量化层数调整前缀 (len(num_emb_list))
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>"] 

    # **新逻辑**: 直接使用索引作为原始ID
    for i in tqdm(range(len(all_rq_indices)), desc="Saving Mappings"):
        original_id = i # 数组索引 i 直接对应原始物品ID i
        
        integer_codes = all_rq_indices[i]
        string_code = [prefix[j].format(code) for j, code in enumerate(integer_codes)]
        
        orig_id_to_rq_code[str(original_id)] = string_code
        orig_id_to_recon_emb[str(original_id)] = all_reconstructed_embs[i]

    # 4. 保存文件
    # 保存RQ编码映射
    output_code_path = os.path.join(args.output_dir, "original_item_id_to_rq_code.json")
    with open(output_code_path, 'w') as f:
        json.dump(orig_id_to_rq_code, f, indent=4)
    logging.info(f"RQ code mapping saved to: {output_code_path}")
    
    # 保存重构嵌入映射
    output_emb_path = os.path.join(args.output_dir, "original_item_id_to_reconstructed_emb.npz")
    np.savez_compressed(output_emb_path, **orig_id_to_recon_emb)
    logging.info(f"Reconstructed embedding mapping saved to: {output_emb_path}")

    logging.info("\n--- All tasks completed successfully! ---")





    # --- 新增: 统计编码冲突 ---
    logging.info("\n--- Phase 4: Analyzing Code Collision ---")
    
    total_items = len(all_rq_indices)
    
    # 使用 np.unique 找到所有唯一的编码行
    unique_codes = np.unique(all_rq_indices, axis=0)
    num_unique_codes = len(unique_codes)
    
    # 计算冲突的 item 数量和比例
    collided_item_count = total_items - num_unique_codes
    collision_rate = (collided_item_count / total_items) * 100 if total_items > 0 else 0
    
    logging.info(f"Total items: {total_items}")
    logging.info(f"Number of unique codes generated: {num_unique_codes}")
    logging.info(f"Number of items with duplicated codes: {collided_item_count}")
    logging.info(f"Collision rate: {collision_rate:.2f}% of items share a code with another item.")

    # (可选) 打印更详细的冲突信息
    # 注意: 如果物品数量巨大，这可能会消耗较多内存和时间
    if total_items < 100000: # 仅在数据集不大时执行详细分析
        logging.info("Performing detailed collision analysis...")
        # 将 numpy array 的每一行转换为 tuple，以便 Counter 可以统计
        code_tuples = [tuple(row) for row in all_rq_indices]
        code_counts = Counter(code_tuples)
        
        # 找出所有被多次使用的编码
        duplicated_codes = {code: count for code, count in code_counts.items() if count > 1}
        
        if duplicated_codes:
            # 按出现次数降序排序
            most_common_duplicates = sorted(duplicated_codes.items(), key=lambda item: item[1], reverse=True)
            logging.info(f"Found {len(duplicated_codes)} codes that are used by more than one item.")
            logging.info("Top 5 most common duplicated codes:")
            for code, count in most_common_duplicates[:5]:
                logging.info(f"  - Code {code} was used {count} times.")
        else:
            logging.info("No code collisions found.")
