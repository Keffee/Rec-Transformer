import argparse
import random
import torch
import numpy as np
import pandas as pd
import logging
import os
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 确保这些模块可以被正确导入 ---
from models.rqvae import RQVAE
from trainer import Trainer # 假设您的 Trainer 类在这里

#
# --- 第1部分: 脚本设置和辅助函数 ---
#

def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-VAE and Generate Mapped Codes")

    # --- 输入/输出路径参数 ---
    parser.add_argument("--sasrec_emb_path", type=str, default=r'/home/jovyan/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K-no-feature/3_IdZeroIsPadding_item_embeddings.npy',
                        help="Path to the item embeddings .npy file from SASRec.")
    parser.add_argument("--item_map_path", type=str, default=r'/home/jovyan/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K-no-feature/2_item_id_map.csv',
                        help="Path to the item_id_map.csv file.")
    parser.add_argument("--ckpt_dir", type=str, default="./rqvae_checkpoints",
                        help="Directory to save RQ-VAE model checkpoints.")
    parser.add_argument("--output_dir", type=str, default="./rqvae_output",
                        help="Directory to save the final code and embedding mappings.")

    # --- 训练超参数 (从您原来的脚本中保留) ---
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs') # 减少默认值以便快速测试
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--weight_decay", type=float, default=0.0, help='l2 regularization weight')
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    # --- RQ-VAE 模型特定参数 (从您原来的脚本中保留) ---
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
    parser.add_argument('--save_limit', type=int, default=5)

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
    logging.basicConfig(level=logging.INFO)
    print("====================== ARGS ======================")
    print(args)
    print("==================================================")

    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 训练阶段 ---
    logging.info("--- Phase 1: Training RQ-VAE Model ---")

    # 1. 加载SASRec嵌入数据
    logging.info(f"Loading SASRec embeddings from: {args.sasrec_emb_path}")
    all_embeddings = np.load(args.sasrec_emb_path)
    
    # **重要**: SASRec的嵌入向量中，索引0是为padding保留的。我们只应该用有效的物品嵌入来训练RQ-VAE。
    valid_embeddings = all_embeddings[1:]
    embedding_dim = valid_embeddings.shape[1]
    logging.info(f"Loaded {valid_embeddings.shape[0]} valid item embeddings with dimension {embedding_dim}.")

    # 2. 创建数据集和数据加载器
    train_dataset = SimpleEmbeddingDataset(valid_embeddings)
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
    best_loss, best_collision_rate = trainer.fit(data_loader)
    logging.info(f"Training finished. Best Loss: {best_loss:.6f}, Best Collision Rate: {best_collision_rate:.6f}")


    # --- 推断与映射阶段 ---
    logging.info("\n--- Phase 2: Generating Codes with the Best Model ---")
    
    # 1. 加载刚刚训练好的最佳模型
    # **假设**: 您的Trainer实现会将最佳模型保存为 'best_model.ckpt'
    best_model_path = os.path.join(trainer.ckpt_dir, 'best_collision_model.pth')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Could not find the best model at '{best_model_path}'. Please check your Trainer's save logic.")

    logging.info(f"Loading best trained model from: {best_model_path}")
    # 注意: 加载模型时需要重新初始化一个结构完全相同的实例
    ckpt = torch.load(best_model_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 2. 加载 item_id 映射文件
    logging.info(f"Loading item ID map from: {args.item_map_path}")
    item_map_df = pd.read_csv(args.item_map_path)
    id_map_dict = pd.Series(item_map_df.original_item_id.values, index=item_map_df.new_item_id).to_dict()

    # 3. 对所有有效嵌入进行推断
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

    # 4. 将结果映射回原始ID并保存
    logging.info("Mapping results back to original item IDs and saving...")
    orig_id_to_rq_code = {}
    orig_id_to_recon_emb = {}
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>"] # 根据你的量化层数调整

    for i in range(len(valid_embeddings)):
        new_id = i + 1 # new_item_id从1开始, 数组索引i从0开始
        original_id = id_map_dict.get(new_id)
        if original_id is None: continue

        integer_codes = all_rq_indices[i]
        string_code = [prefix[j].format(code) for j, code in enumerate(integer_codes)]
        
        orig_id_to_rq_code[str(original_id)] = string_code
        orig_id_to_recon_emb[str(original_id)] = all_reconstructed_embs[i]

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