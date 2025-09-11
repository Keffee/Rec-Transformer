import os
import time
import torch
import argparse

from model import SASRec
from utils import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='KuaiRand_27K')
parser.add_argument('--train_dir', default='default')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--patience', default=5, type=int, help='Number of epochs to wait for improvement before early stopping.')
parser.add_argument('--feature_path', default=r'/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K/KuaiRand-27K-Processed/item_feat_norm.pkl', type=str, help='Path to the item features .pkl file.')
parser.add_argument('--feature_emb_dim', default=5, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    # # global dataset
    # # dataset = data_partition(args.dataset)
    # csv_file_path = r'/home/kfwang/20250813复现Onerec/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K/KuaiRand-27K-Processed/1_positive_data_0.csv'
    # u2i_index, i2u_index = build_index_from_csv(csv_file_path)
    # dataset = csv_data_partition(csv_file_path)

    # 1. 定义你的数据文件所在的目录和文件命名规则
    #input_csv_path = r'/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K/KuaiRand-27K-Processed/1_1_train.csv'
    base_output_dir = "../../../../output_"+args.dataset
    input_csv_path = os.path.join(base_output_dir, "1_1_train.csv")
    embedding_output_file = os.path.join(base_output_dir, "item_embeddings_sasrec.npy")
    args.feature_path = os.path.join(base_output_dir, "item_feat_norm.pkl")
    # # 文件名的模板，{}是后续用来填充数字的占位符
    # filename_pattern = "1_positive_data_{}.csv" 
    # # 你想要加载的文件的编号，range(4) 会生成 0, 1, 2, 3
    # file_indices_to_load = range(4) 
    best_model_path = r'/home/jovyan/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K-with_feature/34_from_data_to_rq_code/3_data_to_embedding/SASRec.pytorch/python/output_KuaiRand-27K-0501/bs80_lr0.0001_L2000_d128_blk2_h1_fd64/SASRec_best.pth'

    #embedding_output_file = r'/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K/KuaiRand-27K-Processed/34_from_data_to_rq_code/3_data_to_embedding/SASRec.pytorch/KuaiRand_27K_default_unfinished_but_stoped_earlier/item_embeddings_24epoch.npy'


    # 2. 调用新函数来加载并合并指定的数据文件
    combined_data_df = load_single_file_as_dataframe(input_csv_path)

    # 3. 检查数据是否成功加载，然后传递给处理函数
    if combined_data_df is not None:
        dataset = partition_data_from_dataframe(combined_data_df)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
        
        # --- 修改：调用新的特征处理函数 ---
        feature_info = None
        if args.feature_path:
            feature_info = load_and_preprocess_features(args.feature_path, itemnum)
            if feature_info is None:
                print("特征加载失败，将不使用特征进行训练。")
        else:
            print("未提供特征文件路径，将不使用特征进行训练。")
        # --- 修改结束 ---
    else:
        raise ValueError("combined_data_df is None")


    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    #best_model_path = r'/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K/KuaiRand-27K-Processed/34_from_data_to_rq_code/3_data_to_embedding/SASRec.pytorch/KuaiRand_27K_default_unfinished_but_stoped_earlier/SASRec.epoch=24.lr=0.0001.layer=2.head=1.hidden=50.maxlen=10000.pth'

    #embedding_output_file = r'/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer/data/KuaiRand-27K/KuaiRand-27K-Processed/34_from_data_to_rq_code/3_data_to_embedding/SASRec.pytorch/KuaiRand_27K_default_unfinished_but_stoped_earlier/item_embeddings_24epoch.npy'

    print(f"\n--- 开始提取物品嵌入 ---")
    print(f"从模型检查点加载: {best_model_path}")

    # 1. 重新初始化模型结构
    #    这是必要的，因为我们需要一个干净的模型实例来加载状态字典。
    model = SASRec(usernum, itemnum, args, feature_info=feature_info).to(args.device)

    # 2. 加载最佳模型的状态字典
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device(args.device)))
    model.eval()  # 设置为评估模式

    #extract_and_save_item_embeddings(
    #    model=model,
    #    output_path=embedding_output_file
    #)
    extract_and_save_item_embeddings_batch(
        model=model,
        output_path=embedding_output_file
    )
    print("Done")
