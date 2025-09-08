import os
import sys
import copy
import torch
import random
import numpy as np
import pandas as pd
from model import SASRec
from collections import defaultdict
from multiprocessing import Process, Queue
from tqdm import tqdm
from queue import Empty

# --- 优化方案V3 (已添加连续特征标准化): 使用向量化极速处理特征 ---
def load_and_preprocess_features(feature_path, itemnum):
    """
    (向量化优化版 + 连续特征标准化)
    从 .pkl 文件加载物品特征，解析离散和连续特征，计算元数据，并构建对齐的特征张量。
    """
    print(f"--- 开始加载和预处理高级物品特征 (向量化版本) ---")
    print(f"从 '{feature_path}' 加载...")
    
    if not os.path.exists(feature_path):
        print(f"错误：特征文件未找到 '{feature_path}'")
        return None

    try:
        df_features = pd.read_pickle(feature_path)
        id_col = df_features.columns[0]

        # 1. 自动识别特征列
        discrete_cols = [c for c in df_features.columns if c.startswith('I_B_')]
        continuous_cols = [c for c in df_features.columns if c.startswith('I_S_')]
        print(f"识别到 {len(discrete_cols)} 个离散特征和 {len(continuous_cols)} 个连续特征。")

        # 2. 计算离散特征的基数
        print("正在计算离散特征基数...")
        discrete_cardinalities = [int(df_features[col].max() + 1) for col in discrete_cols]
        for col, card in zip(discrete_cols, discrete_cardinalities):
            print(f"  - 离散特征 '{col}' 的基数: {card}")

        # 3. 创建空的特征张量
        discrete_tensor = np.zeros((itemnum + 1, len(discrete_cols)), dtype=np.int64)
        continuous_tensor = np.zeros((itemnum + 1, len(continuous_cols)), dtype=np.float32)

        # --- 4. 向量化填充 ---
        print("开始向量化填充特征矩阵...")
        
        target_indices = df_features[id_col].values.astype(np.int64) + 1
        mask = target_indices <= itemnum
        valid_indices = target_indices[mask]
        
        # a. 一次性获取所有有效的离散和连续特征数据
        discrete_values = df_features.loc[mask, discrete_cols].values
        continuous_values = df_features.loc[mask, continuous_cols].values
        
        # # --- 5. 对连续特征进行标准化 (核心修改点) ---
        # if len(continuous_cols) > 0 and continuous_values.shape[0] > 0:
        #     print("正在对连续特征进行标准化 (StandardScaler)...")
        #     scaler = StandardScaler()
            
        #     # 使用从DataFrame中提取的有效值来拟合和转换
        #     scaled_continuous_values = scaler.fit_transform(continuous_values)
            
        #     # 检查标准化后的结果
        #     # print(f"  - 标准化后均值 (应接近0): {np.mean(scaled_continuous_values, axis=0)}")
        #     # print(f"  - 标准化后标准差 (应接近1): {np.std(scaled_continuous_values, axis=0)}")
        # else:
        #     # 如果没有连续特征或没有有效值，则创建一个形状正确的空数组
        #     scaled_continuous_values = np.zeros_like(continuous_values)

        # --- 6. 使用Numpy高级索引进行最终填充 ---
        discrete_tensor[valid_indices] = discrete_values
        # 使用标准化后的连续特征值进行填充
        continuous_tensor[valid_indices] = continuous_values
        
        print("特征矩阵填充完成。")

        feature_info = {
            'discrete_features_tensor': torch.LongTensor(discrete_tensor),
            'continuous_features_tensor': torch.FloatTensor(continuous_tensor),
            'discrete_cardinalities': discrete_cardinalities,
            'continuous_feature_count': len(continuous_cols)
        }
        
        print("--- 特征预处理完成 ---")
        return feature_info

    except Exception as e:
        print(f"加载或处理特征文件时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_single_file_as_dataframe(input_csv_path: str):
    """
    从单个CSV文件中加载数据并返回一个Pandas DataFrame。

    这个函数用于直接处理单个数据集文件，逻辑比合并多个文件更简单。

    Args:
        input_csv_path (str): 输入的CSV文件的完整路径。

    Returns:
        pandas.DataFrame: 如果成功，返回包含文件内容的DataFrame。
                          如果文件不存在或读取失败，则返回None。
    """
    print(f"正在从单个文件加载数据: '{input_csv_path}'")

    # 1. 检查文件是否存在
    if not os.path.exists(input_csv_path):
        print(f"错误：输入文件未找到，请检查路径 '{input_csv_path}'")
        return None

    # 2. 尝试读取CSV文件
    try:
        df = pd.read_csv(input_csv_path)
        
        # 检查读取后的DataFrame是否为空
        if df.empty:
            print("警告：文件已读取，但内容为空。")
        else:
            print(f"文件加载成功。总记录数: {len(df):,}")
            
        return df
        
    except Exception as e:
        print(f"读取文件 '{input_csv_path}' 时发生错误: {e}")
        return None


# --- 第2步：数据处理函数 (与之前的版本相同，无需修改) ---
# 这些函数接收DataFrame作为输入，因此可以复用

# --- 已修改：适配 0-based ID ---
def build_indices_from_dataframe(df):
    """
    从合并后的DataFrame中构建 u2i 和 i2u 索引。
    此版本会自动将 0-based ID 转换为 1-based ID (通过将所有 ID 加 1)。
    """
    print("正在从DataFrame构建索引 (将 0-based ID 转换为 1-based)...")
    df.dropna(subset=['user_id', 'sequence_item_ids'], inplace=True)
    df = df[df['sequence_item_ids'] != ''].copy()

    # --- 修改点开始 ---
    # 将 user_id 加 1
    df['user_id'] = df['user_id'].astype(int) + 1
    # 将 sequence_item_ids 中的每个 item_id 加 1
    df['sequence_item_ids'] = df['sequence_item_ids'].apply(
        lambda x: ','.join([str(int(i) + 1) for i in x.split(',')])
    )
    # --- 修改点结束 ---
    
    if df.empty:
        print("警告：DataFrame为空，无法构建索引。")
        return [], []

    n_users = df['user_id'].max()
    all_items_series = df['sequence_item_ids'].str.split(',').explode()
    n_items = pd.to_numeric(all_items_series, errors='coerce').max()
    print(f"数据转换后，发现 {n_users} 个用户和 {n_items} 个物品 (ID 从 1 开始)。")
    
    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]
    
    for _, row in df.iterrows():
        # 此处的 user_id 和 item_ids 已经是 1-based
        user_id = row['user_id']
        item_ids = [int(item) for item in str(row['sequence_item_ids']).split(',')]
        u2i_index[user_id].extend(item_ids)
        for item_id in item_ids:
            i2u_index[item_id].append(user_id)
            
    return u2i_index, i2u_index

# --- 已修改：适配 0-based ID ---
def partition_data_from_dataframe(df):
    """
    从合并后的DataFrame中划分训练集、验证集和测试集。
    此版本会自动将 0-based ID 转换为 1-based ID (通过将所有 ID 加 1)。
    """
    print("正在从DataFrame划分数据集 (将 0-based ID 转换为 1-based)...")
    User = {}
    df_clean = df.dropna(subset=['user_id', 'sequence_item_ids'])
    df_clean = df_clean[df_clean['sequence_item_ids'] != ''].copy()
    
    for _, row in df_clean.iterrows():
        # --- 修改点开始 ---
        # 读取原始 ID 并加 1
        user_id = int(row['user_id']) + 1
        sequence_items = [int(item) + 1 for item in str(row['sequence_item_ids']).split(',')]
        # --- 修改点结束 ---
        
        if user_id in User:
            User[user_id].extend(sequence_items)
        else:
            User[user_id] = sequence_items
            
    user_train, user_valid, user_test = {}, {}, {}
    for user, items in User.items():
        if len(items) < 3:
            user_train[user], user_valid[user], user_test[user] = items, [], []
        else:
            user_train[user] = items[:-2]
            user_valid[user] = [items[-2]]
            user_test[user] = [items[-1]]
            
    # usernum 和 itemnum 现在是转换后的最大 ID (1-based)
    usernum = max(User.keys()) if User else 0
    all_items = {item for items in User.values() for item in items}
    itemnum = max(all_items) if all_items else 0
    
    print(f"数据转换并划分后，用户数 (最大ID): {usernum}, 物品数 (最大ID): {itemnum}")
    
    return [user_train, user_valid, user_test, usernum, itemnum]


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while uid not in user_train or len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if u not in train or len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if u not in train or len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def extract_and_save_item_embeddings(model, output_path):
    """
    从训练好的模型中提取最终的、特征增强的物品嵌入。
    """
    print(f"\n--- 开始提取特征增强的物品嵌入 ---")
    model.eval()

    # 使用模型提供的方法获取所有物品的最终嵌入
    with torch.no_grad():
        item_embeddings = model.get_all_item_embeddings()
    
    item_embeddings_np = item_embeddings.cpu().numpy()
    np.save(output_path, item_embeddings_np)
    
    print(f"成功提取 {item_embeddings_np.shape[0]} 个物品的增强嵌入，维度为 {item_embeddings_np.shape[1]}")
    print(f"嵌入已保存到: {output_path}")
    print(f"--- 提取完成 ---")