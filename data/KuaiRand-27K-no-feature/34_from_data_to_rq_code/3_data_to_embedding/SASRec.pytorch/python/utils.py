import sys
import copy
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Queue
import csv
import sys

csv.field_size_limit(sys.maxsize)
def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

def build_index_from_csv(csv_path):
    """
    从我们生成的包含序列的CSV文件中构建 u2i 和 i2u 索引。

    Args:
        csv_path (str): 'remapped_sequences.csv' 文件的路径。

    Returns:
        tuple: (u2i_index, i2u_index)
               u2i_index: 一个列表，索引为user_id，值为该用户交互过的item_id列表。
               i2u_index: 一个列表，索引为item_id，值为与该物品交互过的user_id列表。
    """
    print(f"正在从新的CSV格式构建索引: {csv_path}")
    
    # 1. 使用 pandas 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：索引文件未找到，请检查路径 '{csv_path}'")
        return None, None

    # 删除可能在处理过程中产生的空序列行
    df.dropna(subset=['user_id', 'sequence_item_ids'], inplace=True)
    df = df[df['sequence_item_ids'] != ''].copy()

    # 确保 user_id 是整数类型
    df['user_id'] = df['user_id'].astype(int)

    # --- 确定用户和物品ID的最大值 ---
    n_users = df['user_id'].max()

    # 要找到最大的 item_id，我们需要解析所有的序列字符串
    # 使用 .str.split().explode() 是一个高效的方法
    all_items_series = df['sequence_item_ids'].str.split(',').explode()
    # 转换为数值类型并找到最大值
    n_items = pd.to_numeric(all_items_series, errors='coerce').max()

    print(f"发现 {n_users} 个用户和 {n_items} 个物品。")

    # --- 初始化索引列表 ---
    # +1 是因为ID通常从1开始，而列表索引从0开始
    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    # --- 遍历DataFrame的每一行来填充索引 ---
    for _, row in df.iterrows():
        user_id = row['user_id']
        sequence_str = row['sequence_item_ids']
        
        # 将序列字符串分割成 item ID 列表（并转换为整数）
        item_ids = [int(item) for item in sequence_str.split(',')]

        # u2i_index: 对于这个user_id，添加他所有的item_id
        # 使用 extend 比循环 append 更高效
        u2i_index[user_id].extend(item_ids)

        # i2u_index: 对于这个序列中的每一个item_id，添加当前的user_id
        for item_id in item_ids:
            i2u_index[item_id].append(user_id)
            
    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

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


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def csv_data_partition(csv_file_path):
    import csv
    # 初始化用户-物品交互字典
    User = {}
    
    # 读取CSV文件
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头(如果有的话)
        
        for row in reader:
            user_id = int(row[0])
            sequence_items = [int(item) for item in row[1].split(',')]
            User[user_id] = sequence_items
    
    # 初始化训练集、验证集和测试集
    user_train = {}
    user_valid = {}
    user_test = {}
    
    # 处理每个用户的序列
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    
    # 找出最大的user_id和item_id
    usernum = max(User.keys()) 
    
    # 找出所有item_id
    all_items = set()
    for items in User.values():
        all_items.update(items)
    
    itemnum = max(all_items) 
    
    return [user_train, user_valid, user_test, usernum, itemnum]

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

        if len(train[u]) < 1 or len(test[u]) < 1: continue

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
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

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
