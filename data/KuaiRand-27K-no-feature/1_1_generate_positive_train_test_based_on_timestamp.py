# 这个程序和原有的1_select_positive_interactions_parallel.py不冲突，原来生成的那个非常干净，适合过sasrec然后rq，这里生成的结果是补了0的，非常难用。
# 大体上的目的是将每个序列都看作timestamp前的训练序列和timestamp后的测试序列，保证测试序列长度一致，多的用0填充，然后分别生成train.csv和test.csv
# test里面，时间戳后最长子序列长度为6223，

import pandas as pd
import csv
import multiprocessing
import numpy as np
from typing import List, Dict, Any, Tuple
from functools import partial
import os
# --- 步骤 1: 定义核心分析函数，用于第一阶段的并行处理 ---
def analyze_chunk(df_chunk: pd.DataFrame, cutoff_timestamp: int) -> Tuple[List[Dict[str, Any]], int]:
    """
    分析 DataFrame 的一个子集（块），筛选有效互动，并按时间戳分割序列。

    Args:
        df_chunk (pd.DataFrame): 输入的 DataFrame 数据块。
        cutoff_timestamp (int): 用于分割训练集和测试集的时间戳。

    Returns:
        Tuple[List[Dict[str, Any]], int]: 
            - 一个包含处理后用户数据的列表。每个用户是一个字典，包含 user_id, before_items, 和 after_items。
            - 这个数据块中，时间戳之后最长子序列的长度。
    """
    
    # 定义需要检查的action列
    action_columns = [
        'sequence_is_click', 'sequence_is_like', 'sequence_is_follow',
        'sequence_is_comment', 'sequence_is_forward', 'sequence_long_view',
        'sequence_profile_stay_time', 'sequence_comment_stay_time',
        'sequence_is_profile_enter'
    ]
    
    chunk_results = []
    local_max_after_len = 0

    # 遍历数据块中的每一行
    for _, row in df_chunk.iterrows():
        user_id = row['user_id']
        
        # 跳过没有序列数据的行
        if pd.isna(row['sequence_item_ids']) or not row['sequence_item_ids']:
            continue
            
        item_ids = str(row['sequence_item_ids']).split(',')
        is_hate = str(row['sequence_is_hate']).split(',')
        timestamps = str(row['sequence_timestamps']).split(',')
        
        is_recall = str(row['sequence_is_recall_candidate']).split(',') # added by wu
        
        actions_data = {col: str(row[col]).split(',') for col in action_columns}
        
        before_items = []
        after_items = []
        num_interactions = len(item_ids)
        
        for i in range(num_interactions):
            try:
                '''
                # 规则 1: 过滤掉 is_hate != 0 的记录
                if int(is_hate[i]) != 0:
                    continue

                # 规则 2: 检查是否有至少一个有效互动
                action_sum = 0
                for col in action_columns:
                    val_str = actions_data[col][i]
                    # 对于 'stay_time' 列，大于0即算作一次有效互动
                    if 'stay_time' in col:
                        if int(val_str) > 0:
                            action_sum += 1
                    else:
                        action_sum += int(val_str)
                '''
                action_sum = 0
                if is_recall[i]=='True': # added by wu
                    action_sum=1
                # 如果满足有效互动条件，则根据时间戳分类
                if action_sum >= 1:
                    timestamp = int(timestamps[i])
                    if timestamp < cutoff_timestamp:
                        before_items.append(item_ids[i])
                    else:
                        after_items.append(item_ids[i])

            except (IndexError, ValueError) as e:
                # 警告并跳过格式错误的记录
                print(f"警告：在处理 User ID {user_id} 的第 {i+1} 次交互时遇到数据格式问题: {e}。已跳过此条记录。")
                continue

        # 只有当用户至少有一个有效互动时，才保留该用户
        if before_items or after_items:
            chunk_results.append({
                'user_id': user_id,
                'before_items': before_items,
                'after_items': after_items
            })
            # 更新此块内的最长 "after" 序列长度
            local_max_after_len = max(local_max_after_len, len(after_items))
            
    return chunk_results, local_max_after_len

def process_and_split_data_parallel(input_files: List[str], train_output_path: str, test_output_path: str, cutoff_timestamp: int, num_processes: int = None):
    """
    通过多核并行处理多个CSV文件，筛选并按时间戳分割用户序列，最终生成统一的训练集和测试集文件。

    Args:
        input_files (List[str]): 输入的CSV文件路径列表。
        train_output_path (str): 训练集输出路径。
        test_output_path (str): 测试集输出路径。
        cutoff_timestamp (int): 分割时间戳。
        num_processes (int, optional): 使用的进程数。默认为机器的CPU核心数。
    """
    print("步骤 1: 开始读取并合并所有输入文件...")
    try:
        df_list = [pd.read_csv(f) for f in input_files]
        df = pd.concat(df_list, ignore_index=True)
    except FileNotFoundError as e:
        print(f"错误：输入文件未找到: {e}。请检查文件路径。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    if df.empty:
        print("所有输入文件均为空，无需处理。")
        return

    print(f"文件合并完成，共 {len(df)} 条记录。")

    # --- 步骤 2: 将 DataFrame 分割成多个块，并进行并行分析 ---
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    print(f"步骤 2: 使用 {num_processes} 个核心进行并行分析...")
    
    df_chunks = np.array_split(df, num_processes)
    
    # 使用 functools.partial 将固定的 cutoff_timestamp 参数传递给工作函数
    worker_func = partial(analyze_chunk, cutoff_timestamp=cutoff_timestamp)

    with multiprocessing.Pool(processes=num_processes) as pool:
        # pool.map 会返回一个包含所有结果的列表，每个结果是 (chunk_results, local_max_after_len)
        parallel_results = pool.map(worker_func, df_chunks)

    # --- 步骤 3: 汇总所有进程的结果 ---
    print("步骤 3: 汇总所有并行处理结果...")
    all_processed_users = []
    global_max_after_len = 0
    
    for chunk_users, local_max_len in parallel_results:
        all_processed_users.extend(chunk_users)
        global_max_after_len = max(global_max_after_len, local_max_len)
        
    print(f"分析完成！共找到 {len(all_processed_users)} 个有效用户。")
    print(f"时间戳之后的最长子序列长度为: {global_max_after_len}")

    # --- 步骤 4: 生成最终的训练集和测试集 ---
    print("步骤 4: 生成最终的训练集和测试集...")
    train_data = []
    test_data = []

    for user_record in all_processed_users:
        user_id = user_record['user_id']
        before_items = user_record['before_items']
        after_items = user_record['after_items']

        # 生成训练数据：只要 'before' 序列不为空，就加入训练集
        if before_items:
            train_data.append({
                'user_id': user_id,
                'sequence_item_ids': ",".join(before_items)
            })

        # 生成测试数据：'before' 序列作为输入，'after' 序列作为目标（需要填充）
        # 通常，测试集中也要求用户有历史行为
        if before_items:
            # 对 'after' 序列进行填充
            padding_needed = global_max_after_len - len(after_items)
            padded_after_items = after_items + ['0'] * padding_needed
            
            # 测试集的序列是 'before' 和 填充后 'after' 的拼接
            test_sequence = before_items + padded_after_items
            
            test_data.append({
                'user_id': user_id,
                'sequence_item_ids': ",".join(test_sequence)
            })

    # --- 步骤 5: 保存到 CSV 文件 ---
    print("步骤 5: 保存文件...")
    if train_data:
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(train_output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"训练集已成功保存到 '{train_output_path}'，共 {len(train_df)} 条记录。")
    else:
        print("没有生成任何训练数据。")

    if test_data:
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(test_output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"测试集已成功保存到 '{test_output_path}'，共 {len(test_df)} 条记录。")
    else:
        print("没有生成任何测试数据。")


# --- 使用示例 ---
if __name__ == '__main__':
    # 为了让 multiprocessing 在某些操作系统（如Windows）上正常工作，
    # 必须将主逻辑代码块放在 `if __name__ == '__main__':` 内部。
    
    # 定义输入和输出文件
    # 这里我们假设有4个输入文件，您可以根据实际情况修改
    #input_file_list = [f'sasrec_format_{i}.csv' for i in range(4)]
    input_file_list = [os.path.join('/home/jovyan/data/kuairand/KuaiRand-27K-Processed', f'sasrec_format_{i}.csv') for i in range(4)]
    train_output_file = '1_1_train.csv'
    test_output_file = '1_1_test.csv'
    
    # 定义分割时间戳
    CUTOFF_TIMESTAMP = 1651795200000
    
    # 调用新的并行处理与分割函数
    # 您可以手动指定进程数，例如 num_processes=4
    process_and_split_data_parallel(
        input_files=input_file_list,
        train_output_path=train_output_file,
        test_output_path=test_output_file,
        cutoff_timestamp=CUTOFF_TIMESTAMP,
        num_processes=4
    )