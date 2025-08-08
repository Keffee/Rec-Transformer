import pandas as pd
import csv

def process_user_sequences(input_csv_path: str, output_csv_path: str):
    """
    筛选用户交互序列并生成新的CSV文件。

    Args:
        input_csv_path (str): 输入的CSV文件路径。
        output_csv_path (str): 处理后输出的CSV文件路径。
    """
    try:
        # 读取原始CSV文件
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"错误：输入文件未找到，请检查路径 '{input_csv_path}'")
        return

    # 定义需要检查的action列
    # 注意：两个stay_time列不是二值的，但我们将其视为 > 0 时为有效action
    action_columns = [
        'sequence_is_click', 'sequence_is_like', 'sequence_is_follow',
        'sequence_is_comment', 'sequence_is_forward', 'sequence_long_view',
        'sequence_profile_stay_time', 'sequence_comment_stay_time',
        'sequence_is_profile_enter'
    ]

    # 存储最终结果的列表
    final_results = []

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        user_id = row['user_id']
        
        # 处理空序列的情况
        if pd.isna(row['sequence_item_ids']) or not row['sequence_item_ids']:
            continue
            
        # 将所有序列字符串按逗号分割成列表
        item_ids = str(row['sequence_item_ids']).split(',')
        is_hate = str(row['sequence_is_hate']).split(',')
        
        # 将所有action列也分割成列表
        actions_data = {col: str(row[col]).split(',') for col in action_columns}
        
        # 用于存储该用户筛选后保留的item_id
        kept_item_ids = []
        
        # 遍历序列中的每一次交互
        num_interactions = len(item_ids)
        for i in range(num_interactions):
            try:
                # --- 条件1: sequence_is_hate == 0 ---
                if int(is_hate[i]) != 0:
                    continue  # 如果is_hate为1，则跳过此次交互

                # --- 条件2: 至少有一个action为1 ---
                action_sum = 0
                
                # 累加所有action的值
                # 对于二值型action，直接加其值 (0 或 1)
                # 对于数值型action (如 stay_time)，如果值 > 0，则算作 1
                for col in action_columns:
                    val_str = actions_data[col][i]
                    if 'stay_time' in col:
                        # 对时间列，大于0则视为有action
                        if int(val_str) > 0:
                            action_sum += 1
                    else:
                        # 对其他0/1列，直接加其值
                        action_sum += int(val_str)

                # 如果action总和大于等于1，则满足条件
                if action_sum >= 1:
                    kept_item_ids.append(item_ids[i])

            except (IndexError, ValueError) as e:
                # 如果某一行的数据格式有问题（例如序列长度不一致），打印错误并跳过
                print(f"警告：在处理 User ID {user_id} 的第 {i+1} 次交互时遇到数据格式问题: {e}。已跳过此条记录。")
                continue

        # 如果筛选后仍有保留的item，则将结果加入最终列表
        if kept_item_ids:
            final_results.append({
                'user_id': user_id,
                'sequence_item_ids': ",".join(kept_item_ids)
            })

    # 如果有结果，则创建新的DataFrame并保存到CSV
    if final_results:
        output_df = pd.DataFrame(final_results)
        # 使用 quoting=csv.QUOTE_NONNUMERIC 来确保输出的 item_ids 字符串被引号包围，与示例格式一致
        output_df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"处理完成！结果已保存到 '{output_csv_path}'")
    else:
        print("处理完成，但没有符合条件的记录被保留。")


# --- 使用示例 ---
if __name__ == '__main__':
    # 将 'your_input_file.csv' 替换成你的原始文件名
    input_file = 'seq_data_100k.csv'
    
    # 定义你希望输出的文件名
    output_file = '1_positive_data_100k.csv'
    
    process_user_sequences(input_file, output_file)