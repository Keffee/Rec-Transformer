import pandas as pd
import csv
from pathlib import Path

def remap_ids_and_save(input_csv_path: str, output_dir: str = '.'):
    """
    读取筛选后的序列数据，为user_id和item_id创建新的连续映射，
    并保存映射文件和最终的重映射数据集。

    Args:
        input_csv_path (str): 经过筛选的CSV文件路径 (例如 'filtered_user_sequences.csv').
        output_dir (str): 输出文件（映射文件和最终数据）的存放目录。
    """
    input_path = Path(input_csv_path)
    output_path = Path(output_dir)

    if not input_path.is_file():
        print(f"错误：输入文件未找到，请检查路径 '{input_csv_path}'")
        return

    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 定义输出文件的路径
    remapped_data_path = output_path / '2_remapped_positive_data_100k.csv'
    user_map_path = output_path / '2_user_id_map.csv'
    item_map_path = output_path / '2_item_id_map.csv'

    print("开始读取数据...")
    df = pd.read_csv(input_path)

    # --- 1. 处理 User ID ---
    print("正在创建 User ID 映射...")
    # 获取所有唯一的、排序后的原始user_id
    original_users = sorted(df['user_id'].unique())
    # 创建从原始ID到新ID（从1开始）的映射
    user_map = {original_id: new_id for new_id, original_id in enumerate(original_users, start=1)}
    
    # 将映射关系保存到文件
    user_map_df = pd.DataFrame(user_map.items(), columns=['original_user_id', 'new_user_id'])
    user_map_df.to_csv(user_map_path, index=False)
    print(f"User ID 映射已保存到 '{user_map_path}'")

    # --- 2. 处理 Item ID ---
    print("正在创建 Item ID 映射...")
    # 使用集合来高效地收集所有唯一的item_id
    all_items = set()
    for seq in df['sequence_item_ids'].dropna():
        # 分割字符串并添加到集合中
        # 确保将item_id转换为整数，以便正确排序和作为字典键
        try:
            items = [int(i) for i in seq.split(',') if i]
            all_items.update(items)
        except ValueError:
            print(f"警告：在序列 '{seq}' 中发现非整数的item_id，已跳过。")
            
    # 获取所有唯一的、排序后的原始item_id
    original_items = sorted(list(all_items))
    # 创建从原始ID到新ID（从1开始）的映射
    item_map = {original_id: new_id for new_id, original_id in enumerate(original_items, start=1)}

    # 将映射关系保存到文件
    item_map_df = pd.DataFrame(item_map.items(), columns=['original_item_id', 'new_item_id'])
    item_map_df.to_csv(item_map_path, index=False)
    print(f"Item ID 映射已保存到 '{item_map_path}'")
    
    # --- 3. 创建并保存最终的重映射数据 ---
    print("正在生成最终的重映射数据文件...")
    
    # 应用映射来转换user_id列
    df['user_id'] = df['user_id'].map(user_map)

    # 定义一个函数来转换item_id序列
    def remap_sequence(seq_str):
        if pd.isna(seq_str) or not seq_str:
            return ""
        
        original_ids = [int(i) for i in seq_str.split(',') if i]
        # 使用item_map将原始ID列表转换为新ID列表
        new_ids = [str(item_map[oid]) for oid in original_ids]
        return ",".join(new_ids)

    # 应用该函数来转换sequence_item_ids列
    df['sequence_item_ids'] = df['sequence_item_ids'].apply(remap_sequence)

    # 保存最终的DataFrame
    # 使用 quoting=csv.QUOTE_NONNUMERIC 使得输出的序列字符串被引号包围
    df.to_csv(remapped_data_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"重映射后的数据已保存到 '{remapped_data_path}'")
    print("\n所有处理已完成！")


# --- 使用示例 ---
if __name__ == '__main__':
    # 这是上一步生成的输入文件名
    input_filtered_file = '1_1_train.csv'
    
    # 调用函数执行重映射
    remap_ids_and_save(input_filtered_file)