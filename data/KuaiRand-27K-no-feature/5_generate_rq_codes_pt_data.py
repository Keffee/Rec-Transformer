import argparse
import json
import pandas as pd
from tqdm import tqdm

def convert_sequences(args):
    """
    将物品ID序列转换为RQ编码序列。

    Args:
        args (argparse.Namespace): 包含所有文件路径的命令行参数。
    """
    print("--- 开始将物品ID序列转换为RQ编码序列 ---")

    # --- 第1步: 加载RQ编码映射文件 ---
    # 这个文件是我们的“词典”，它告诉我们每个原始item_id对应哪个RQ编码。
    try:
        print(f"正在加载RQ编码映射文件: {args.rq_map_path}")
        with open(args.rq_map_path, 'r') as f:
            rq_code_map = json.load(f)
        print(f"成功加载 {len(rq_code_map)} 个物品的RQ编码映射。")
    except FileNotFoundError:
        print(f"错误：RQ编码映射文件未找到，请检查路径: '{args.rq_map_path}'")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解析JSON文件，请检查文件格式: '{args.rq_map_path}'")
        return

    # --- 第2步: 加载并处理物品序列文件 ---
    # 这个文件是我们需要翻译的原始序列数据。
    try:
        print(f"正在加载物品序列文件: {args.sequence_data_path}")
        df = pd.read_csv(args.sequence_data_path)
    except FileNotFoundError:
        print(f"错误：物品序列文件未找到，请检查路径: '{args.sequence_data_path}'")
        return

    # --- 第3步: 遍历序列，进行转换和拼接 ---
    print("正在转换序列...")
    final_output_list = []
    missing_ids = set() # 用于记录所有未在映射文件中找到的ID

    # 使用tqdm来显示进度条，对处理大数据非常有用
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Sequences"):
        sequence_str = row.get('sequence_item_ids')

        # 跳过空的或格式不正确的序列
        if not isinstance(sequence_str, str) or not sequence_str:
            continue
        
        # 将逗号分隔的ID字符串分割成列表
        original_ids = sequence_str.split(',')
        
        full_sequence_codes = []


        # 假设固定的 rq_codes
        fixed_rq_codes = ["<a_256>", "<b_256>", "<c_256>"]  # 根据需要替换为你的固定值

        for item_id in original_ids:
            # item_id 在CSV中是字符串，正好可以作为JSON加载的字典的键
            rq_codes = rq_code_map.get(item_id)

            if rq_codes:
                # 如果找到了对应的编码，就将其加入最终序列
                full_sequence_codes.extend(rq_codes)
            else:
                # 如果某个ID在映射文件中不存在，则记录下来并跳过
                missing_ids.add(item_id)

                # 检查 rq_codes 是否为 0
                if item_id == 0:
                    # 填充与其他 rq_codes 相同长度的 0
                    # 假设其他 rq_codes 的长度为 len(other_rq_codes)
                    length_of_other_rq_codes = 3  # 如果 rq_codes 为空，则长度为 0
                    full_sequence_codes.extend([0] * length_of_other_rq_codes)
                else:
                    # 使用固定的 rq_codes 填充
                    full_sequence_codes.extend(fixed_rq_codes)
        
        # 如果这个序列经过转换后不为空，则进行格式化
        if full_sequence_codes:
            # 将所有RQ编码用空格连接成一个长字符串
            final_text = " ".join(full_sequence_codes)
            
            # 按照要求的格式创建字典
            output_item = {"text": final_text}
            
            # 加入最终的输出列表
            final_output_list.append(output_item)

    if missing_ids:
        print(f"\n警告：处理过程中发现 {len(missing_ids)} 个无法在映射文件中找到的物品ID。")
        print(f"部分缺失的ID示例: {list(missing_ids)[:10]}")

    # --- 第4步: 将最终结果保存为JSON文件 ---
    print(f"\n转换完成，共生成 {len(final_output_list)} 条编码序列。")
    print(f"正在将结果保存到: {args.output_path}")
    
    try:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            # 使用indent=2可以让输出的JSON文件格式优美，易于阅读
            json.dump(final_output_list, f, ensure_ascii=False, indent=2)
        print("--- 所有任务已成功完成！ ---")
    except IOError as e:
        print(f"错误：无法写入输出文件。请检查路径和权限: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert item ID sequences to RQ code sequences.")
    
    parser.add_argument('--sequence_data_path', type=str, default='1_positive_data_100k.csv',
                        help="Path to the input CSV file with item sequences (e.g., '1_positive_data_100k.csv').")
    
    parser.add_argument('--rq_map_path', type=str, default='4_item_id_to_rq_code.json',
                        help="Path to the JSON file mapping original item IDs to RQ codes (e.g., '4_item_id_to_rq_code.json').")
                        
    parser.add_argument('--output_path', type=str, default='5_rq_codes_pt_data.json',
                        help="Path for the output JSON file.")

    args = parser.parse_args()
    convert_sequences(args)