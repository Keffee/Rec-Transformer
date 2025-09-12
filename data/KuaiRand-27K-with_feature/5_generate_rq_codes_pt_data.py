import argparse
import json
import pandas as pd
from tqdm import tqdm
import os
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
    count = 0
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Sequences"):
        sequence_str = row.get('sequence_item_ids')
        ground_truth_str = row.get('ground_truth_item_ids')
        user_id = row.get('user_id')

        # 跳过空的或格式不正确的序列
        if not isinstance(sequence_str, str) or not sequence_str:
            continue
        if not isinstance(ground_truth_str, str) or not ground_truth_str:
            count += 1
            continue
        
        # 将逗号分隔的ID字符串分割成列表
        original_ids = sequence_str.split(',')
        ground_truth_ids = ground_truth_str.split(',')
        full_sequence_codes = []
        ground_truth_codes = []

        # 假设固定的 rq_codes
        #fixed_rq_codes = ["<a_256>", "<b_256>", "<c_256>"]  # 根据需要替换为你的固定值
        fixed_rq_codes = ["[UNK]"] * args.codelen  # 根据需要替换为你的固定值
        pad_codes = ["[PAD]"] * args.codelen  # 根据需要替换为你的固定值

        for item_id in original_ids:
            # item_id 在CSV中是字符串，正好可以作为JSON加载的字典的键
            rq_codes = rq_code_map.get(item_id)
            if rq_codes:
                # 如果找到了对应的编码，就将其加入最终序列
                full_sequence_codes.extend(rq_codes)
            else:
                # 检查 rq_codes 是否为 0
                if item_id == "0":
                    # 填充与其他 rq_codes 相同长度的 0
                    # 假设其他 rq_codes 的长度为 len(other_rq_codes)
                    #length_of_other_rq_codes = 3  # 如果 rq_codes 为空，则长度为 0
                    #full_sequence_codes.extend([0] * length_of_other_rq_codes)
                    full_sequence_codes.extend(pad_codes)
                else:
                    # 如果某个ID在映射文件中不存在，则记录下来并跳过
                    missing_ids.add(item_id)
                    # 使用固定的 rq_codes 填充
                    full_sequence_codes.extend(fixed_rq_codes)
                    
        '''
        这一段是将ground_truth转化成rq_code的，暂时改为下面的直接用id的部分
        for item_id in ground_truth_ids:
            ground_truth_rq_codes = rq_code_map.get(item_id)
            if ground_truth_rq_codes:
                ground_truth_codes.extend(ground_truth_rq_codes)
            else:
                ground_truth_codes.extend(fixed_rq_codes)'''
        for item_id in ground_truth_ids:
            ground_truth_codes.extend(item_id)
        
        # 如果这个序列经过转换后不为空，则进行格式化
        if full_sequence_codes:
            # 将所有RQ编码用空格连接成一个长字符串
            final_text = " ".join(full_sequence_codes)
            final_ground_truth = " ".join(ground_truth_codes)
            
            # 按照要求的格式创建字典
            output_item = {"text": final_text, "ground_truth": final_ground_truth, "user_id": user_id}

            # 加入最终的输出列表
            final_output_list.append(output_item)

    if missing_ids:
        print(f"\n警告：处理过程中发现 {len(set(missing_ids))} 个无法在映射文件中找到的物品ID。")
        print(f"部分缺失的ID示例: {list(missing_ids)[:10]}")

    # --- 第4步: 将最终结果保存为JSON文件 ---
    print(f"\n转换完成，共生成 {len(final_output_list)} 条编码序列。")
    print(f"正在将结果保存到: {args.output_path}")
    print('one sample of history first 30 tokens: ', final_output_list[0]['text'].split()[:30])
    print('one sample of history last 30 tokens: ', final_output_list[0]['text'].split()[-30:])   
    print('one sample of ground truth: ', final_output_list[0]['ground_truth'].split())

    try:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            # 使用indent=2可以让输出的JSON文件格式优美，易于阅读
            json.dump(final_output_list, f, ensure_ascii=False, indent=2)
        print("--- 所有任务已成功完成！ ---")
    except IOError as e:
        print(f"错误：无法写入输出文件。请检查路径和权限: {e}")
    print(f"共有 {count} 条序列因没有ground_truth被跳过。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert item ID sequences to RQ code sequences.")

    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name, e.g., KuaiRand-27K-0501")
    
    parser.add_argument('--codelen', type=int, default=3,
                        help="Number of rq_codes per item (default: 3)")

    args = parser.parse_args()

    # Base directory: output_{dataset}
    base_dir = f"output_{args.dataset}"

    # train_sequence_path = os.path.join(base_dir, "1_1_train.csv")
    # test_sequence_path = os.path.join(base_dir, "1_1_test.csv")
    # train_output_path = os.path.join(base_dir, "5_train_rq_codes_pt_data.json")
    # test_output_path = os.path.join(base_dir, "5_test_rq_codes_pt_data.json")
    all_sequence_path = os.path.join(base_dir, "1_1_KuaiRand-27K.csv")
    all_output_path = os.path.join(base_dir, "5_KuaiRand-27K_pt_data.json")


    input_json = os.path.join(base_dir, "original_item_id_to_rq_code.json")
    output_json = os.path.join(base_dir, "4_item_id_to_rq_code.json")

    if not os.path.exists(output_json):

        with open(input_json, "r", encoding="utf-8") as f:
            rq_code_map = json.load(f)

        # shift keys
        shifted_map = {str(int(k) + 1): v for k, v in rq_code_map.items()}

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(shifted_map, f, ensure_ascii=False, indent=2)

        print(f"Shifted rq_code_map saved to {output_json}")

    '''# --- Run for train ---
    train_args = argparse.Namespace(
        rq_map_path=output_json,
        sequence_data_path=train_sequence_path,
        output_path=train_output_path,
        codelen=args.codelen
    )
    convert_sequences(train_args)

    # --- Run for test ---
    test_args = argparse.Namespace(
        rq_map_path=output_json,
        sequence_data_path=test_sequence_path,
        output_path=test_output_path,
        codelen=args.codelen        
    )
    convert_sequences(test_args)'''
        # --- Run for all ---
    all_args = argparse.Namespace(
        rq_map_path=output_json,
        sequence_data_path=all_sequence_path,
        output_path=all_output_path,
        codelen=args.codelen        
    )
    convert_sequences(all_args)

    '''
    parser = argparse.ArgumentParser(description="Convert item ID sequences to RQ code sequences.")
    
    #parser.add_argument('--sequence_data_path', type=str, default='1_positive_data_100k.csv',
    #                    help="Path to the input CSV file with item sequences (e.g., '1_positive_data_100k.csv').")
    parser.add_argument('--sequence_data_path', type=str, default='1_1_test.csv',
                        help="Path to the input CSV file with item sequences (e.g., '1_positive_data_100k.csv').")
        
    parser.add_argument('--rq_map_path', type=str, default='4_item_id_to_rq_code.json',
                        help="Path to the JSON file mapping original item IDs to RQ codes (e.g., '4_item_id_to_rq_code.json').")
                        
    parser.add_argument('--output_path', type=str, default='5_test_rq_codes_pt_data.json',
                        help="Path for the output JSON file.")

    args = parser.parse_args()
    convert_sequences(args)
    '''