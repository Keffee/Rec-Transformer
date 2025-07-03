import csv
import json
import argparse
import os

def convert_csv_to_llamafactory_json(csv_file_path: str, output_file_path: str, column_name: str = "sequence_item_ids"):
    """
    Reads a CSV file, extracts a specific column, and converts it into a JSON format
    suitable for Llama-Factory.

    The output format will be a list of dictionaries, e.g., [{"text": "value1"}, {"text": "value2"}].

    Args:
        csv_file_path (str): The path to the input CSV file.
        output_file_path (str): The path where the output JSON file will be saved.
        column_name (str): The name of the column to extract from the CSV.
                           Defaults to "sequence_item_ids".
    """
    # 检查输入文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误：输入文件不存在 -> {csv_file_path}")
        return

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"正在读取 CSV 文件: {csv_file_path}...")
    
    data_to_write = []
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as infile:
            # 使用 DictReader 可以方便地通过列名访问数据
            reader = csv.DictReader(infile)
            
            # 检查指定的列是否存在
            if column_name not in reader.fieldnames:
                print(f"错误：在 CSV 文件中找不到列名 '{column_name}'。")
                print(f"可用的列名有: {', '.join(reader.fieldnames)}")
                return

            for i, row in enumerate(reader):
                # 获取 sequence_item_ids 列的值
                # row.get(column_name, '') 确保即使某行该列为空，也不会报错
                sequence_text = row.get(column_name, '').strip()
                
                # 只有当该列有内容时才添加
                if sequence_text:
                    # 构建 Llama-Factory 期望的字典格式
                    data_to_write.append({"text": sequence_text})
                else:
                    print(f"警告：第 {i+2} 行的 '{column_name}' 列为空，已跳过。")
            
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return

    print(f"成功处理了 {len(data_to_write)} 条记录。")

    print(f"正在将数据写入 JSON 文件: {output_file_path}...")
    try:
        with open(output_file_path, mode='w', encoding='utf-8') as outfile:
            # ensure_ascii=False 保证中文字符等不会被转义
            # indent=2 使 JSON 文件更具可读性（可选，对于 Llama-Factory 不是必须的）
            json.dump(data_to_write, outfile, ensure_ascii=False, indent=2)
        print("转换完成！")
    except Exception as e:
        print(f"写入 JSON 文件时发生错误: {e}")

if __name__ == '__main__':
    # 使用 argparse 来让脚本更灵活，可以从命令行接收参数
    parser = argparse.ArgumentParser(
        description="将推荐数据集的 CSV 文件转换为 Llama-Factory 训练所需的 JSON 格式。"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="输入的 CSV 文件路径。"
    )
    parser.add_argument(
        "output_json",
        type=str,
        help="输出的 JSON 文件路径。"
    )
    parser.add_argument(
        "--column",
        type=str,
        default="sequence_item_ids",
        help="要提取的列名 (默认为 'sequence_item_ids')."
    )

    args = parser.parse_args()

    convert_csv_to_llamafactory_json(args.input_csv, args.output_json, args.column)