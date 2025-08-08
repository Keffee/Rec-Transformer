import argparse
import os
import json
import datasets
import pandas as pd # 使用 pandas 可以更方便地加载列表格式的JSON

def extract_ground_truth(output_str: str) -> str:
    """
    从模型的输出字符串中提取最后一个数字作为 ground_truth。
    例如，从 "2320, 2876, 160" 中提取 "160"。
    """
    try:
        last_item = output_str.split(',')[-1].strip()
        float(last_item) # 验证是否为数字
        return last_item
    except (IndexError, ValueError) as e:
        print(f"警告：无法从 '{output_str}' 中提取有效的最后一个数字。错误: {e}。将返回空字符串。")
        return ""

def main():
    parser = argparse.ArgumentParser(description="将自定义的推荐数据集JSON转换为VeRL风格的Parquet格式，并自动划分为训练集和测试集。")
    parser.add_argument("--input_json", type=str, default=r'/home/kfwang/20250613Rec-Factory/data/amazon_SPIAO_from_qspan/llama_leave_3_sft_format.json', help="输入的JSON文件路径。")
    parser.add_argument("--output_dir", type=str, default=r'/home/kfwang/20250613Rec-Factory/data/amazon_SPIAO_from_qspan/SPIAO_parquet', help="输出Parquet文件的目录。")
    parser.add_argument("--data_source_name", type=str, default="SPIAO", help="为数据源指定一个名称。")
    parser.add_argument("--test_size", type=float, default=0.1, help="测试集所占的比例。")
    parser.add_argument("--seed", type=int, default=42, help="用于划分数据集的随机种子，确保可复现。")
    
    args = parser.parse_args()

    print(f"--- 开始处理 ---")
    print(f"加载数据源: {args.input_json}")

    # 使用 pandas 加载 JSON 文件，然后转换为 Hugging Face Dataset
    df = pd.read_json(args.input_json)
    dataset = datasets.Dataset.from_pandas(df)

    # --- ！！！核心修改：在这里划分数据集！！！ ---
    print(f"正在划分数据集... 测试集比例: {args.test_size}, 随机种子: {args.seed}")
    split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    # 分别获取训练集和测试集
    train_original_dataset = split_dataset['train']
    test_original_dataset = split_dataset['test']
    print(f"划分完成 -> 训练集大小: {len(train_original_dataset)}, 测试集大小: {len(test_original_dataset)}")

    # ---------------------------------------------------

    
    instruction_following = "根据用户历史交互序列，预测他们接下来可能感兴趣的物品。"

    # 创建一个映射函数
    def make_map_fn(split_name):
        def process_fn(example, idx):
            instruction_raw = example.get("instruction", "")
            output_raw = example.get("output", "")
            # prompt_content = f"历史序列: {instruction_raw}\n{instruction_following}"
            prompt_content = instruction_raw
            ground_truth = extract_ground_truth(output_raw)

            processed_data = {
                "data_source": args.data_source_name,
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "recommendation",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split_name, 
                    "index": idx,
                    "answer": output_raw,
                    "question": instruction_raw,
                },
            }
            return processed_data
        return process_fn

    print("\n正在转换训练集格式...")
    processed_train_dataset = train_original_dataset.map(
        function=make_map_fn("train"), 
        with_indices=True,
        remove_columns=train_original_dataset.column_names
    )

    print("正在转换测试集格式...")
    processed_test_dataset = test_original_dataset.map(
        function=make_map_fn("test"),
        with_indices=True,
        remove_columns=test_original_dataset.column_names
    )
    # ----------------------------------------------------

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    train_output_path = os.path.join(args.output_dir, "train.parquet")
    test_output_path = os.path.join(args.output_dir, "test.parquet")
    
    # --- ！！！核心修改：分别保存两个文件！！！ ---
    print(f"\n正在保存处理后的训练集到: {train_output_path}")
    processed_train_dataset.to_parquet(train_output_path)
    
    print(f"正在保存处理后的测试集到: {test_output_path}")
    processed_test_dataset.to_parquet(test_output_path)
    # ------------------------------------------------

    print("\n--- 处理完成！ ---")
    print("\n查看一条转换后的【训练集】数据示例:")
    print(json.dumps(processed_train_dataset[0], indent=2, ensure_ascii=False))

    if len(processed_test_dataset) > 0:
        print("\n查看一条转换后的【测试集】数据示例:")
        print(json.dumps(processed_test_dataset[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()