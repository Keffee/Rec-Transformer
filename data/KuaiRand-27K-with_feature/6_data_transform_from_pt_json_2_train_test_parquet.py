import argparse
import os
import json
import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="将自定义的推荐数据集转换为VeRL风格的Parquet格式，并自动划分为训练集和测试集。")
    parser.add_argument("--input_json", type=str, default='5_KuaiRand-27K_pt_data.json', help="输入的JSON文件路径。")
    parser.add_argument("--output_dir", type=str, default='./6_parquet_for_verl', help="输出Parquet文件的目录。")
    parser.add_argument("--data_source_name", type=str, default="KuaiRand-27K", help="为数据源指定一个名称。")
    parser.add_argument("--test_size", type=float, default=0.1, help="测试集所占的比例。")
    parser.add_argument("--seed", type=int, default=42, help="用于划分数据集的随机种子，确保可复现。")

    args = parser.parse_args()

    print("--- 开始处理 ---")
    args.input_json = os.path.join('output_'+args.data_source_name, args.input_json)
    print(f"加载文本数据源: {args.input_json}")

    try:
        df = pd.read_json(args.input_json)
    except FileNotFoundError as e:
        print(f"错误：输入文件未找到。请检查路径。详细信息: {e}")
        return
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return

    print(f"\n正在划分数据集... 测试集比例: {args.test_size}, 随机种子: {args.seed}")
    if len(df) < 2 or (len(df) * args.test_size < 1):
        print("警告：数据集太小，将所有数据用作训练集。")
        train_df = df
        test_df = df.iloc[0:0]
    else:
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed)

    print(f"划分完成 -> 训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")

    def process_dataframe_to_list(df, split_name):
        processed_list = []
        for idx, row in df.iterrows():
            text = row.get("text", "")
            user_id = row.get("user_id", "N/A")
            ground_truth_text = row.get("ground_truth", "")

            prompt_content = " ".join(text.split())
            #print('prompt content', prompt_content)
            ground_truth = " ".join(ground_truth_text.split())

            processed_data = {
                "data_source": args.data_source_name,
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "recommendation",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                # extra_info 只包含 user_id
                "extra_info": {
                    "user_id": user_id,
                    "original_index": idx,
                    "split": split_name
                },
            }
            processed_list.append(processed_data)
        return processed_list

    print("\n正在转换训练集格式...")
    train_data_list = process_dataframe_to_list(train_df, "train")

    print("正在转换测试集格式...")
    test_data_list = process_dataframe_to_list(test_df, "test")

    print("\n正在从处理后的列表创建 Hugging Face Datasets...")
    processed_train_dataset = datasets.Dataset.from_list(train_data_list)
    
    if test_data_list:
        processed_test_dataset = datasets.Dataset.from_list(test_data_list)
    else:
        processed_test_dataset = datasets.Dataset.from_list([])
    args.output_dir = os.path.join(args.output_dir, args.data_source_name)
    os.makedirs(args.output_dir, exist_ok=True)
    train_output_path = os.path.join(args.output_dir, "train.parquet")
    test_output_path = os.path.join(args.output_dir, "test.parquet")

    print(f"\n正在保存处理后的训练集到: {train_output_path}")
    processed_train_dataset.to_parquet(train_output_path)

    if len(processed_test_dataset) > 0:
        print(f"正在保存处理后的测试集到: {test_output_path}")
        processed_test_dataset.to_parquet(test_output_path)

    print("\n--- 处理完成！ ---")
    if len(processed_train_dataset) > 0:
        print("\n查看一条转换后的【训练集】数据示例:")
        print(json.dumps(processed_train_dataset[0], indent=2, ensure_ascii=False))

    if len(processed_test_dataset) > 0:
        print("\n查看一条转换后的【测试集】数据示例:")
        print(json.dumps(processed_test_dataset[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()