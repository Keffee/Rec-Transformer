import argparse
import os
import json
import datasets
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="将自定义的推荐数据集转换为VeRL风格的Parquet格式，并自动划分为训练集和测试集。")
    # --- 修改: 增加了 user_id_csv 参数 ---
    parser.add_argument("--input_json", type=str, default='5_rq_codes_pt_data.json', help="输入的JSON文件路径，包含 'text' 字段。")
    parser.add_argument("--user_id_csv", type=str, default='1_1_test.csv', help="包含 user_id 的CSV文件路径，需与JSON文件行对应。")
    # ------------------------------------
    parser.add_argument("--output_dir", type=str, default='./6_parquet_for_verl', help="输出Parquet文件的目录。")
    parser.add_argument("--data_source_name", type=str, default="KuaiRand-27K-demo", help="为数据源指定一个名称。")
    parser.add_argument("--test_size", type=float, default=0.1, help="测试集所占的比例。")
    parser.add_argument("--seed", type=int, default=42, help="用于划分数据集的随机种子，确保可复现。")
    
    args = parser.parse_args()

    print("--- 开始处理 ---")
    print(f"加载文本数据源: {args.input_json}")
    print(f"加载用户ID数据源: {args.user_id_csv}")

    try:
        # --- 修改: 分别加载 JSON 和 CSV，然后合并 ---
        df_text = pd.read_json(args.input_json)
        df_users = pd.read_csv(args.user_id_csv)

        # 校验：确保两个文件的行数完全一致
        if len(df_text) != len(df_users):
            raise ValueError(
                f"输入文件行数不匹配！JSON文件有 {len(df_text)} 行, "
                f"而CSV文件有 {len(df_users)} 行。它们必须严格对应。"
            )

        # 将 user_id 列添加到主 DataFrame 中
        df_text['user_id'] = df_users['user_id']
        
        # 从合并后的 DataFrame 创建 Hugging Face Dataset
        dataset = datasets.Dataset.from_pandas(df_text)
        
    except FileNotFoundError as e:
        print(f"错误：输入文件未找到。请检查路径。详细信息: {e}")
        return
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return

    # 划分数据集 (这部分逻辑保持不变)
    print(f"\n正在划分数据集... 测试集比例: {args.test_size}, 随机种子: {args.seed}")
    if len(dataset) < 2 or (len(dataset) * args.test_size < 1):
        print("警告：数据集太小，无法进行有效的训练/测试集划分。将所有数据用作训练集。")
        train_original_dataset = dataset
        # 创建一个空的测试集以避免后续代码出错
        test_original_dataset = dataset.select([]) 
    else:
        split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
        train_original_dataset = split_dataset['train']
        test_original_dataset = split_dataset['test']
        
    print(f"划分完成 -> 训练集大小: {len(train_original_dataset)}, 测试集大小: {len(test_original_dataset)}")

    # 创建一个映射函数
    def make_map_fn(split_name):
        def process_fn(example, idx):
            # --- 修改: 核心处理逻辑重写 ---
            text_raw = example.get("text", "")
            user_id = example.get("user_id", "N/A") # 从合并后的数据中获取 user_id

            # 1. 根据空格分割 token
            tokens = text_raw.split()

            # 2. 根据新的规则确定 prompt 和 ground_truth
            # 使用6223是因为似乎test子序列的最终填充结果长度是6223
            if len(tokens) >= 3*6223:
                prompt_tokens = tokens[:-3*6223]
                ground_truth_tokens = tokens[-3*6223:]
                
                prompt_content = " ".join(prompt_tokens)
                ground_truth = " ".join(ground_truth_tokens)
            else:
                # 处理序列长度小于3的边界情况
                print(f"警告：在处理索引 {idx} 时，token 数量 ({len(tokens)}) 小于3*6223。将使用全部内容作为 ground_truth。\n这通常不应该发生，得去看看是不是第5步test生成的时候没有顺利生成正确长度的rq编码，不过也可以先忽略反正强化也能勉强跑起来")
                prompt_content = ""
                ground_truth = text_raw

            # 3. 构建新的输出字典
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
            # -------------------------------------
            return processed_data
        return process_fn

    # 应用转换 (这部分逻辑保持不变，但内部的 process_fn 已更新)
    print("\n正在转换训练集格式...")
    processed_train_dataset = train_original_dataset.map(
        function=make_map_fn("train"), 
        with_indices=True,
        # 移除处理前的列: 'text' 和 'user_id'
        remove_columns=train_original_dataset.column_names 
    )

    print("正在转换测试集格式...")
    if len(test_original_dataset) > 0:
        processed_test_dataset = test_original_dataset.map(
            function=make_map_fn("test"),
            with_indices=True,
            remove_columns=test_original_dataset.column_names
        )
    else:
        processed_test_dataset = test_original_dataset # 保持为空

    # 保存文件 (这部分逻辑保持不变)
    os.makedirs(args.output_dir, exist_ok=True)
    train_output_path = os.path.join(args.output_dir, "train.parquet")
    test_output_path = os.path.join(args.output_dir, "test.parquet")
    
    print(f"\n正在保存处理后的训练集到: {train_output_path}")
    processed_train_dataset.to_parquet(train_output_path)
    
    if len(processed_test_dataset) > 0:
        print(f"正在保存处理后的测试集到: {test_output_path}")
        processed_test_dataset.to_parquet(test_output_path)

    # 打印示例 (这部分逻辑保持不变)
    print("\n--- 处理完成！ ---")
    if len(processed_train_dataset) > 0:
        print("\n查看一条转换后的【训练集】数据示例:")
        print(json.dumps(processed_train_dataset[0], indent=2, ensure_ascii=False))

    if len(processed_test_dataset) > 0:
        print("\n查看一条转换后的【测试集】数据示例:")
        print(json.dumps(processed_test_dataset[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # 为了方便本地测试，您可以像下面这样创建一个虚拟的数据文件
    # if not os.path.exists('data'):
    #     os.makedirs('data')
    # with open('data/sample_input.json', 'w') as f:
    #     json.dump([
    #         {"text": "<a_107> <b_8> <c_123> <a_107> <b_119> <c_49> <a_107> <b_138> <c_96> <a_100> <b_110> <c_168> <a_69> <b_236> <c_148>"},
    #         {"text": "<a_44> <b_2> <c_73> <a_190> <b_141> <c_134> <a_190> <b_3> <c_145> <a_64> <b_4> <c_33>"},
    #         {"text": "<a_1> <b_2>"}
    #     ], f)
    # with open('data/sample_users.csv', 'w') as f:
    #     f.write('"user_id","sequence_item_ids"\n')
    #     f.write('0,"1,2,3,4,5"\n')
    #     f.write('1,"6,7,8,9"\n')
    #     f.write('2,"10"\n')

    main()