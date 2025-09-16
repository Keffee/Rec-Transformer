# preprocess_and_tokenize.py

import os
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm

# =================================================================
# 1. 定义路径
# =================================================================
# 输入的 JSON 文件
json_file_path = '5_KuaiRand-27K_pt_data.json'

# Tokenizer 所在的模型检查点路径
tokenizer_path = '/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer/try_train/llama-rec_KuaiRand-27K-with-feature-checkpoints/checkpoint-2277'

# 预处理后数据集的保存路径 (一个新文件夹)
output_dir = './7_KuaiRand-27K-tokenized-for-grpo'

# =================================================================
# 2. 加载 Tokenizer
# =================================================================
print(f"从 '{tokenizer_path}' 加载 Tokenizer...")
# 使用 AutoTokenizer.from_pretrained 加载
# fast=True 会使用更快的 Rust-based tokenizer (如果可用)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

# =================================================================
# 3. 加载并处理数据集
# =================================================================
print(f"从 '{json_file_path}' 加载原始数据...")
# 使用 datasets 库直接加载 JSON 文件，这比手动读取更高效
# split='train' 表示我们将整个文件视为训练集
dataset = load_dataset('json', data_files=json_file_path, split='train')

# 定义分词函数
def tokenize_function(examples):
    """对一批数据进行分词"""
    # 关键步骤：
    # 1. 我们对 'text' 字段进行分词。
    # 2. return_token_type_ids=False 确保不会生成那个导致报错的字段。
    # 3. padding 和 truncation 是可选的，但通常是好习惯，
    #    这里我们暂时不设置，因为 GRPOTrainer 内部有自己的处理逻辑。
    tokenized_output = tokenizer(
        examples['text'],
        return_token_type_ids=False 
    )
    return tokenized_output

print("开始对数据集进行分词...")
# 使用 .map() 方法将分词函数应用到整个数据集
# batched=True 让函数一次性接收一批数据，速度更快
# remove_columns=['text'] 会在分词后移除原始的文本字段，如果需要保留可以去掉
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    # 如果您的机器内存足够，可以增加 num_proc 来并行处理，进一步提速
    # num_proc=os.cpu_count() 
)

# 为了方便 GRPOTrainer 使用，最好将原始的 'text' 列重命名为 'prompt'
# 如果 tokenize_function 中没有移除 'text' 列，可以这样做
# 如果已经移除了，可以从 input_ids 解码回来，但更简单的做法是在分词前就处理好列名
# 让我们采用一个更干净的流程：
# 1. 重命名列
dataset_with_prompt = dataset.rename_column('text', 'prompt')
# 2. 分词时基于新的 'prompt' 列
def tokenize_prompt(examples):
    return tokenizer(examples['prompt'], return_token_type_ids=False)

tokenized_dataset = dataset_with_prompt.map(
    tokenize_prompt,
    batched=True,
    # 我们保留 'prompt' 列，因为 GRPOTrainer 在计算奖励时可能需要原始文本
)


# =================================================================
# 4. 保存处理好的数据集
# =================================================================
print(f"分词完成，将数据集保存到 '{output_dir}'...")
# save_to_disk 会将数据集以高效的 Apache Arrow 格式保存
tokenized_dataset.save_to_disk(output_dir)

print("\n处理完成！")
print(f"现在您可以在训练脚本中使用 'datasets.load_from_disk(\"{output_dir}\")' 来加载这个预处理好的数据集。")
print("\n数据示例（第一条）：")
print(tokenized_dataset[0])