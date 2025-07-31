import os
import logging
from typing import List

# 我们需要 Dataset 来构建词汇表
from datasets import Dataset 
# 为了方便，我们模拟一个 TrainingArguments 类来传递配置
from dataclasses import dataclass, field

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, normalizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO)

# 使用 dataclass 模拟 Hugging Face 的 TrainingArguments，以便传递配置
@dataclass
class MockTrainingArguments:
    output_dir: str = field(default="./hybrid_item_tokenizer", metadata={"help": "The output directory for the tokenizer."})
    max_length: int = field(default=512, metadata={"help": "The maximum sequence length."})


def create_hybrid_item_tokenizer(dataset: Dataset, training_args: MockTrainingArguments) -> PreTrainedTokenizerFast:
    """
    创建一个混合型 Tokenizer，它具备以下特点：
    1. 词汇表由数据集驱动 (Data-Driven)。
    2. 使用强大的 Regex 预分词器处理 "id1, id2, id3" 格式的输入字符串。
    3. 使用标准的特殊 Token ([UNK], [PAD], [BOS], [EOS]) 以保证健壮性。

    Args:
        dataset (Dataset): Hugging Face 数据集，应包含一个名为 'item_sequence' 的列，
                           其内容为 item ID 的列表 (e.g., [101, 42, 999])。
        training_args (MockTrainingArguments): 包含输出路径和最大长度等配置。

    Returns:
        PreTrainedTokenizerFast: 初始化完成的、可用于 Transformers 的 Tokenizer。
    """
    
    # --- 1. 从数据集中提取词汇表 (Data-Driven Approach) ---
    logging.info("Step 1: Extracting vocabulary from dataset...")
    all_item_ids = set()
    for row in dataset:
        if 'item_sequence' in row and isinstance(row['item_sequence'], list):
            all_item_ids.update(row['item_sequence'])
        else:
            logging.warning(f"Row skipped: 'item_sequence' not found or not a list in row: {row}")
    
    # 排序以保证每次生成的词汇表 ID 映射一致
    sorted_unique_items = sorted(list(all_item_ids))
    logging.info(f"Found {len(sorted_unique_items)} unique item IDs.")

    # --- 2. 定义标准特殊 Token ---
    logging.info("Step 2: Defining special tokens...")
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    # --- 3. 构建词汇表 (token -> id 映射) ---
    logging.info("Step 3: Building vocabulary map...")
    vocab = {}
    # 首先加入特殊 Token
    for i, token in enumerate(special_tokens):
        vocab[token] = i
    
    # 接着加入 Item ID Token，ID 从特殊 Token 之后开始
    offset = len(special_tokens)
    # Token 本身就是 Item ID 的字符串形式，例如 "101"
    for i, item_id in enumerate(sorted_unique_items):
        vocab[str(item_id)] = i + offset
        
    logging.info(f"Total vocabulary size: {len(vocab)}")
    
    # --- 4. 初始化 Tokenizer 核心 ---
    logging.info("Step 4: Initializing WordLevel tokenizer...")
    custom_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))

    # --- 步骤 5. 设置 Regex 预分词器 (关键修改) ---
    logging.info("Step 5: Setting up Regex pre-tokenizer to ONLY capture digits...")
    # 只保留匹配到的数字，丢弃所有其他字符
    # custom_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    #     pre_tokenizers.Split(pattern=Regex(r"\d+"), behavior="isolated"),
    #     pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False) # 这是一个技巧，用于过滤掉非数字的 token 块
    # ])

    # 1. 定义一个 Normalizer，它会在分词前运行
    #    这个 Normalizer 会完成所有的数据清洗工作
    custom_tokenizer.normalizer = normalizers.Sequence([
        # 步骤 A: 将所有不是数字(\d)也不是空白符(\s)的字符，替换成一个空格。
        # e.g., '127, 385' -> '127  385'
        normalizers.Replace(Regex(r"[^\d\s]+"), " "),
        # 步骤 B: 将多个连续的空格合并成一个
        normalizers.Replace(Regex(r"\s+"), " "),
        # 步骤 C: 去掉字符串首尾的空格
        normalizers.Strip(),
    ])

    custom_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # --- 6. 设置 Post-Processor (自动添加 BOS/EOS) ---
    logging.info("Step 6: Setting up post-processor to add BOS/EOS tokens...")
    # 下面这段暂时删除，将来想要添加什么新token在这里加就行，
    # custom_tokenizer.post_processor = TemplateProcessing(
    #     single="[BOS] $A [EOS]",
    #     # single="$A",
    #     special_tokens=[
    #         ("[BOS]", vocab["[BOS]"]),
    #         ("[EOS]", vocab["[EOS]"]),
    #     ],
    # )

    # --- [新增] 步骤 6.5: 在核心 Tokenizer 上配置 Padding ---
    logging.info("Step 6.5: Enabling padding on the core tokenizer...")
    custom_tokenizer.enable_padding(
        direction="left", # 与 transformers 的 'padding_side' 保持一致
        pad_id=vocab["[PAD]"],
        pad_token="[PAD]",
        length=training_args.max_length, # 可选：设置一个默认的 padding 长度
        # strategy=PaddingStrategy.MAX_LENGTH # 可选：设置一个默认的 padding 策略
    )
    
    # --- 7. 保存并包装 ---
    logging.info("Step 7: Saving tokenizer and wrapping with PreTrainedTokenizerFast...")
    os.makedirs(training_args.output_dir, exist_ok=True)
    tokenizer_file_path = os.path.join(training_args.output_dir, "tokenizer.json")
    custom_tokenizer.save(tokenizer_file_path)
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file_path,
        # 必须明确指定这些特殊 Token，以便 Transformers 正确使用它们
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=training_args.max_length,
        padding_side="left",
    )

    logging.info("Tokenizer creation complete!")
    return hf_tokenizer


# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 准备一个模拟的数据集
    # 注意 item_sequence 是一个整数列表 List[int]
    dummy_data = {
        "user_id": [1, 2, 3],
        "item_sequence": [
            [101, 42, 999, 5000],
            [42, 888],
            [101, 1, 999, 2048]
        ]
    }
    dummy_dataset = Dataset.from_dict(dummy_data)
    print("--- Dummy Dataset ---")
    print(dummy_dataset)
    print(dummy_dataset[0])
    print("-" * 20 + "\n")

    # 2. 准备配置参数
    args = MockTrainingArguments()

    # 3. 创建 Tokenizer
    tokenizer = create_hybrid_item_tokenizer(dataset=dummy_dataset, training_args=args)

    # 4. 测试 Tokenizer
    print("\n--- Testing the Tokenizer ---")
    
    # 测试用例 1: 正常的序列
    sequence_str = "101, 42, 999"
    print(f"Original sequence string: '{sequence_str}'")
    encoded = tokenizer.encode(sequence_str)
    print(f"Encoded IDs: {encoded}")
    
    # 验证 BOS/EOS token
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    print(f"BOS ID: {bos_id}, EOS ID: {eos_id}")
    assert encoded[0] == bos_id and encoded[-1] == eos_id
    
    # 解码
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"Decoded string (skipped special tokens): '{decoded}'")
    # 注意：默认解码器用空格连接 token，这是预期的行为
    assert decoded == "101 42 999" 
    
    print("-" * 20)

    # 测试用例 2: 包含未在数据集中出现过的 item (UNK token 测试)
    # 9999 没有在我们的 dummy_dataset 中出现过
    unknown_sequence_str = "42, 9999, 101"
    print(f"Sequence with unknown item: '{unknown_sequence_str}'")
    encoded_unknown = tokenizer.encode(unknown_sequence_str)
    print(f"Encoded IDs with UNK: {encoded_unknown}")
    
    unk_id = tokenizer.unk_token_id
    print(f"UNK ID: {unk_id}")
    # 验证第二个 token 是否是 UNK ID
    assert encoded_unknown[2] == unk_id
    
    decoded_unknown = tokenizer.decode(encoded_unknown)
    print(f"Decoded string with UNK: '{decoded_unknown}'")
    assert decoded_unknown == "[BOS] 42 [UNK] 101 [EOS]"
    
    print("\n✅ All tests passed!")