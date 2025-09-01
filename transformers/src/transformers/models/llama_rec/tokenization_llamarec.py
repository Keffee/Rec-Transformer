import os
import logging
from typing import List

from datasets import Dataset 
from dataclasses import dataclass, field

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO)

@dataclass
class MockTrainingArguments:
    output_dir: str = field(default="./rq_code_tokenizer", metadata={"help": "The output directory for the tokenizer."})
    max_length: int = field(default=512, metadata={"help": "The maximum sequence length."})

def create_rq_code_tokenizer(dataset: Dataset, training_args: MockTrainingArguments) -> PreTrainedTokenizerFast:
    """
    创建一个专门处理 RQ 编码序列的 Tokenizer。
    例如，处理形如 "<a_101> <b_54> <c_201>" 的字符串。

    Args:
        dataset (Dataset): Hugging Face 数据集，应包含一个名为 'text' 的列，
                           其内容为由空格分隔的 RQ 编码字符串。
        training_args (MockTrainingArguments): 包含输出路径和最大长度等配置。

    Returns:
        PreTrainedTokenizerFast: 初始化完成的、可用于 Transformers 的 Tokenizer。
    """
    
    # --- 1. 从数据集中提取词汇表 ---
    logging.info("Step 1: Extracting vocabulary from dataset...")
    all_rq_tokens = set()
    for row in dataset:
        if 'text' in row and isinstance(row['text'], str):
            # 按空格分割字符串，得到所有独立的 RQ 编码
            tokens = row['text'].split(' ')
            all_rq_tokens.update(tokens)
        else:
            logging.warning(f"Row skipped: 'text' not found or not a string in row: {row}")
    
    # 排序以保证每次生成的词汇表 ID 映射一致
    sorted_unique_tokens = sorted(list(all_rq_tokens))
    logging.info(f"Found {len(sorted_unique_tokens)} unique RQ code tokens.")

    # --- 2. 定义标准特殊 Token ---
    logging.info("Step 2: Defining special tokens...")
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    # --- 3. 构建词汇表 (token -> id 映射) ---
    logging.info("Step 3: Building vocabulary map...")
    vocab = {token: i for i, token in enumerate(special_tokens)}
    # 将 RQ 编码加入词汇表，ID 从特殊 Token 之后开始
    offset = len(special_tokens)
    for i, token in enumerate(sorted_unique_tokens):
        vocab[token] = i + offset
        
    logging.info(f"Total vocabulary size: {len(vocab)}")
    
    # --- 4. 初始化 Tokenizer 核心 ---
    logging.info("Step 4: Initializing WordLevel tokenizer...")
    # WordLevel 模型非常适合我们的场景，因为它将每个词汇表中的key视为一个独立的单元
    custom_tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))

    # --- 步骤 5. 设置预分词器 (关键修改) ---
    logging.info("Step 5: Setting up Whitespace pre-tokenizer...")
    # 我们不再需要复杂的 Regex。新的规则非常简单：用空格分割。
    custom_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # --- 6. 设置 Post-Processor (自动添加 BOS/EOS) ---
    logging.info("Step 6: Setting up post-processor to add BOS/EOS tokens...")
    # 重新启用这个功能，因为它对序列模型很有用
    # 话虽如此，我相信对一个语言模型来说EOS很有用，但是对于一个序列推荐来说，什么时候是停止？什么时候是结束？因此我认为不应该添加eos token
    custom_tokenizer.post_processor = TemplateProcessing(
        # single="[BOS] $A [EOS]",
        single= "$A",
        special_tokens=[
            ("[BOS]", vocab["[BOS]"]),
            ("[EOS]", vocab["[EOS]"]),
        ],
    )

    # --- 7. 配置 Padding ---
    logging.info("Step 7: Enabling padding on the core tokenizer...")
    custom_tokenizer.enable_padding(
        direction="left",
        pad_id=vocab["[PAD]"],
        pad_token="[PAD]",
        length=training_args.max_length,
    )
    
    # --- 8. 保存并包装 ---
    logging.info("Step 8: Saving tokenizer and wrapping with PreTrainedTokenizerFast...")
    os.makedirs(training_args.output_dir, exist_ok=True)
    tokenizer_file_path = os.path.join(training_args.output_dir, "tokenizer.json")
    custom_tokenizer.save(tokenizer_file_path)
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file_path,
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
    # 1. 准备一个模拟的数据集，格式与 final_rq_sequences.json 一致
    dummy_data = {
        "text": [
            "<a_101> <b_42> <c_999> <a_5000>",
            "<b_42> <c_888>",
            "<a_101> <c_1> <c_999> <d_2048>"
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
    tokenizer = create_rq_code_tokenizer(dataset=dummy_dataset, training_args=args)

    # 4. 测试 Tokenizer
    print("\n--- Testing the Tokenizer ---")
    
    # 测试用例 1: 正常的序列
    sequence_str = "<a_101> <b_42> <c_999>"
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
    # 解码后由空格连接是预期行为
    assert decoded == "<a_101> <b_42> <c_999>" 
    
    print("-" * 20)

    # 测试用例 2: 包含未知 token (UNK token 测试)
    unknown_sequence_str = "<b_42> <z_9999> <a_101>"
    print(f"Sequence with unknown token: '{unknown_sequence_str}'")
    encoded_unknown = tokenizer.encode(unknown_sequence_str)
    print(f"Encoded IDs with UNK: {encoded_unknown}")
    
    unk_id = tokenizer.unk_token_id
    print(f"UNK ID: {unk_id}")
    # 验证第二个 token 是否是 UNK ID
    assert encoded_unknown[2] == unk_id
    
    decoded_unknown = tokenizer.decode(encoded_unknown, skip_special_tokens=False)
    print(f"Decoded string with UNK: '{decoded_unknown}'")
    assert decoded_unknown == "[BOS] <b_42> [UNK] <a_101> [EOS]"
    
    print("\n✅ All tests passed!")