# train.py (修改版)

import os
from typing import Dict, List
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from modeling_llama_rec import LlamaForRec, LlamaConfig  # 导入你的自定义模型和配置类

# --- 1. 注册你的自定义模型 (保持不变) ---
print("正在注册自定义的 LlamaForRec 模型...")
MODEL_TYPE = "llama-rec"
AutoConfig.register(MODEL_TYPE, LlamaConfig)
AutoModelForCausalLM.register(LlamaConfig, LlamaForRec)


def preprocess_function(examples: Dict[str, List[str]], tokenizer, max_length: int) -> Dict[str, List[int]]:
    """(保持不变)"""
    all_input_ids = []
    for text in examples["text"]:
        item_ids = [int(t.strip()) for t in text.split(',') if t.strip()]
        
        # 截断或填充
        if len(item_ids) > max_length:
            item_ids = item_ids[:max_length]
        else:
            item_ids = item_ids + [tokenizer.pad_token_id] * (max_length - len(item_ids))
        
        all_input_ids.append(item_ids)

    return {
        "input_ids": all_input_ids,
        "labels": all_input_ids.copy(),
    }


def main():
    # --- 2. 定义模型和数据路径 (保持不变) ---
    dataset_path = "output.json"
    output_dir = "./llama-rec-scratch-checkpoints" # 新的输出目录
    max_seq_length = 128
    
    # 假设你的 item ID 范围是 0 到 10000
    # 这个值应该大于你数据集中出现过的最大 item ID
    VOCAB_SIZE = 10001 

    # --- 3. [修改] 创建一个简单的 Tokenizer ---
    # 因为我们不再依赖预训练模型，所以需要自己构建一个 Tokenizer。
    # 对于 item ID，最简单的方式是创建一个能识别所有数字 ID 的 "WordLevel" Tokenizer。
    print("正在从头创建 Tokenizer...")
    
    # 定义特殊 token
    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    
    # 创建一个空的 WordLevel tokenizer
    # 我们将把每个 item ID 作为一个独立的 "word"
    # Whitespace pre-tokenizer 在这里效果很好，因为它会按空格分割
    # 但由于我们的输入是数字ID，所以更简单的方法是手动构建词汇表
    vocab = {str(i): i for i in range(VOCAB_SIZE)}
    # 将特殊 token 添加到词汇表的末尾
    for i, token in enumerate(special_tokens):
        vocab[token] = VOCAB_SIZE + i
        
    custom_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    custom_tokenizer.pre_tokenizer = Whitespace() # 简单起见，用空格分割

    # 包装成 transformers 的 Tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=custom_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    
    # 确保 pad_token_id 设置正确
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = vocab["[PAD]"]


    # --- 4. [修改] 加载和预处理数据集 (与之前类似，但使用新tokenizer) ---
    # 这部分逻辑基本不变，只是传入的 tokenizer 是我们自己创建的
    print(f"正在从 {dataset_path} 加载数据集...")
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    print("正在预处理数据集...")
    from functools import partial
    process_func_with_args = partial(preprocess_function, tokenizer=tokenizer, max_length=max_seq_length)
    
    processed_dataset = raw_dataset.map(
        process_func_with_args,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=4,
    )
    
    print("数据集示例:")
    print(processed_dataset[0])

    # --- 5. [核心修改] 从头创建模型 ---
    print("正在从头创建 LlamaForRec 模型...")
    
    # (A) 定义一个小的模型配置
    config = LlamaConfig(
        model_type=MODEL_TYPE, # 告诉 transformers 这是我们的自定义模型
        vocab_size=len(tokenizer), # 词汇表大小 = item数量 + 特殊token数量
        hidden_size=256,         # 大幅减小！
        intermediate_size=512,   # 大幅减小！
        num_hidden_layers=4,       # 大幅减小！
        num_attention_heads=4,     # 大幅减小！
        max_position_embeddings=max_seq_length,
        rms_norm_eps=1e-6,
        use_cache=False,           # 训练时通常不使用 cache
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # 其他 Llama 参数可以保持默认或根据需要调整
    )

    # (B) 直接用配置实例化模型，而不是 from_pretrained
    model = LlamaForRec(config)

    print("模型结构:")
    print(model)
    print(f"模型总参数量: {model.num_parameters() / 1e6:.2f} M")


    # --- 6. 定义训练参数 (保持不变) ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32, # 可以适当增大了，因为模型变小了
        gradient_accumulation_steps=1,
        learning_rate=5e-4, # 从头训练，学习率可以设置得大一些
        num_train_epochs=20, # 从头训练，需要更多 epoch
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch", # 按 epoch 保存
        save_total_limit=3,
        fp16=True,
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    # --- 7. 定义数据整理器 (保持不变) ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --- 8. 实例化 Trainer (保持不变) ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 9. 开始训练！ (保持不变) ---
    print("开始从头训练...")
    trainer.train()

    # --- 10. 保存最终模型 (保持不变) ---
    final_model_path = f"{output_dir}/final"
    print(f"训练完成，正在保存最终模型到 {final_model_path}")
    trainer.save_model(final_model_path)
    # 也要保存 tokenizer，以便后续加载
    tokenizer.save_pretrained(final_model_path)
    print("所有操作完成！")


if __name__ == "__main__":
    main()