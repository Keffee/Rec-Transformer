from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import json

# 方法1：使用tokenizers库直接加载tokenizer.json
def test_with_tokenizers_library():
    # 加载保存好的tokenizer
    tokenizer = Tokenizer.from_file("my_tokenizer.json")
    
    # 编码测试
    encoded = tokenizer.encode("<|begin_of_text|> 1 2 3 114514 0 1 <|end_of_text|>")
    print("使用tokenizers库:")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")

# 方法2：使用transformers库加载完整的tokenizer配置
def test_with_transformers_library():
    # 从保存的文件加载tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="my_tokenizer.json",
        tokenizer_config_file="tokenizer_config.json",
        special_tokens_map_file="special_tokens_map.json"
    )
    
    # 编码测试
    encoded = tokenizer("<|begin_of_text|> 1 2 3 114514 0 1 <|end_of_text|>")
    print("\n使用transformers库:")
    print(f"Input IDs: {encoded['input_ids']}")
    
    # 解码测试
    decoded = tokenizer.decode(encoded["input_ids"])
    print(f"解码结果: {decoded}")
    
    # 检查特殊token
    print(f"BOS token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")
    print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")

# 创建一个special_tokens_map.json文件(如果还没有的话)
def create_special_tokens_map():
    special_tokens_map = {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "unk_token": "[UNK]",
        "pad_token": "0"  # 使用"0"作为padding token
    }
    
    with open("special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)
    
    print("已创建special_tokens_map.json文件")

# 运行测试
if __name__ == "__main__":
    # 如果需要，创建special_tokens_map.json
    create_special_tokens_map()
    
    # 使用两种方法测试tokenizer
    test_with_tokenizers_library()
    test_with_transformers_library()
