from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

# 创建词汇表
vocab = {"1": 0, "2": 1, "3": 2, "10": 9, "[UNK]": 3}

# 创建tokenizer
tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))

# 可以不设置pre_tokenizer，或者用一个不会打断你格式的
# 这里用WhitespaceSplit只是示例，实际上你可能不需要任何pre_tokenizer
tokenizer.pre_tokenizer = WhitespaceSplit()

# 编码测试
encoded = tokenizer.encode("1 2 3 10 2 1")
print(encoded.tokens)  # 应该输出 ["1#", "2#", "3#", "1#"]
print(encoded.ids)     # 应该输出 [0, 1, 2, 0]

tokenizer.save("my_tokenizer.json")


