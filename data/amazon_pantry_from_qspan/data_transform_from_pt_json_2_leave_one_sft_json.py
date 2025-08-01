import json

# --- 1. 设置输入和输出文件名 ---
input_filename = 'llama_pt_format.json'  # <--- 请将这里替换成你的输入文件名
output_filename = 'llama_leave_one_sft_format.json' # <--- 这是转换后输出的文件名

# --- 2. 读取和处理数据 ---
transformed_data = []

try:
    with open(input_filename, 'r', encoding='utf-8') as f:
        print(f"正在读取文件: {input_filename}")
        original_data = json.load(f)

    print("开始转换数据格式...")
    # 遍历原始数据中的每一条记录
    for record in original_data:
        text = record.get("text", "") # 使用 .get() 安全地获取 "text" 字段
        
        # 使用逗号分割字符串，并移除每个数字周围可能存在的空格
        # 这个列表推导式会处理像 "1, 2" 和 "1,2" 这样的情况
        items = [item.strip() for item in text.split(',') if item.strip()]
        
        # 确保序列中至少有两个项目（一个用于 instruction，一个用于 output）
        if len(items) >= 2:
            # 最后一个元素是 output
            output = items[-1]
            
            # 除了最后一个元素之外的所有元素都是 instruction
            instruction_list = items[:-1]
            instruction = ", ".join(instruction_list)
            
            # 创建新的格式化记录
            new_record = {
                "instruction": instruction,
                "output": output
            }
            transformed_data.append(new_record)
        else:
            # 如果序列太短（少于2个元素），则打印警告并跳过该记录
            print(f"警告: 记录 '{text}' 因长度不足而被跳过。")

except FileNotFoundError:
    print(f"错误: 输入文件 '{input_filename}' 未找到。请检查文件名和路径是否正确。")
    exit() # 如果文件不存在，则退出脚本
except json.JSONDecodeError:
    print(f"错误: 文件 '{input_filename}' 不是有效的JSON格式。")
    exit()

# --- 3. 写入新的JSON文件 ---
if transformed_data:
    with open(output_filename, 'w', encoding='utf-8') as f:
        # 使用 indent=2 参数使输出的JSON文件格式优美，易于阅读
        # ensure_ascii=False 确保中文字符（如果有）能被正确写入
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    
    print("-" * 20)
    print(f"✅ 转换成功!")
    print(f"总共处理了 {len(transformed_data)} 条记录。")
    print(f"结果已保存到: {output_filename}")
else:
    print("没有生成任何数据，请检查输入文件内容和格式。")