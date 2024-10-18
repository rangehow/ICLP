from datasets import load_dataset
import json
import math

# 流式加载 fineweb-edu 数据集
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

# 初始化变量
total_length = 0
total_count = 100000  # 总共要处理的数据条数
batch_size = 1000  # 每个文件存储的数据条数
total_files = math.ceil(total_count / batch_size)  # 总文件数

for batch_num in range(total_files):
    data_list = []
    batch_length = 0
    
    # 文件名格式："{batch_num:02d} out of {total_files}.jsonl"
    output_file = f"{batch_num:02d} out of {total_files}.jsonl"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(batch_size):
            try:
                example = next(iter(dataset))
            except StopIteration:
                break  # 如果数据集已经遍历完，就退出循环
            
            # 将每条数据转为 JSON 格式，并写入一行
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            # 假设每条数据的文本内容在 "text" 字段，统计字符长度
            text = example.get('text', '')
            batch_length += len(text)
            total_length += len(text)
    
    # 计算并打印当前批次的平均长度
    batch_average_length = batch_length / batch_size if batch_size > 0 else 0
    print(f"批次 {batch_num + 1} 数据已保存到 {output_file}")
    print(f"这 {batch_size} 条数据的平均长度为 {batch_average_length:.2f} 个字符")

# 计算总体平均长度
total_average_length = total_length / total_count
print(f"\n总共处理了 {total_count} 条数据")
print(f"所有数据的平均长度为 {total_average_length:.2f} 个字符")