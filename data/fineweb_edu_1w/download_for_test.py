from datasets import load_dataset
import json

# 流式加载 fineweb-edu 数据集
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

# 初始化保存列表和长度统计
data_list = []
total_length = 0

# 打开文件准备写入
output_file = "fineweb_edu_500.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    # 迭代获取前500条数据
    for i, example in enumerate(dataset):
        if i >= 10000:
            break
        
        # 将每条数据转为 JSON 格式，并写入一行
        f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        # 假设每条数据的文本内容在 "text" 字段，统计字符长度
        text = example.get('text', '')
        total_length += len(text)

# 计算平均长度
average_length = total_length / 500

print(f"前500条数据已保存到 {output_file}")
print(f"这500条数据的平均长度为 {average_length:.2f} 个字符")
