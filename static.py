import json
from collections import defaultdict

def count_numbers(data, counts):
    if isinstance(data, dict):
        for value in data.values():
            count_numbers(value, counts)
    elif isinstance(data, list):
        for item in data:
            count_numbers(item, counts)
    elif isinstance(data, (int, float)):
        counts[data] += 1

def main(json_file):
    # 读取JSON文件
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # 使用defaultdict来统计数字出现的次数
    label_counts = defaultdict(int)
    instance_counts = defaultdict(int)
    
    # 统计labels中的数字
    if 'labels' in data:
        count_numbers(data['labels'], label_counts)
    
    # 统计instances中的数字
    if 'instances' in data:
        count_numbers(data['instances'], instance_counts)
    
    # 转换为普通字典并按值和键排序
    label_counts = dict(sorted(label_counts.items(), key=lambda item: (item[1], item[0])))
    instance_counts = dict(sorted(instance_counts.items(), key=lambda item: (item[1], item[0])))
    
    # 计算总数
    total_label_count = sum(label_counts.values())
    total_instance_count = sum(instance_counts.values())
    
    # 输出结果
    print("Label Counts:", label_counts)
    print("Total Label Count:", total_label_count)
    print("Instance Counts:", instance_counts)
    print("Total Instance Count:", total_instance_count)

if __name__ == "__main__":
    json_file = '/home/hiro/3d_tooth_seg/data/json/01K17AN8/01K17AN8_lower.json'  # 将此处替换为你的JSON文件路径
    main(json_file)