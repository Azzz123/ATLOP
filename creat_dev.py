import json
import random
import os
import argparse


def split_dataset_correctly(input_file, output_dir, train_ratio=0.9):
    """
    Correctly reads a standard JSON array file, shuffles the objects
    within the array, and splits it into a new train.json and dev.json file,
    both as valid JSON arrays.
    """
    print(f"Reading JSON array data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        # 正确的读取方式：将整个文件作为一个JSON对象加载
        data_list = json.load(f)

    print(f"Total {len(data_list)} samples found.")

    # 在对象层面进行随机打乱，而不是在文本行层面
    random.seed(42)  # 使用固定的随机种子，保证每次分割结果一致
    random.shuffle(data_list)
    print("Data shuffled correctly at the object level.")

    # 计算分割点
    split_point = int(len(data_list) * train_ratio)

    train_data = data_list[:split_point]
    dev_data = data_list[split_point:]

    # 定义输出路径
    train_output_path = os.path.join(output_dir, 'train.json')
    dev_output_path = os.path.join(output_dir, 'dev.json')

    # 正确的写入方式：将Python列表写回为格式化的JSON数组
    with open(train_output_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保证中文不被转义
        # indent=4 使得JSON文件可读性更强（可选，但推荐）
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    print(f"New training set with {len(train_data)} samples saved to {train_output_path}")

    with open(dev_output_path, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)
    print(f"New development set with {len(dev_data)} samples saved to {dev_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the original large train.json file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the new train.json and dev.json")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    split_dataset_correctly(args.input_file, args.output_dir)