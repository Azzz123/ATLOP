import json
import random
import os
import argparse
from tqdm import tqdm


def final_data_splitter(input_dir, output_dir, dev_ratio=0.1, seed=42):
    """
    Reads original train.json and test.json.
    1. Ensures no documents from test.json exist in train.json (data cleaning).
    2. Splits the cleaned train data into a new train set and a dev set.
    3. Saves the new train, new dev, and original test sets to the output directory.
    This guarantees three fully independent datasets.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- 1. 定义并加载原始文件 ---
    original_train_path = os.path.join(input_dir, 'train.json')
    original_test_path = os.path.join(input_dir, 'test.json')

    print(f"Loading original train data from: {original_train_path}")
    with open(original_train_path, 'r', encoding='utf-8') as f:
        original_train_data = json.load(f)
    print(f"Loaded {len(original_train_data)} samples from original train set.")

    print(f"Loading original test data from: {original_test_path}")
    with open(original_test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} samples from original test set.")

    # --- 2. 关键步骤：清洗训练集，移除所有在测试集中出现过的文档 ---
    print("\n--- Starting Data Cleaning: Ensuring no leakage from test set ---")
    test_texts = {json.loads(sample['input'])['text'] for sample in test_data}

    clean_train_pool = []
    for sample in tqdm(original_train_data, desc="Cleaning train data"):
        doc_text = json.loads(sample['input'])['text']
        if doc_text not in test_texts:
            clean_train_pool.append(sample)

    removed_count = len(original_train_data) - len(clean_train_pool)
    if removed_count > 0:
        print(
            f"Cleaning complete. Removed {removed_count} samples from train set because they appeared in the test set.")
    else:
        print("Cleaning complete. No overlapping documents found between train and test sets. Excellent!")

    # --- 3. 在清洗后的训练数据池中，划分出新的训练集和验证集 ---
    print(f"\n--- Splitting the clean train pool of {len(clean_train_pool)} samples ---")
    random.seed(seed)
    random.shuffle(clean_train_pool)

    # 计算分割点
    split_index = int(len(clean_train_pool) * (1 - dev_ratio))

    new_train_data = clean_train_pool[:split_index]
    new_dev_data = clean_train_pool[split_index:]

    # --- 4. 保存三个最终的、干净的数据集 ---
    final_train_path = os.path.join(output_dir, 'train.json')
    final_dev_path = os.path.join(output_dir, 'dev.json')
    final_test_path = os.path.join(output_dir, 'test.json')

    with open(final_train_path, 'w', encoding='utf-8') as f:
        json.dump(new_train_data, f, ensure_ascii=False, indent=4)
    print(f"Saved final train set with {len(new_train_data)} samples to {final_train_path}")

    with open(final_dev_path, 'w', encoding='utf-8') as f:
        json.dump(new_dev_data, f, ensure_ascii=False, indent=4)
    print(f"Saved final dev set with {len(new_dev_data)} samples to {final_dev_path}")

    with open(final_test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    print(f"Saved final test set with {len(test_data)} samples to {final_test_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cleanly split train/test data into final train/dev/test sets.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the original train.json and test.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the new, clean split files.")
    parser.add_argument("--dev_ratio", type=float, default=0.1,
                        help="Ratio of the original train set to be used for the dev set (e.g., 0.1 for 10%).")
    args = parser.parse_args()

    final_data_splitter(args.input_dir, args.output_dir, args.dev_ratio)