import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer


def find_trigger_positions(sentence_tokens, trigger):
    """在分词后的句子中找到触发词的精确token位置。"""
    trigger_tokens = TOKENIZER.tokenize(trigger)
    if not trigger_tokens: return [-1, -1]
    for i in range(len(sentence_tokens) - len(trigger_tokens) + 1):
        if sentence_tokens[i:i + len(trigger_tokens)] == trigger_tokens:
            return [i, i + len(trigger_tokens)]
    return [-1, -1]


def process_milcause_for_atlop(input_file, output_dir, dataset_name):
    """主处理函数：将MilCause格式的数据转换为ATLOP所需的格式。"""
    dataset_path = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    with open(input_file, 'r', encoding='utf-8') as f:
        # ==================================================================== #
        # 核心修正 1: 采纳你的数据加载方式，读取整个JSON数组
        milcause_data = json.load(f)
        # ==================================================================== #

    atlop_data = []
    for doc_id_counter, doc in enumerate(tqdm(milcause_data, desc=f"Processing {os.path.basename(input_file)}")):
        inp = json.loads(doc['input'])
        text = inp['text']
        candidate_pairs = inp['candidate_pairs']

        output_str = doc.get('output', '[]')
        try:
            gold_relations_list = json.loads(output_str) if isinstance(output_str, str) else output_str
        except json.JSONDecodeError:
            gold_relations_list = []

        gold_pairs = set((rel['cause']['event_id'], rel['effect']['event_id']) for rel in gold_relations_list)

        sentences_text = [s.strip() for s in
                          text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n') if s.strip()]
        sents_tokens = [TOKENIZER.tokenize(s) for s in sentences_text]

        vertex_set = []
        event_id_to_info = {}
        for pair in candidate_pairs:
            for event_key in ['event_1', 'event_2']:
                event = pair[event_key]
                if event['event_id'] not in event_id_to_info:
                    event_id_to_info[event['event_id']] = event

        event_id_to_vertex_idx = {}
        for event_id, event_info in sorted(event_id_to_info.items()):
            trigger = event_info['trigger']
            found = False
            for i, sent_tokens in enumerate(sents_tokens):
                pos = find_trigger_positions(sent_tokens, trigger)
                if pos[0] != -1:
                    vertex = {"name": trigger, "sent_id": i, "pos": pos, "type": event_info['event_type']}
                    vertex_set.append([vertex])
                    event_id_to_vertex_idx[event_id] = len(vertex_set) - 1
                    found = True
                    break
            if not found:
                pass

        labels = []
        for h_id, t_id in gold_pairs:
            if h_id in event_id_to_vertex_idx and t_id in event_id_to_vertex_idx:
                h_idx = event_id_to_vertex_idx[h_id]
                t_idx = event_id_to_vertex_idx[t_id]
                labels.append({"h": h_idx, "t": t_idx, "r": "Causal", "evidence": []})

        atlop_doc = {"title": f"doc_{doc_id_counter}", "sents": sents_tokens, "vertexSet": vertex_set, "labels": labels}
        atlop_data.append(atlop_doc)

    # ==================================================================== #
    # 核心修正 2: 采纳你的文件名映射逻辑，以适配ATLOP主代码
    file_name_map = {
        'train.json': 'train_annotated.json',
        'dev.json': 'dev.json',
        'test.json': 'test.json'
    }
    original_basename = os.path.basename(input_file)
    output_basename = file_name_map.get(original_basename, original_basename)
    output_file_path = os.path.join(dataset_path, output_basename)
    # ==================================================================== #

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(atlop_data, f, ensure_ascii=False)
    print(f"Successfully converted {input_file} to {output_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing MilCause train.json, dev.json, test.json")
    parser.add_argument("--output_root", type=str, default='./', help="Root directory of ATLOP project")
    parser.add_argument("--dataset_name", type=str, default='milcause', help="Name for the converted dataset")
    # ==================================================================== #
    # 核心修正 3: 新增命令行参数以支持本地模型
    parser.add_argument("--model_path", type=str, default='bert-base-cased',
                        help="Path to pre-trained model or model identifier from huggingface.co/models")
    # ==================================================================== #
    args = parser.parse_args()

    # 使用全局变量TOKENIZER，在主函数中进行初始化
    TOKENIZER = AutoTokenizer.from_pretrained(args.model_path)

    # 创建目录结构
    if not os.path.exists(os.path.join(args.output_root, 'dataset')):
        os.makedirs(os.path.join(args.output_root, 'dataset'))
    if not os.path.exists(os.path.join(args.output_root, 'meta')):
        os.makedirs(os.path.join(args.output_root, 'meta'))

    # 处理数据文件
    for split in ['train', 'dev', 'test']:
        input_file = os.path.join(args.input_dir, f"{split}.json")
        if os.path.exists(input_file):
            process_milcause_for_atlop(input_file, os.path.join(args.output_root, 'dataset'), args.dataset_name)
        else:
            print(f"Warning: {input_file} not found. Skipping.")

    # 创建关系映射文件
    rel2id = {"NA": 0, "Causal": 1}
    with open(os.path.join(args.output_root, 'meta', 'rel2id.json'), 'w', encoding='utf-8') as f:
        json.dump(rel2id, f)
    print(f"rel2id.json created in {os.path.join(args.output_root, 'meta')}")