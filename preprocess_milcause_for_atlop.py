import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

# 全局变量，将在主函数中初始化
TOKENIZER = None


def find_trigger_positions(sentence_tokens, trigger):
    """在分词后的句子中找到触发词的精确token位置。"""
    trigger_tokens = TOKENIZER.tokenize(trigger)
    if not trigger_tokens: return [-1, -1]
    for i in range(len(sentence_tokens) - len(trigger_tokens) + 1):
        if sentence_tokens[i:i + len(trigger_tokens)] == trigger_tokens:
            return [i, i + len(trigger_tokens)]
    return [-1, -1]


def process_file(input_path, output_path):
    """处理单个文件的核心逻辑"""
    with open(input_path, 'r', encoding='utf-8') as f:
        milcause_data = json.load(f)

    atlop_data = []
    for doc_id_counter, doc in enumerate(tqdm(milcause_data, desc=f"Processing {os.path.basename(input_path)}")):
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

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(atlop_data, f, ensure_ascii=False)
    print(f"Successfully converted {input_path} to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess MilCause data for ATLOP framework.")
    # ==================================================================== #
    # 核心修改：使用更清晰、更独立的参数
    # ==================================================================== #
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing clean train.json, dev.json, test.json")
    parser.add_argument("--output_data_dir", type=str, required=True,
                        help="Directory to save the processed ATLOP-formatted data files.")
    parser.add_argument("--output_meta_path", type=str, default='./meta/rel2id.json',
                        help="Path to save the rel2id.json file.")
    parser.add_argument("--model_path", type=str, default='bert-base-cased',
                        help="Path to pre-trained model for tokenizer.")
    args = parser.parse_args()

    # 初始化全局分词器
    TOKENIZER = AutoTokenizer.from_pretrained(args.model_path)

    # 确保输出目录存在
    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    meta_dir = os.path.dirname(args.output_meta_path)
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    # 定义文件名映射
    file_name_map = {
        'train.json': 'train.json',
        'dev.json': 'dev.json',
        'test.json': 'test.json'
    }

    # 循环处理文件
    for input_name, output_name in file_name_map.items():
        input_file = os.path.join(args.input_dir, input_name)
        output_file = os.path.join(args.output_data_dir, output_name)
        if os.path.exists(input_file):
            process_file(input_file, output_file)
        else:
            print(f"Warning: {input_file} not found. Skipping.")

    # 创建关系映射文件
    rel2id = {"NA": 0, "Causal": 1}
    with open(args.output_meta_path, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f)
    print(f"rel2id.json saved to {args.output_meta_path}")