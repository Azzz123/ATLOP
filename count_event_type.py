import json
from collections import Counter

# 你的所有数据文件
data_files = ['data/MilCause/train.json', 'data/MilCause/dev.json', 'data/MilCause/test.json']

all_event_types = set()

for file_path in data_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for doc in data:
            inp = json.loads(doc['input'])
            for pair in inp['candidate_pairs']:
                all_event_types.add(pair['event_1']['event_type'])
                all_event_types.add(pair['event_2']['event_type'])

print(f"Found {len(all_event_types)} unique event types:")
for event_type in sorted(list(all_event_types)):
    print(f"- {event_type}")

print(f"\n=> You should set --num_labels to {len(all_event_types)}")