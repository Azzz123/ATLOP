import argparse
import os

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
# import wandb


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps, scaler):
        best_score = -1
        train_log_path = os.path.join(args.output_dir, "training_log.txt")
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs[0] / args.gradient_accumulation_steps
                # 使用 scaler 来缩放 loss 并进行反向传播
                scaler.scale(loss).backward()
                if step % args.gradient_accumulation_steps == 0:
                    # 在 scaler.step 之前，需要先 unscale 梯度，然后再裁剪
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    # 使用 scaler 来更新优化器和 scaler 自身
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                # 每隔 log_steps 步，就记录一次信息
                if num_steps % args.log_steps == 0:
                    # 获取当前的学习率
                    current_lr = scheduler.get_last_lr()[0]
                    # 格式化日志信息
                    log_message = f"Step: {num_steps}, Epoch: {epoch}, Loss: {loss.item():.6f}, LR: {current_lr:.8f}"
                    # 打印到终端
                    print(log_message)
                    # 追加写入到日志文件
                    with open(train_log_path, "a") as log_writer:
                        log_writer.write(log_message + "\n")
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, epoch, tag="dev")
                    # wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        pred = report(args, model, test_features)
                        with open("result.json", "w") as fh:
                            json.dump(pred, fh)
                        # 而是直接保存到output_dir 模型文件名可以包含更多信息，方便区分
                        model_path = os.path.join(args.output_dir, f"best_model_{args.transformer_type}.pt")
                        torch.save(model.state_dict(), model_path)
                        print(f"Best model saved to {model_path}")
        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scaler = GradScaler()
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps, scaler)


def evaluate(args, model, features, epoch, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, best_p, best_r, best_f1_ign = official_evaluate(ans, args.data_dir)
    else:
        best_f1, best_p, best_r, best_f1_ign = 0, 0, 0, 0
    output = {
        tag + "_Precision": best_p * 100,
        tag + "_Recall": best_r * 100,
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    # 新增：将评估结果写入日志文件 构建日志文件的完整路径
    eval_log_path = os.path.join(args.output_dir, "eval_results.txt")

    # 以追加模式打开文件，这样每次评估的结果都会被记录下来
    eval_log_path = os.path.join(args.output_dir, "eval_results.txt")
    with open(eval_log_path, "a") as writer:
        # 如果epoch是-1，说明是最终测试，不打印epoch号
        if epoch == -1:
            print(f"***** Final Eval results on {tag} *****", file=writer)
        else:
            print(f"***** Eval results on {tag} : epoch {epoch} *****", file=writer)
        for key, value in output.items():
            print(f"  {key} = {value}", file=writer)
        print("\n", file=writer)
    # ==================================================================== #

    return best_f1, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The directory to save output files (e.g., result.json and model checkpoints)")
    parser.add_argument("--log_steps", default=10, type=int,
                        help="Log loss and learning rate every N steps.")
    args = parser.parse_args()
    # 如果输出目录不存在，则创建它
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # wandb.init(project="DocRED")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    # 1. 先初始化模型结构
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(0)

    # 2. 根据模式（训练或测试）来决定加载哪些数据
    if args.load_path == "":  # Training Mode
        print("[INFO] Loading data for TRAINING...")
        train_file = os.path.join(args.data_dir, args.train_file)
        dev_file = os.path.join(args.data_dir, args.dev_file)
        test_file = os.path.join(args.data_dir, args.test_file)

        train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
        dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
        test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)

        print("[INFO] Starting training...")
        train(args, model, train_features, dev_features, test_features)


    else:  # Testing Mode

        print("[INFO] Loading TEST data for final evaluation...")

        test_file = os.path.join(args.data_dir, args.test_file)
        test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)

        print(f"--- Loading model from {args.load_path} for final evaluation ---")

        model.load_state_dict(torch.load(args.load_path))

        print("\n[INFO] Evaluating on TEST set...")

        test_score, test_output = evaluate(args, model, test_features, epoch=-1, tag="test")

        print("--- FINAL TEST SET PERFORMANCE ---")
        print(test_output)
        print("------------------------------------")
        print("\n[INFO] Generating submission file (result.json)...")

        pred = report(args, model, test_features)
        output_path = os.path.join(args.output_dir, "result.json")
        with open(output_path, "w") as fh:
            json.dump(pred, fh)

        print(f"Submission file saved to {output_path}")

if __name__ == "__main__":
    main()
