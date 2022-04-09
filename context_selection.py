import os
import datasets
import transformers
from transformers import AdamW, BertTokenizer, BertForMultipleChoice, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch
from tqdm import tqdm
import json
import numpy as np
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        help="train | test",
        default="train"
        )
    # parser.add_argument('-do_valid', action='store_true')
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/context_selection/",
        )
    parser.add_argument(
        "--context_path",
        type=Path,
        help="Path to the context file.",
        default="./data/context.json"
        )
    parser.add_argument(
        "--input_file",
        type=Path,
        help="Path to the training or testing file.",
        default="./data/train.json"
        )
    parser.add_argument(
        "--val_file",
        type=Path,
        help="Path to the validation file.",
        # default="./data/valid.json"
        )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path of the output file.",
        default="./selection.json"
    )

    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--pretrained_model", type=str, default="bert-base-chinese")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation", type=int, default=64)
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("--logging_step", type=int, default=100)

    args = parser.parse_args()
    return args

class CS_Dataset(Dataset):
    def __init__(self, file):
        with  open(file, 'r', encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if "relevant" in data:
            return data['id'], data['question'], data['paragraphs'], data['paragraphs'].index(data['relevant'])
        else:
            return data['id'], data['question'], data['paragraphs'], 0


def train(args):
    with open(args.context_path, 'r', encoding="utf-8") as f:
        context = json.load(f)
    train_set = CS_Dataset(args.input_file)
    if args.val_file is not None:
        val_set = CS_Dataset(args.val_file)
        train_set = ConcatDataset([train_set, val_set])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    model = BertForMultipleChoice.from_pretrained(args.pretrained_model).to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    update_step = args.num_epoch * len(train_loader) // args.gradient_accumulation + args.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * update_step, update_step)

    for epoch in range(args.num_epoch):
        step = 1
        train_loss = train_hit = 0
        model.train()
        for i, (_, questions, choices, answers) in enumerate(train_loader):
            encoding = tokenizer([questions[0] for j in range(len(choices))], [context[j] for j in choices], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
            outputs = model(**{k: v.unsqueeze(0).to(args.device) for k, v in encoding.items()}, labels=answers.to(args.device))
            train_hit += (outputs.logits.argmax(-1).cpu() == answers).item()
            train_loss += outputs.loss.item()
            outputs.loss.backward()
            
            if step % args.gradient_accumulation == 0 or (i + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            step += 1

            if step % args.logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss / args.logging_step:.3f}, acc = {train_hit / args.logging_step:.3f}")
                train_loss = train_hit = 0

    model.save_pretrained(os.path.join(args.ckpt_dir, args.pretrained_model))

def test(args):
    with open(args.context_path, 'r', encoding="utf-8") as f:
        context = json.load(f)
    test_set = CS_Dataset(args.input_file)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model_dir = os.path.join(args.ckpt_dir, args.pretrained_model)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    model = BertForMultipleChoice.from_pretrained(model_dir).to(args.device)

    selection = []

    model.eval()
    for id, questions, choices, _ in tqdm(test_loader):
        encoding = tokenizer([questions[0] for j in range(len(choices))], [context[j] for j in choices], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        outputs = model(**{k: v.unsqueeze(0).to(args.device) for k, v in encoding.items()})
        selection.append({
            'id': id[0],
            'question': questions[0],
            'relevant': choices[outputs.logits.argmax(-1)].item()
        })

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    transformers.logging.set_verbosity_error()
    if args.mode == "test":
        test(args)
    else:
        train(args)