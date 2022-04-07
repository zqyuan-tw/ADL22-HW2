import os
import transformers
from transformers import AdamW, BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
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
    # parser.add_argument('--do_valid', action='store_true')
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent_classification/",
        )
    parser.add_argument(
        "--input_file",
        type=Path,
        help="Path to the training or testing file.",
        default="../ADL21-HW1/data/intent/train.json"
        )
    parser.add_argument(
        "--mapping",
        type=Path,
        help="Path to the training or testing file.",
        default="../ADL21-HW1/cache/intent/intent2idx.json"
        )
    # parser.add_argument(
    #     "--val_file",
    #     type=Path,
    #     help="Path to the validation file.",
    #     default="../ADL21-HW1/data/intent/eval.json"
    #     )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path of the output file.",
        default="./intent.json"
    )

    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation", type=int, default=64)
    parser.add_argument("--num_epoch", type=int, default=2)
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("--logging_step", type=int, default=500)

    args = parser.parse_args()
    return args

class IC_Dataset(Dataset):
    def __init__(self, file, mapping):
        with open(file, 'r') as f, open(mapping, 'r') as m:
            self.data = json.load(f)
            self.intent2label = json.load(m)
        self.label2intent = {value: key for (key, value) in self.intent2label.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if "intent" in data:
            return data['id'], data['text'], self.intent2label[data['intent']]
        else:
            return data['id'], data['text'], 0


def train(args):
    train_set = IC_Dataset(args.input_file, args.mapping)
    # if args.do_valid and args.val_file is not None:
    #     val_set = IC_Dataset(args.val_file, args.mapping)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=len(train_set.intent2label)).to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    update_step = args.num_epoch * len(train_loader) // args.gradient_accumulation + args.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * update_step, update_step)

    for epoch in range(args.num_epoch):
        step = 1
        train_loss = train_hit = 0
        model.train()
        for i, (_, text, label) in enumerate(train_loader):
            inputs = tokenizer(text[0], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to(args.device)
            outputs = model(**inputs, labels=label.to(args.device))
            train_hit += (outputs.logits.argmax().cpu() == label).item()
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
    test_set = IC_Dataset(args.input_file, args.mapping)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model_dir = os.path.join(args.ckpt_dir, args.pretrained_model)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=len(test_set.intent2label)).to(args.device)

    intent = []

    model.eval()
    for id, text, _ in tqdm(test_loader):
        inputs = tokenizer(text[0], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        intent.append({
            'text': text[0],
            'intent': test_set.label2intent[outputs.logits.argmax().item()],
            'id': id[0]           
        })

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(intent, f, indent=2)

if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    transformers.logging.set_verbosity_error()
    if args.mode == "test":
        test(args)
    else:
        train(args)