import json
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context", 
        type=Path,
        help="Path to the context file.",
        default="./data/context.json"
        )
    parser.add_argument(
        "--input_file", 
        type=Path,
        help="Path to the input file.",
        required=True
        )
    parser.add_argument(
        "--output_file", 
        type=Path,
        help="Path to the output file.",
        default="./squad_like.json"
        )
    args = parser.parse_args()
    with open(args.context, "r", encoding="utf-8") as c, open(args.input_file, "r", encoding="utf-8") as f:
        context = json.load(c)
        file = json.load(f)
    
    squad = []
    for data in file:
        s = {
            "context" : context[data["relevant"]],
            "id" : data["id"],
            "question" : data["question"],
        }
        if "answer" in data:
            s["answers"] = { 
                "text" : [data["answer"]["text"]],
                "answer_start" : [data["answer"]["start"]],
            }
        squad.append(s)
        
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({"data": squad},f, ensure_ascii=False, indent=2)