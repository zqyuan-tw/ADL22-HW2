import json
import argparse
from pathlib import Path
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", 
        type=Path,
        help="Path to the input file.",
        default="./predict_predictions.json"
        )
    parser.add_argument(
        "--output_file", 
        type=Path,
        help="Path to the output file.",
        default="./submission.csv"
        )
    args = parser.parse_args()
    with open(args.input_file, "r", encoding="utf-8") as f:
        file = json.load(f)
    
    with open(args.output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        for id in file:
            writer.writerow([id, file[id]])