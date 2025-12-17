import json
from datasets import load_dataset
from pathlib import Path
import argparse

DATASET_CONFIGS = {
    "math500": {
        "path": "HuggingFaceH4/MATH-500",
        "split": "test",
        "input_col": "problem",
        "target_col": "answer",
        "postprocess_target": lambda x: float(x)
    },
    "amc23": {
        "path": "math-ai/amc23",
        "split": "test",
        "input_col": "question",
        "target_col": "answer",
        "postprocess_target": lambda x: x
    },
    "aime2025": {
        "path": "MathArena/aime_2025",
        "split": "train",
        "input_col": "problem",
        "target_col": "answer",
        "postprocess_target": lambda x: str(x)
    },
    "aime2024": {
        "path": "Maxwell-Jia/AIME_2024",
        "split": "train",
        "input_col": "Problem",
        "target_col": "Answer",
        "postprocess_target": lambda x: str(x)
    },
    "math500": {
        "path": "HuggingFaceH4/MATH-500",
        "split": "test",
        "input_col": "problem",
        "target_col": "answer",
        "postprocess_target": lambda x: str(x)
    },
    "minerva": {
        "path": "math-ai/minervamath",
        "split": "test",
        "input_col": "question",
        "target_col": "answer",
        "postprocess_target": lambda x: str(x)
    },      
}


def export_to_jsonl(
    dataset_name: str,
    output_path: str,
):
    cfg = DATASET_CONFIGS[dataset_name]

    ds = load_dataset(cfg["path"], split=cfg["split"])

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        for ex in ds:
            item = {
                "input": ex[cfg["input_col"]],
                "target": cfg["postprocess_target"](ex[cfg["target_col"]]),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(ds)} examples to {out_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export HF math datasets to JSONL format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_CONFIGS.keys(),
        help="Dataset name to export",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: datasets/<dataset>.jsonl)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_path = (
        args.output
        if args.output is not None
        else f"datasets/{args.dataset}.jsonl"
    )

    export_to_jsonl(
        dataset_name=args.dataset,
        output_path=output_path,
    )