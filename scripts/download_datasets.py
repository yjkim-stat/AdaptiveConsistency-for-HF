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
    "olympiad": {
        "path": "Hothan/OlympiadBench",
        "split": "train",
        "input_col": "question",
        "target_col": "final_answer",
        "postprocess_target": lambda x: str(x[0])
    },      
    "mmlupro": {
        "path": "TIGER-Lab/MMLU-Pro",
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

    if dataset_name == 'olympiad':
        ds = load_dataset(cfg["path"], 'OE_TO_maths_en_COMP', split=cfg["split"])
    else:
        ds = load_dataset(cfg["path"], split=cfg["split"])

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

        # test_query = test_sample["question"]
        # options = test_sample["options"]
        # formatted_query = f"{test_query}\n"
    with out_file.open("w", encoding="utf-8") as f:
        for ex in ds:
            if dataset_name == 'mmlupro':
                test_query = ex[cfg["input_col"]]
                options = ex["options"]
                input_str = f"{test_query}\n"
                for i, choice in enumerate(options):
                    input_str += f"{i}. {choice}\n"
                item = {
                    "input": input_str,
                    "target": cfg["postprocess_target"](ex[cfg["target_col"]]),
                }
            else:
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