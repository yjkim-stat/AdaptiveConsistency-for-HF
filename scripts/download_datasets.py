import json
from datasets import load_dataset
from pathlib import Path


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
    }    
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


dataset_name = 'amc23'
dataset_name = 'aime2025'
export_to_jsonl(
    dataset_name=f"{dataset_name}",
    output_path=f"datasets/{dataset_name}.jsonl"
)
