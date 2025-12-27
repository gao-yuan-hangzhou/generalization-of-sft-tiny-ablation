from __future__ import annotations

import sys

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sft_format_following.format_eval_callback import FormatEvalConfig, format_success_rate
from sft_format_following.metrics import check_strict_json_schema, count_json_objects


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--eval_file", type=str, required=True)
    p.add_argument("--num_prompts", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--out_json", type=Path, default=None)
    p.add_argument("--dump_jsonl", type=Path, default=None, help="If set, write per-example outputs to JSONL.")
    p.add_argument(
        "--only_failures",
        action="store_true",
        help="With --dump_jsonl, write only examples that fail strict JSON checks.",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--strip_leading_assistant_marker",
        action="store_true",
        help="Relaxed metric: ignore a leading `assistant\\n` role marker in the generated completion.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    ds = load_dataset("json", data_files={"eval": args.eval_file})["eval"]
    rows = ds.to_list()

    rate, reasons = format_success_rate(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        cfg=FormatEvalConfig(
            max_new_tokens=args.max_new_tokens,
            num_prompts=args.num_prompts,
            strip_leading_assistant_marker=args.strip_leading_assistant_marker,
        ),
        device=device,
    )
    reasons_clean = {k: v for k, v in reasons.items() if not k.startswith("__")}
    extras = {k: v for k, v in reasons.items() if k.startswith("__")}
    total = int(extras.get("__total", args.num_prompts) or 1)
    duplicate_json_objects_rate = int(extras.get("__duplicate_json_objects", 0)) / max(total, 1)
    result = {
        "format_success_rate": rate,
        "duplicate_json_objects_rate": duplicate_json_objects_rate,
        "fail_reasons": reasons_clean,
        "extras": extras,
    }
    print(json.dumps(result, indent=2))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.dump_jsonl is not None:
        # Generate and score per-example outputs (use the same slicing logic as format_success_rate).
        rows_subset = rows[: args.num_prompts]
        prompts = [r["prompt"] for r in rows_subset]
        schemas = [r.get("schema", {}) for r in rows_subset]

        if hasattr(tokenizer, "apply_chat_template"):
            generation_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
        else:
            generation_prompts = prompts

        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        args.dump_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_jsonl.open("w", encoding="utf-8") as f:
            for start in range(0, len(generation_prompts), args.batch_size):
                batch_prompts = generation_prompts[start : start + args.batch_size]
                batch_rows = rows_subset[start : start + args.batch_size]
                batch_schemas = schemas[start : start + args.batch_size]

                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=getattr(tokenizer, "model_max_length", 2048),
                ).to(device)
                prompt_len = int(inputs["input_ids"].shape[1])

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                for i, (row, schema) in enumerate(zip(batch_rows, batch_schemas, strict=True)):
                    completion_ids = outputs[i, prompt_len:]
                    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                    raw_completion_text = completion_text
                    if args.strip_leading_assistant_marker:
                        completion_text = completion_text.lstrip()
                        if completion_text.startswith("assistant") and "\n" in completion_text:
                            completion_text = completion_text.split("\n", 1)[1].lstrip()
                    check = check_strict_json_schema(completion_text, schema)
                    num_json_objects = count_json_objects(completion_text, max_objects=3)
                    if args.only_failures and check.ok:
                        continue
                    out_row = {
                        "id": row.get("id"),
                        "ok": check.ok,
                        "reason": check.reason,
                        "prompt": row.get("prompt"),
                        "pred": completion_text,
                        "pred_raw": raw_completion_text,
                        "num_json_objects": num_json_objects,
                        "gold": row.get("completion"),
                        "schema": schema,
                    }
                    f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        print(f"Wrote {args.dump_jsonl}")


if __name__ == "__main__":
    main()
