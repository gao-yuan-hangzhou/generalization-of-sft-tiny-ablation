from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sft_format_following.format_eval_callback import FormatEvalConfig, format_success_rate  # noqa: E402
from sft_format_following.metrics import check_strict_json_schema  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter_dir", type=str, default=None, help="If set, compare base vs base+adapter.")
    p.add_argument("--eval_file", type=str, required=True)
    p.add_argument("--num_prompts", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--only_failures", action="store_true", help="Print only examples that fail strict checks.")
    return p.parse_args()


def _load_4bit(base_model: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files={"eval": args.eval_file})["eval"]
    rows = ds.to_list()[: args.num_prompts]

    base = _load_4bit(args.base_model)
    base_rate, _ = format_success_rate(
        model=base,
        tokenizer=tokenizer,
        rows=rows,
        cfg=FormatEvalConfig(max_new_tokens=args.max_new_tokens, num_prompts=args.num_prompts),
        device=device,
    )
    print(json.dumps({"base_format_success_rate": base_rate}, indent=2))

    if not args.adapter_dir:
        return

    tuned = PeftModel.from_pretrained(base, args.adapter_dir)
    tuned.eval()
    tuned_rate, _ = format_success_rate(
        model=tuned,
        tokenizer=tokenizer,
        rows=rows,
        cfg=FormatEvalConfig(max_new_tokens=args.max_new_tokens, num_prompts=args.num_prompts),
        device=device,
    )
    print(json.dumps({"tuned_format_success_rate": tuned_rate}, indent=2))

    # Print concrete before/after samples (completion-only, sliced after prompt tokens).
    from sft_format_following.format_eval_callback import _build_generation_prompts  # noqa: E402

    gen_prompts = _build_generation_prompts(tokenizer, [r["prompt"] for r in rows])
    inputs = tokenizer(gen_prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.inference_mode():
        base_ids = base.generate(**inputs, **gen_kwargs)
        tuned_ids = tuned.generate(**inputs, **gen_kwargs)

    for i, row in enumerate(rows, start=1):
        schema = row.get("schema", {})
        b_comp = tokenizer.decode(base_ids[i - 1, prompt_len:], skip_special_tokens=True).strip()
        t_comp = tokenizer.decode(tuned_ids[i - 1, prompt_len:], skip_special_tokens=True).strip()
        b_ok = check_strict_json_schema(b_comp, schema).ok
        t_ok = check_strict_json_schema(t_comp, schema).ok

        if args.only_failures and b_ok and t_ok:
            continue

        print("\n" + "=" * 80)
        print(f"Example {i} | base_ok={b_ok} tuned_ok={t_ok}")
        print("- Prompt:")
        print(row["prompt"])
        print("- Base output:")
        print(b_comp)
        print("- Tuned output:")
        print(t_comp)


if __name__ == "__main__":
    main()
