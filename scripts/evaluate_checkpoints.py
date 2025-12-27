from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sft_format_following.format_eval_callback import FormatEvalConfig, format_success_rate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--run_dir", type=Path, required=True, help="Training output dir containing checkpoint-*.")
    p.add_argument("--eval_file", type=str, required=True)
    p.add_argument("--num_prompts", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--out_csv", type=Path, default=None)
    p.add_argument("--tb_logdir", type=Path, default=None, help="If set, writes eval_format_success_rate scalar.")
    p.add_argument(
        "--strip_leading_assistant_marker",
        action="store_true",
        help="Relaxed metric: ignore a leading `assistant\\n` role marker in the generated completion.",
    )
    return p.parse_args()


def _step_from_checkpoint_name(name: str) -> int:
    # checkpoint-1234
    try:
        return int(name.split("-", 1)[1])
    except Exception:  # noqa: BLE001
        return -1


def _load_4bit_base(base_model: str, compute_dtype: torch.dtype):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=compute_dtype,
    )
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_rows = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            eval_rows.append(json.loads(line))

    checkpoints = sorted(
        [p for p in args.run_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: _step_from_checkpoint_name(p.name),
    )
    if not checkpoints:
        raise SystemExit(f"No checkpoint-* dirs found in {args.run_dir}")

    tb_writer = SummaryWriter(log_dir=str(args.tb_logdir)) if args.tb_logdir else None
    rows_out: list[dict[str, object]] = []

    for ckpt in checkpoints:
        step = _step_from_checkpoint_name(ckpt.name)
        base = _load_4bit_base(args.base_model, compute_dtype=compute_dtype)
        model = PeftModel.from_pretrained(base, str(ckpt))
        model.eval()
        rate, _ = format_success_rate(
            model=model,
            tokenizer=tokenizer,
            rows=eval_rows,
            cfg=FormatEvalConfig(
                max_new_tokens=args.max_new_tokens,
                num_prompts=args.num_prompts,
                strip_leading_assistant_marker=args.strip_leading_assistant_marker,
            ),
            device=device,
        )
        rows_out.append({"step": step, "checkpoint": ckpt.name, "eval_format_success_rate": rate})
        print(f"{ckpt.name}\tstep={step}\teval_format_success_rate={rate:.4f}")
        if tb_writer is not None and step >= 0:
            tag = "eval/format_success_rate_relaxed" if args.strip_leading_assistant_marker else "eval/format_success_rate"
            tb_writer.add_scalar(tag, rate, step)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["step", "checkpoint", "eval_format_success_rate"])
            w.writeheader()
            w.writerows(rows_out)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
