from __future__ import annotations

import sys

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sft_format_following.data import build_sft_text
from sft_format_following.format_eval_callback import (
    FormatEvalCallback,
    FormatEvalConfig,
)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config file. If it contains a top-level `train` object, it will be used.",
    )
    pre_args, _ = pre.parse_known_args()

    config_path = pre_args.config
    if config_path is None and Path("run_config.json").exists():
        config_path = Path("run_config.json")

    cfg_train: dict = {}
    if config_path is not None and config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg_train = cfg.get("train", cfg)

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--model_name", type=str, default=cfg_train.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    p.add_argument("--train_file", type=str, default=cfg_train.get("train_file"))
    p.add_argument("--eval_file", type=str, default=cfg_train.get("eval_file"))
    p.add_argument("--output_dir", type=str, default=cfg_train.get("output_dir"))
    p.add_argument(
        "--tb_logdir",
        type=str,
        default=cfg_train.get("tb_logdir", ""),
        help="TensorBoard log directory. Default: <output_dir>/tb",
    )

    p.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default=cfg_train.get("precision"),
        help="Mixed precision mode. Default: bf16 if supported else fp16.",
    )

    p.add_argument("--max_seq_length", type=int, default=cfg_train.get("max_seq_length", 512))
    p.add_argument("--per_device_train_batch_size", type=int, default=cfg_train.get("per_device_train_batch_size", 1))
    p.add_argument("--per_device_eval_batch_size", type=int, default=cfg_train.get("per_device_eval_batch_size", 1))
    p.add_argument("--gradient_accumulation_steps", type=int, default=cfg_train.get("gradient_accumulation_steps", 16))
    p.add_argument("--learning_rate", type=float, default=cfg_train.get("learning_rate", 2e-4))
    p.add_argument("--num_train_epochs", type=float, default=cfg_train.get("num_train_epochs", 2.0))
    p.add_argument("--warmup_ratio", type=float, default=cfg_train.get("warmup_ratio", 0.03))
    p.add_argument("--seed", type=int, default=cfg_train.get("seed", 42))

    p.add_argument("--logging_steps", type=int, default=cfg_train.get("logging_steps", 10))
    p.add_argument("--eval_steps", type=int, default=cfg_train.get("eval_steps", 100))
    p.add_argument("--save_steps", type=int, default=cfg_train.get("save_steps", 100))
    p.add_argument("--save_total_limit", type=int, default=cfg_train.get("save_total_limit", 2))

    p.add_argument("--max_train_samples", type=int, default=cfg_train.get("max_train_samples", 0))
    p.add_argument("--max_eval_samples", type=int, default=cfg_train.get("max_eval_samples", 0))

    # QLoRA / LoRA
    p.add_argument("--lora_r", type=int, default=cfg_train.get("lora_r", 8))
    p.add_argument("--lora_alpha", type=int, default=cfg_train.get("lora_alpha", 16))
    p.add_argument("--lora_dropout", type=float, default=cfg_train.get("lora_dropout", 0.05))

    # Memory stability
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")

    # Format metric via generation
    p.add_argument("--n_format_eval_prompts", type=int, default=cfg_train.get("n_format_eval_prompts", 32))
    p.add_argument("--format_eval_max_new_tokens", type=int, default=cfg_train.get("format_eval_max_new_tokens", 128))
    p.add_argument(
        "--eval_on_start",
        action="store_true",
        default=bool(cfg_train.get("eval_on_start", True)),
        help="Run one evaluation at global_step=0 (logs to TensorBoard) before training.",
    )
    p.add_argument(
        "--no_eval_on_start",
        action="store_false",
        dest="eval_on_start",
        help="Disable the initial evaluation at global_step=0.",
    )

    args = p.parse_args()
    missing = [k for k in ["train_file", "eval_file", "output_dir"] if not getattr(args, k)]
    if missing:
        raise SystemExit(f"Missing required args (provide via CLI or --config): {', '.join(missing)}")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tb_dir = Path(args.tb_logdir) if args.tb_logdir else (output_dir / "tb")
    tb_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    precision = args.precision or ("bf16" if bf16_supported else "fp16")
    if precision == "bf16" and not bf16_supported:
        raise SystemExit("Requested --precision bf16 but torch reports bf16 is not supported on this GPU.")

    # Avoid mismatched precision settings from an existing `accelerate config`.
    os.environ.setdefault(
        "ACCELERATE_MIXED_PRECISION",
        "bf16" if precision == "bf16" else ("fp16" if precision == "fp16" else "no"),
    )

    if precision == "bf16":
        compute_dtype = torch.bfloat16
    elif precision == "fp16":
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",
    )
    if precision != "fp32":
        model_kwargs["dtype"] = compute_dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    ds = load_dataset("json", data_files={"train": args.train_file, "eval": args.eval_file})

    def to_text(batch):  # noqa: ANN001
        return {
            "text": [
                build_sft_text(tokenizer, p, c) for p, c in zip(batch["prompt"], batch["completion"], strict=True)
            ]
        }

    train_ds = ds["train"].map(to_text, batched=True, remove_columns=ds["train"].column_names)
    eval_ds_raw = ds["eval"]
    eval_ds = ds["eval"].map(to_text, batched=True, remove_columns=ds["eval"].column_names)

    if args.max_train_samples and args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples and args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
        eval_ds_raw = eval_ds_raw.select(range(min(args.max_eval_samples, len(eval_ds_raw))))

    sft_config_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        optim="paged_adamw_8bit",
        report_to=["tensorboard"],
        logging_dir=str(tb_dir),
        seed=args.seed,
        dataloader_pin_memory=False,
        eval_strategy="steps",
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        completion_only_loss=True,
    )
    sft_config = SFTConfig(**sft_config_kwargs)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        args=sft_config,
    )

    eval_rows = eval_ds_raw.to_list()
    trainer.add_callback(
        FormatEvalCallback(
            tokenizer=tokenizer,
            eval_rows=eval_rows,
            cfg=FormatEvalConfig(
                max_new_tokens=args.format_eval_max_new_tokens,
                num_prompts=args.n_format_eval_prompts,
            ),
        )
    )

    if args.eval_on_start:
        # Baseline metrics for the (effectively) base model at step=0.
        # This triggers the same logging pipeline as periodic evals (including format_success_rate).
        print("[baseline_eval] running initial eval at global_step=0 ...")
        trainer.evaluate()

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
