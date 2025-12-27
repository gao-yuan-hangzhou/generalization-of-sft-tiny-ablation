from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback

from .data import build_chat_text, iter_examples
from .metrics import (
    check_extractable_json_schema,
    check_strict_json_schema,
    strip_leading_assistant_marker,
)


@dataclass(frozen=True)
class FormatEvalConfig:
    max_new_tokens: int = 128
    num_prompts: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    strip_leading_assistant_marker: bool = False


def _build_generation_prompts(tokenizer: Any, prompts: list[str]) -> list[str]:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompts
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]


@torch.inference_mode()
def format_success_rate(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    cfg: FormatEvalConfig,
    device: torch.device,
) -> tuple[float, dict[str, int]]:
    examples = list(iter_examples(rows))[: cfg.num_prompts]
    prompts = [ex.prompt for ex in examples]
    schemas = [ex.schema for ex in examples]

    generation_prompts = _build_generation_prompts(tokenizer, prompts)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        generation_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=getattr(tokenizer, "model_max_length", 2048),
    ).to(device)
    # Important: `generate()` appends after the full padded input length (input_ids.shape[1]),
    # not after the per-example non-pad token count. Using attention_mask.sum() would
    # incorrectly include prompt tail tokens in the "completion" slice for left-padded batches.
    prompt_len = int(inputs["input_ids"].shape[1])

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.temperature > 0.0,
        temperature=cfg.temperature if cfg.temperature > 0.0 else None,
        top_p=cfg.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    outputs = model.generate(**inputs, **gen_kwargs)

    ok = 0
    extracted_ok = 0
    first_char_brace_ok = 0
    reasons: dict[str, int] = {}
    for i, schema in enumerate(schemas):
        completion_ids = outputs[i, prompt_len:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        if cfg.strip_leading_assistant_marker:
            completion_text = strip_leading_assistant_marker(completion_text)
        stripped = completion_text.lstrip()
        if stripped.startswith("{"):
            first_char_brace_ok += 1

        result = check_strict_json_schema(completion_text, schema)
        if result.ok:
            ok += 1
        else:
            reasons[result.reason or "unknown"] = reasons.get(result.reason or "unknown", 0) + 1

        extracted = check_extractable_json_schema(completion_text, schema)
        if extracted.ok:
            extracted_ok += 1

    total = max(len(examples), 1)
    # We keep reasons only for strict failures; extra counters are returned with "__" keys.
    reasons["__extractable_ok"] = extracted_ok
    reasons["__first_char_brace_ok"] = first_char_brace_ok
    reasons["__total"] = total
    return ok / total, reasons


class FormatEvalCallback(TrainerCallback):
    def __init__(self, tokenizer: Any, eval_rows: list[dict[str, Any]], cfg: FormatEvalConfig):
        self._tokenizer = tokenizer
        self._eval_rows = eval_rows
        self._cfg = cfg
        self._writer: SummaryWriter | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
        model = kwargs.get("model")
        if model is None:
            return

        device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        rate, reasons = format_success_rate(
            model=model,
            tokenizer=self._tokenizer,
            rows=self._eval_rows,
            cfg=self._cfg,
            device=device,
        )
        relaxed_rate, relaxed_reasons = format_success_rate(
            model=model,
            tokenizer=self._tokenizer,
            rows=self._eval_rows,
            cfg=FormatEvalConfig(
                max_new_tokens=self._cfg.max_new_tokens,
                num_prompts=self._cfg.num_prompts,
                temperature=self._cfg.temperature,
                top_p=self._cfg.top_p,
                strip_leading_assistant_marker=True,
            ),
            device=device,
        )

        # Robust logging: write directly to TensorBoard. Some Trainer versions do not
        # propagate callback-mutated `metrics` to loggers.
        logdir = getattr(args, "logging_dir", None)
        if logdir:
            if self._writer is None:
                self._writer = SummaryWriter(log_dir=logdir)
            step = int(getattr(state, "global_step", 0))
            self._writer.add_scalar("eval/format_success_rate", rate, step)
            self._writer.add_scalar("eval/format_success_rate_relaxed", relaxed_rate, step)
            # Extra, more sensitive diagnostics.
            total = int(reasons.get("__total", 1))
            extracted_ok = int(reasons.get("__extractable_ok", 0))
            first_char_ok = int(reasons.get("__first_char_brace_ok", 0))
            relaxed_total = int(relaxed_reasons.get("__total", total))
            relaxed_extracted_ok = int(relaxed_reasons.get("__extractable_ok", 0))
            relaxed_first_char_ok = int(relaxed_reasons.get("__first_char_brace_ok", 0))
            self._writer.add_scalar("eval/extractable_json_rate", extracted_ok / max(total, 1), step)
            self._writer.add_scalar("eval/prefix_brace_rate", first_char_ok / max(total, 1), step)
            self._writer.add_scalar(
                "eval/extractable_json_rate_relaxed", relaxed_extracted_ok / max(relaxed_total, 1), step
            )
            self._writer.add_scalar("eval/prefix_brace_rate_relaxed", relaxed_first_char_ok / max(relaxed_total, 1), step)
            self._writer.flush()
            print(
                f"[format_eval] step={step} eval/format_success_rate={rate:.4f} "
                f"eval/format_success_rate_relaxed={relaxed_rate:.4f}"
            )
            print(
                f"[format_eval] step={step} eval/extractable_json_rate={extracted_ok / max(total, 1):.4f} "
                f"eval/prefix_brace_rate={first_char_ok / max(total, 1):.4f}"
            )

        if metrics is not None:
            metrics["eval_format_success_rate"] = rate
            metrics["eval_format_success_rate_relaxed"] = relaxed_rate
            metrics["eval_extractable_json_rate"] = int(reasons.get("__extractable_ok", 0)) / max(
                int(reasons.get("__total", 1)), 1
            )
            metrics["eval_prefix_brace_rate"] = int(reasons.get("__first_char_brace_ok", 0)) / max(
                int(reasons.get("__total", 1)), 1
            )
            for k, v in sorted(reasons.items())[:10]:
                if k.startswith("__"):
                    continue
                metrics[f"eval_format_fail/{k}"] = float(v)
