from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    completion: str
    schema: dict[str, str]


def build_chat_text(tokenizer: Any, prompt: str, completion: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # Fallback: plain prompt/completion concatenation.
    return f"{prompt}\n{completion}"


def build_sft_text(tokenizer: Any, prompt: str, completion: str) -> str:
    """
    Build training text so that the assistant role header is part of the *prompt*
    (add_generation_prompt=True) and the completion starts immediately after it.
    This helps penalize emitting redundant role headers / fences at the start.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        eos = getattr(tokenizer, "eos_token", None) or ""
        return f"{prompt_text}{completion}{eos}"

    return f"{prompt}\n{completion}"


def normalize_json_text(text: str) -> str:
    return text.strip()


def iter_examples(rows: Iterable[dict[str, Any]]) -> Iterable[Example]:
    for row in rows:
        yield Example(
            id=str(row.get("id", "")),
            prompt=str(row["prompt"]),
            completion=str(row["completion"]),
            schema=dict(row.get("schema", {})),
        )
