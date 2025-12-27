from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


NAMES = [
    "Alex Chen",
    "Maya Singh",
    "Jordan Lee",
    "Sam Rivera",
    "Priya Patel",
    "Noah Kim",
    "Emma Garcia",
    "Riley O'Neil",
    "Chen-Liu Zhang",
    "Jean-Pierre Dupont",
    "Zoë Alvarez",
]

ITEMS = [
    "lunch",
    "taxi",
    "coffee",
    "office supplies",
    "train ticket",
    "parking",
]


def _iso_date(rng: random.Random) -> str:
    year = rng.choice([2024, 2025])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def _amount(rng: random.Random) -> float:
    # Keep to 2 decimals but as a JSON number.
    cents = rng.randint(150, 25000)
    return round(cents / 100.0, 2)


def _base_instruction() -> str:
    return (
        "Extract fields from the text.\n"
        "Output ONLY valid JSON (no markdown, no explanation, no extra keys).\n"
        "The output MUST start with '{' as the first character and end with '}' as the last character.\n"
        "Do NOT output any role headers. Do NOT wrap JSON in code fences.\n"
        'JSON schema: {"name": string, "date": "YYYY-MM-DD", "amount": number}.\n'
    )

def _base_instruction_medium() -> str:
    # Intentionally less explicit than `_base_instruction()` so eval isn't trivially easy.
    return (
        "Extract fields from the transaction.\n"
        "Return ONLY a JSON object with keys: name, date, amount.\n"
        "No extra keys. No explanation.\n"
        'date must be "YYYY-MM-DD". amount must be a JSON number.\n'
    )

def _base_instruction_hard() -> str:
    # Hard eval: keep requirements, but reduce "hand-holding" and increase surface complexity.
    return (
        "You will be given multiple transaction-like lines.\n"
        "Output ONLY a JSON object with keys: name, date, amount.\n"
        "Use the designated TARGET line only.\n"
        "No extra keys. No explanation.\n"
    )


def _format_amount_text(rng: random.Random, amount: float) -> str:
    # Make the surface form harder (commas, optional space) but keep completion numeric.
    s = f"{amount:,.2f}"
    if rng.random() < 0.3:
        s = s.rstrip("0").rstrip(".")
    if rng.random() < 0.3:
        return f"$ {s}"
    return f"${s}"


def _make_transaction_text(
    rng: random.Random,
    *,
    name: str,
    date: str,
    amount: float,
    item: str,
    strictness: str,
) -> str:
    amount_text = _format_amount_text(rng, amount)
    if strictness == "strict":
        return f"Paid {name} on {date} {amount_text} for {item}."

    # Medium/Hard difficulty: add distractors but keep it unambiguous via a TARGET marker.
    other_name = rng.choice([n for n in NAMES if n != name])
    other_date = _iso_date(rng)
    other_amount = _format_amount_text(rng, _amount(rng))
    ref = rng.randint(100000, 999999)
    if strictness == "medium":
        return (
            f"TARGET: Paid {name} on {date} {amount_text} for {item}.\n"
            f"Reference: {ref}.\n"
            f"Previous: Paid {other_name} on {other_date} {other_amount} for coffee.\n"
            "Instruction: Extract fields from the TARGET line only."
        )

    # Hard: more distractors and more formatting noise in the non-target lines.
    other_name2 = rng.choice([n for n in NAMES if n not in {name, other_name}])
    other_date2 = _iso_date(rng)
    other_amount2 = _format_amount_text(rng, _amount(rng))
    txn_id = rng.randint(1000, 9999)
    note = rng.choice(
        [
            "NOTE: amounts may include commas; ignore non-target lines.",
            "NOTE: some lines are summaries; do not extract from them.",
            "NOTE: only the TARGET line is authoritative.",
        ]
    )
    return (
        f"{note}\n"
        f"TARGET [{txn_id}]: Paid {name} on {date} {amount_text} for {item}.\n"
        f"Summary: Paid {other_name} on {other_date} {other_amount} for coffee (monthly total).\n"
        f"Legacy record: {other_date2} | {other_name2} | {other_amount2} | misc.\n"
        f"Reference: {ref}."
    )


def _hard_negative_train_prefix() -> str:
    # Priming hard negative: show the exact common wrong pattern so the model learns to avoid it.
    # (Training only; this is intentionally "priming".)
    return (
        "You are evaluating strict formatting compliance.\n"
        "Some models incorrectly respond like:\n"
        "assistant\n"
        "```json\n"
        "{\"name\":\"...\",\"date\":\"...\",\"amount\":...}\n"
        "```\n"
        "Do NOT do that.\n"
    )


def _hard_negative_eval_prefix_non_priming() -> str:
    # Non-priming hard negative: warn against the behaviors without including trigger strings.
    return (
        "Strict formatting compliance test.\n"
        "Common failures include: adding a role label before the JSON, or wrapping the JSON in a fenced code block.\n"
        "Do not add any extra words, labels, or fences.\n"
    )

def _make_bad_output(rng: random.Random, json_text: str) -> str:
    style = rng.choice(["assistant_fence", "assistant_preamble", "fence_only"])
    if style == "assistant_fence":
        return f"assistant\n```json\n{json_text}\n```"
    if style == "assistant_preamble":
        return f"assistant\nSure! Here is the JSON you requested:\n{json_text}"
    return f"```json\n{json_text}\n```"


def _fix_prompt(bad_output: str) -> str:
    return (
        "Fix the following output so that it is ONLY valid JSON.\n"
        "Requirements:\n"
        "- Output MUST start with '{' and end with '}'.\n"
        "- No markdown/code fences.\n"
        "- No role labels.\n"
        "- No extra keys.\n"
        "Current output:\n"
        f"{bad_output}\n"
    )


def make_example(
    rng: random.Random,
    ex_id: str,
    *,
    hard_negative_prob: float,
    hard_negative_template: str,
    fix_prob: float = 0.0,
    instruction_strictness: str = "strict",
) -> dict:
    name = rng.choice(NAMES)
    date = _iso_date(rng)
    amount = _amount(rng)
    item = rng.choice(ITEMS)

    if instruction_strictness == "strict":
        base_instruction = _base_instruction()
    elif instruction_strictness == "medium":
        base_instruction = _base_instruction_medium()
    elif instruction_strictness == "hard":
        base_instruction = _base_instruction_hard()
    else:
        raise ValueError(f"Unknown instruction_strictness: {instruction_strictness}")

    completion_obj = {"name": name, "date": date, "amount": amount}

    # Diversify spacing/newlines while keeping the first character '{' and last '}'.
    fmt = rng.choice(["minified", "spaced", "pretty"])
    if fmt == "minified":
        completion = json.dumps(completion_obj, separators=(",", ":"), ensure_ascii=False)
    elif fmt == "spaced":
        completion = json.dumps(completion_obj, separators=(", ", ": "), ensure_ascii=False)
    else:
        completion = json.dumps(completion_obj, indent=2, ensure_ascii=False)
        if completion.startswith("\n"):
            completion = completion.lstrip("\n")

    if rng.random() < fix_prob:
        bad_output = _make_bad_output(rng, completion)
        tx_text = _make_transaction_text(
            rng, name=name, date=date, amount=amount, item=item, strictness=instruction_strictness
        )
        prompt = _fix_prompt(bad_output) + base_instruction + f"Text: {tx_text}"
        variant = "fix_bad_output"
    else:
        is_hard_negative = rng.random() < hard_negative_prob
        if is_hard_negative:
            if hard_negative_template == "train_priming":
                prompt = (
                    _hard_negative_train_prefix()
                    + base_instruction
                    + f"Text: {_make_transaction_text(rng, name=name, date=date, amount=amount, item=item, strictness=instruction_strictness)}"
                )
                variant = "hard_negative_train_priming"
            elif hard_negative_template == "eval_non_priming":
                prompt = (
                    _hard_negative_eval_prefix_non_priming()
                    + base_instruction
                    + f"Text: {_make_transaction_text(rng, name=name, date=date, amount=amount, item=item, strictness=instruction_strictness)}"
                )
                variant = "hard_negative_eval_non_priming"
            else:
                raise ValueError(f"Unknown hard_negative_template: {hard_negative_template}")
        else:
            prompt = base_instruction + f"Text: {_make_transaction_text(rng, name=name, date=date, amount=amount, item=item, strictness=instruction_strictness)}"
            variant = "basic"

    return {
        "id": ex_id,
        "prompt": prompt,
        "completion": completion,
        "schema": {"name": "str", "date": "date", "amount": "float"},
        "variant": variant,
        "json_format": fmt,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config file. If it contains a top-level `data` object, it will be used.",
    )
    pre_args, _ = pre.parse_known_args()

    config_path = pre_args.config
    if config_path is None and Path("run_config.json").exists():
        config_path = Path("run_config.json")

    cfg_data: dict[str, Any] = {}
    if config_path is not None and config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg_data = cfg.get("data", cfg)

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--out_dir", type=Path, default=cfg_data.get("out_dir"))
    p.add_argument("--n_train", type=int, default=cfg_data.get("n_train", 200))
    p.add_argument("--n_eval", type=int, default=cfg_data.get("n_eval", 50))
    p.add_argument("--seed", type=int, default=cfg_data.get("seed", 7))
    p.add_argument(
        "--hard_negative_prob_train",
        type=float,
        default=cfg_data.get("hard_negative_prob_train", 0.35),
        help="Probability of using a hard-negative prompt template for training examples.",
    )
    p.add_argument(
        "--hard_negative_prob_eval",
        type=float,
        default=cfg_data.get("hard_negative_prob_eval", 0.35),
        help="Probability of using a hard-negative prompt template for eval examples.",
    )
    p.add_argument(
        "--fix_bad_output_prob_train",
        type=float,
        default=cfg_data.get("fix_bad_output_prob_train", 0.15),
        help="Probability of training examples that ask the model to fix a bad (chatty/fenced) output.",
    )
    p.add_argument(
        "--eval_strictness",
        choices=["strict", "medium", "hard"],
        default=cfg_data.get("eval_strictness", "strict"),
        help="Controls how explicit/difficult eval prompts/text are.",
    )
    args = p.parse_args()

    if args.out_dir is None:
        raise SystemExit("Missing required arg: --out_dir (or set data.out_dir in --config / run_config.json).")

    rng = random.Random(args.seed)
    train = [
        make_example(
            rng,
            f"train_{i:06d}",
            hard_negative_prob=args.hard_negative_prob_train,
            hard_negative_template="train_priming",
            fix_prob=args.fix_bad_output_prob_train,
            instruction_strictness="strict",
        )
        for i in range(args.n_train)
    ]
    eval_rows = [
        make_example(
            rng,
            f"eval_{i:06d}",
            hard_negative_prob=args.hard_negative_prob_eval,
            hard_negative_template="eval_non_priming",
            fix_prob=0.0,
            instruction_strictness=args.eval_strictness,
        )
        for i in range(args.n_eval)
    ]

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "eval.jsonl", eval_rows)

    print(f"Wrote {len(train)} train and {len(eval_rows)} eval to {args.out_dir}")


if __name__ == "__main__":
    main()
