from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=Path, required=True, help="Directory containing event files.")
    p.add_argument("--out_png", type=Path, required=True)
    p.add_argument("--tags", type=str, default="train/loss,eval/loss,format_success_rate")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    ea = EventAccumulator(str(args.logdir))
    ea.Reload()

    plt.figure(figsize=(10, 4))
    for tag in tags:
        if tag not in ea.Tags().get("scalars", []):
            continue
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        plt.plot(steps, vals, label=tag)

    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()

