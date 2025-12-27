# sft_format_following

Small-scale QLoRA SFT (fits 4GB laptop GPUs) to improve strict output formatting, especially **JSON-only** responses with zero extra text.

## Setup

1) Create an environment and install deps (install PyTorch separately for your CUDA setup):

```bash
pip install -r requirements.txt
```

2) Generate a small starter dataset:

```bash
python scripts/make_example_data.py --out_dir data/generated --n_train 200 --n_eval 50
```

Alternatively, use the committed tiny dataset in `data/examples/` for a smoke test.

## Config file (optional)

Both `scripts/make_example_data.py` and `scripts/train_sft_qlora.py` accept `--config <json>`.
If `run_config.json` exists in the repo root, both scripts will use it by default.

Start by copying `configs/run_config.example.json` to `run_config.json` and editing paths/params.

## Train (QLoRA + LoRA)

```bash
python scripts/train_sft_qlora.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_file data/generated/train.jsonl \
  --eval_file data/generated/eval.jsonl \
  --output_dir outputs/qwen2_5_0_5b_json_sft \
  --precision bf16 \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 2 \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 100
```

TensorBoard logs go to `outputs/.../tb/`.

## Evaluate format success rate

```bash
python scripts/evaluate_format.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir outputs/qwen2_5_0_5b_json_sft \
  --eval_file data/generated/eval.jsonl \
  --max_new_tokens 128
```

## Visualize

Preferred: `tensorboard --logdir outputs`

Or export scalars to a PNG:

```bash
python scripts/plot_tb_scalars.py --logdir outputs/qwen2_5_0_5b_json_sft/tb --out_png outputs/qwen2_5_0_5b_json_sft/curves.png
```

## Before/after diffs

```bash
python scripts/compare_outputs.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir outputs/qwen2_5_0_5b_json_sft \
  --eval_file data/generated/eval.jsonl \
  --num_prompts 8
```
