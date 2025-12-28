# sft_format_following

Small-scale QLoRA SFT (fits 4GB laptop GPUs) to improve strict output formatting, especially **JSON-only** responses with zero extra text.

## Setup

1) Create an environment and install deps (install PyTorch separately for your CUDA setup):

```bash
pip install -r requirements.txt
```

2) Generate data:

```bash
python scripts/make_example_data.py \
  --out_dir data/generated \
  --n_train 2000 \
  --n_eval 1000 \
  --hard_negative_prob_train 0.4 \
  --hard_negative_prob_eval 0.4 \
  --fix_bad_output_prob_train 0.25 \
  --eval_strictness hard
```

This also writes `stats.json` and `stats.md` into the output folder to document the data mix.

Alternatively, use the committed tiny dataset in `data/examples/` for a smoke test.

## Config file (optional)

Both `scripts/make_example_data.py` and `scripts/train_sft_qlora.py` accept `--config <json>`.
If `run_config.json` exists in the repo root, both scripts will use it by default.

Start by copying `configs/run_config.example.json` to `run_config.json` and editing paths/params.

Hard-eval preset: `configs/run_config_hard_eval.json` (generates `data/generated_hard_eval/`).

Hard-eval + duplicate-fix preset: `configs/run_config_hard_eval_dupfix.json` (generates `data/generated_hard_eval_dupfix_v4/`).

## One-command run (regen → train → dump generations)

```bash
python scripts/make_example_data.py --config configs/run_config_hard_eval_dupfix.json && \
python scripts/train_sft_qlora.py --config configs/run_config_hard_eval_dupfix.json && \
python scripts/evaluate_format.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir outputs/qwen2_5_0_5b_json_sft_hard_eval_dupfix_v4 \
  --eval_file data/generated_hard_eval_dupfix_v4/eval.jsonl \
  --num_prompts 200 \
  --max_new_tokens 128 \
  --dump_jsonl outputs/qwen2_5_0_5b_json_sft_hard_eval_dupfix_v4/all_eval_generations.jsonl
```

## Train (QLoRA + LoRA)

```bash
python scripts/train_sft_qlora.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_file data/generated/train.jsonl \
  --eval_file data/generated/eval.jsonl \
  --output_dir outputs/qwen2_5_0_5b_json_sft \
  --tb_logdir runs/qwen2_5_0_5b_json_sft \
  --precision bf16 \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 2 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --eval_steps 25 \
  --save_steps 25 \
  --save_total_limit 10 \
  --n_format_eval_prompts 200 \
  --format_eval_max_new_tokens 128
```

This runs an initial baseline eval at step `0` and logs metrics on every eval step.

TensorBoard logs go to `runs/...` if you set `--tb_logdir` (otherwise `<output_dir>/tb`).

Note: training text construction appends the tokenizer’s EOS token after each completion (it is stripped at decode time) to strongly encourage the model to stop right after the JSON object, reducing duplicate/extra generations.

## Evaluate format success rate

```bash
python scripts/evaluate_format.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir outputs/qwen2_5_0_5b_json_sft \
  --eval_file data/generated/eval.jsonl \
  --num_prompts 200 \
  --max_new_tokens 128 \
  --dump_jsonl outputs/qwen2_5_0_5b_json_sft/all_eval_generations.jsonl
```

This reports:
- `format_success_rate`: strict JSON-only success rate
- `duplicate_json_objects_rate`: fraction of completions containing 2+ JSON objects (common failure mode)
- `extras`: additional counters used for derived rates

## Visualize

Preferred: `tensorboard --logdir runs`

Or export scalars to a PNG:

```bash
python scripts/plot_tb_scalars.py \
  --logdir runs/qwen2_5_0_5b_json_sft \
  --out_png outputs/qwen2_5_0_5b_json_sft/curves.png
```

## Before/after diffs

```bash
python scripts/compare_outputs.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir outputs/qwen2_5_0_5b_json_sft \
  --eval_file data/generated/eval.jsonl \
  --num_prompts 8
```
