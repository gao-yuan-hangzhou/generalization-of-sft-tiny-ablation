# Small-Scale SFT on a 4GB Laptop GPU (RTX 3050)

## Purpose

This project demonstrates that **meaningful supervised fine-tuning (SFT)** is possible on a **4GB laptop GPU** (RTX 3050, WDDM) by targeting **narrow, high-visibility behaviors**, specifically:

- Strict **format adherence**
- Schema-constrained outputs (e.g., JSON-only)
- Elimination of extraneous text (no markdown, no commentary)

The goal is **not** general capability improvement, but to produce:
- Clear **train / eval curves**
- A **quantifiable behavioral metric**
- Obvious **before/after output diffs**

This serves as a reproducible reference for “small but real” fine-tuning under severe VRAM constraints.

---

## Hardware & OS Constraints

### GPU
- NVIDIA GeForce RTX 3050 (Laptop)
- VRAM: 4 GB (≈3.5–3.7 GB usable under WDDM)
- Power cap: ~45W

### OS
- Windows 11 host
- **WSL2 (Ubuntu)** recommended for training
  - Required for reliable `bitsandbytes` support
  - Native Windows is not recommended for QLoRA

---

## Fine-Tuning Strategy

### Method
- **QLoRA (4-bit)**
- LoRA adapters only (base model frozen)

### Rationale
- Full fine-tuning does not fit in 4 GB
- 8-bit LoRA is unstable on WDDM
- 4-bit NF4 + LoRA is the only reliable approach

---

## Model Selection

### Primary Model
- `Qwen/Qwen2.5-0.5B-Instruct`

### Why This Model
- Strong instruction-following per parameter
- Short-context friendly
- Stable tokenizer (important for formatting tasks)
- Fits comfortably in 4 GB with QLoRA

### Optional (More Aggressive)
- `Qwen/Qwen2.5-1.5B-Instruct`
  - Requires `max_seq_length <= 256`
  - Higher OOM risk on laptop GPUs

---

## Task Definition

### Target Behavior
Strict output formatting with **zero tolerance** for deviations.

Example task:
- Extract structured fields from text
- Output **ONLY valid JSON**
- Exact keys, correct types
- No markdown, no explanations

### Why This Task
- Small datasets produce large, visible gains
- Easy to define binary success/failure
- Ideal for demonstrating SFT effectiveness under constraints

---

## Dataset Specification

### Format
Each example contains:
- A prompt with explicit formatting constraints
- A completion that exactly matches the required format

Example:
```json
{
  "prompt": "Extract fields. Output ONLY valid JSON with keys name, date, amount.\nText: Paid Alex Chen on 2025-10-03 $47.20 for lunch.",
  "completion": "{\"name\":\"Alex Chen\",\"date\":\"2025-10-03\",\"amount\":47.2}"
}
```
Dataset Size
Minimum: ~2,000 examples

Recommended: 5,000–20,000 examples

Data Quality Rules
No ambiguous outputs

No alternative valid formats

Penalize verbosity implicitly by exclusion

Training Configuration
Quantization
python
Copy code
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = torch.float16
LoRA
python
Copy code
r = 8
lora_alpha = 16
lora_dropout = 0.05
bias = "none"
task_type = "CAUSAL_LM"
Core Training Params
yaml
Copy code
max_seq_length: 512
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2e-4
num_train_epochs: 2
fp16: true
Memory Notes
Activation memory dominates → sequence length is the primary lever

If OOM occurs:

Reduce max_seq_length to 256

Reduce LoRA rank to 4

Evaluation Strategy
Standard Metrics
Training loss

Evaluation loss (held-out set)

Primary Behavioral Metric (Required)
Format Success Rate

Definition:

Model output parses successfully (e.g., json.loads)

Required keys present

No extra text before or after

Correct data types

Reported as:

text
Copy code
format_success_rate = valid_outputs / total_outputs
This metric is more informative than loss for format-enforcement tasks.

Logging & Visualization
TensorBoard recommended

Log:

Train loss

Eval loss

Format success rate over steps

Expected behavior:

Loss decreases smoothly

Format success rate jumps sharply after early steps

Expected Results
Before SFT
Extraneous explanations

Markdown fences around JSON

Incorrect typing (numbers as strings)

Occasional schema violations

After SFT
Outputs contain only the target format

Schema adherence >90% on held-out eval

Consistent behavior across prompts

Runtime
~30–60 minutes for 5k examples

Stable VRAM usage ~3.3–3.6 GB

Non-Goals / Explicit Limitations
This setup is not intended to:

Improve long-form reasoning

Handle long-context tasks

Produce general-purpose instruction improvements

Replace large-scale SFT or RLHF

It is a focused, constrained demonstration of real fine-tuning under extreme hardware limits.

Success Criteria
The experiment is considered successful if:

Training runs without OOM on a 4GB laptop GPU

Train/eval curves are stable and interpretable

Format success rate improves substantially

Before/after outputs show obvious behavioral change

Handoff Notes for Implementation
Assume WSL2 + CUDA is available

Use Hugging Face transformers, trl, peft, bitsandbytes

Keep everything minimal; avoid extra callbacks or hooks

Favor determinism and clarity over performance optimizations
