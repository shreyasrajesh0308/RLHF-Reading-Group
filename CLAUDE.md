# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a reading group repository for studying Nathan Lambert's *Reinforcement Learning from Human Feedback* (RLHF) book. The goal is to deeply understand LLM post-training methods — reward modeling, policy gradient RL, and direct alignment — well enough to do research.

The upstream code lives at [natolambert/rlhf-book](https://github.com/natolambert/rlhf-book). The book is at [rlhfbook.com](https://rlhfbook.com).

## Upstream Repository Setup

To work with the book's code examples:

```bash
git clone https://github.com/natolambert/rlhf-book.git
cd rlhf-book/code
uv sync
```

**Always use `uv run python` instead of bare `python`** to ensure the correct virtual environment:

```bash
uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
uv run python -m direct_alignment.train --config direct_alignment/configs/dpo.yaml
uv run python -m reward_models.train_orm --samples 400
```

Requires Python 3.12+. Flash Attention is installed by default on x86_64; ARM64 systems fall back to PyTorch SDPA automatically.

## Code Architecture (upstream `code/` directory)

Three independent packages, each corresponding to a book chapter:

### `policy_gradients/` — Chapter 6: Policy Gradient Methods
- **Algorithms:** REINFORCE, RLOO, PPO, GRPO, Dr. GRPO, GSPO, CISPO
- **Config:** Pydantic `BaseModel` in `config.py`, loaded from YAML files in `configs/`
- **Core abstractions:** `Experience` dataclass (trajectory data), `ReplayBuffer` (collection), loss classes in `loss.py` (all `nn.Module` with `forward(log_probs, experience)`)
- **Training:** `train.py` — rollout phase (generate completions, compute rewards via `reasoning_gym` procedural tasks) → training phase (policy gradient updates with gradient accumulation)
- **Entry:** `uv run python -m policy_gradients.train --config configs/<algo>.yaml`

### `direct_alignment/` — Chapter 8: Direct Alignment (DPO family)
- **Algorithms:** DPO, cDPO, IPO, SimPO, ORPO, KTO
- **Config:** Python `@dataclass` in `config.py` with loss-specific defaults in `__post_init__`
- **Core abstractions:** `PreferenceBatch` dataclass (chosen/rejected pairs), loss classes in `loss.py` (all `nn.Module` with `forward(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps)` returning `(loss, metrics_dict)`)
- **Data:** `data.py` — loads UltraFeedback or Anthropic HH preference datasets, normalizes to `[prompt, chosen, rejected]` format
- **Key function:** `compute_logprobs(logits, labels, mask)` — shared log-probability computation used by all losses
- **Training:** `train.py` — forward pass through policy + frozen reference model, compute loss, gradient accumulation
- **Entry:** `uv run python -m direct_alignment.train --config direct_alignment/configs/<algo>.yaml`

### `reward_models/` — Chapter 5: Reward Models
- **Model types:** ORM (Outcome), PRM (Process), Preference RM (Bradley-Terry)
- **Base class:** `base.py` — `BaseRewardModel(nn.Module)` with backbone + linear reward head, shared training utilities (`training_loop`, `create_optimizer`, wandb helpers)
- **Datasets:** GSM8K (ORM/PRM), UltraFeedback (Preference RM), PRM800K (PRM)
- **Entry:** `uv run python -m reward_models.train_orm`, `train_prm`, `train_preference_rm`

### How to add a new algorithm
- **New policy gradient:** Add loss class to `policy_gradients/loss.py`, register in `get_loss_objective()` in `train.py`, create a config YAML
- **New direct alignment:** Add loss class to `direct_alignment/loss.py`, register in `LOSS_FUNCTIONS` dict + `get_loss_function()`, create a config YAML
- **New reward model:** Subclass `BaseRewardModel` in a new `train_*.py`, add entry point in `pyproject.toml`

## Environment Variables

```bash
export WANDB_API_KEY="..."           # Required for experiment logging
export WANDB_PROJECT="rlhf-book"     # Optional override
export WANDB_MODE="disabled"         # To disable logging entirely
export HF_TOKEN="..."                # For gated HuggingFace models
```

## Model Sizing & Memory

| Model | GPU Memory (full fine-tune) |
|-------|---------------------------|
| Qwen3-0.6B | ~4-6 GB |
| Qwen3-1.7B | ~10-15 GB |
| Qwen2.5-3B | ~20-25 GB |

Default models: Qwen3-0.6B (reward models), Qwen3-1.7B (policy gradients), OLMo-2-0425-1B-SFT (direct alignment). Learning rates: ~1e-5 to 5e-6 for full fine-tuning (10x smaller than LoRA).

## Linting

```bash
uv run ruff check .          # Lint
uv run ruff check --fix .    # Auto-fix
uv run ruff format .         # Format
```

Config: `pyproject.toml` — targets Python 3.12, line length 100, selects E/F/I/W/B/C4 rules.

## Reference Runs

All example training runs are publicly viewable at [wandb.ai/natolambert/rlhf-book](https://wandb.ai/natolambert/rlhf-book) — use these to verify your training curves look reasonable.

## Book Chapter → Code Mapping

| Chapter | Code Package | Key Concept |
|---------|-------------|-------------|
| Ch 5: Reward Models | `reward_models/` | Bradley-Terry loss, ORM vs PRM |
| Ch 6: Policy Gradients | `policy_gradients/` | REINFORCE → PPO → GRPO evolution |
| Ch 8: Direct Alignment | `direct_alignment/` | DPO and why it works without RL |
