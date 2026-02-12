# Week 1 — Feb 12: Introduction, History & Training Overview

**UCLA ECE RLHF Reading Group**

**Chapters:** 1-3 | **Presenter:** Shreyas

---

## Chapter 1: Introduction — Key Takeaways

### What RLHF actually does
- RLHF operates at the **response level** (not per-token like SFT) — it tells the model what a *better* response looks like via contrastive feedback
- The core contribution is **style and behavioral shaping** — warmth, formatting, safety, engagement — things that SFT alone can't reliably instill
- RLHF generalizes far better across domains than SFT (Kirk et al. 2023, Chu et al. 2025)

### Three types of post-training
1. **SFT/IFT** — teaches format, instruction-following. Learns *features* in language
2. **Preference Fine-tuning (PreFT)** — aligns to human preferences. Learns *style* and subtle preferences. **This is where RLHF lives**
3. **RLVR** — RL with verifiable rewards for reasoning domains. Newest, fastest-evolving

### The Elicitation Theory of Post-Training
- Post-training doesn't teach new knowledge — it **extracts latent capabilities** from the base model
- Analogy: base model = F1 chassis, post-training = aerodynamics and systems tuning
- The Superficial Alignment Hypothesis (LIMA paper) gets the direction right but undersells the impact — post-training does far more than superficial style changes, especially with RLVR for reasoning

### Why this matters
- Lambert argues RLHF was *the* technique that enabled ChatGPT's success
- Companies that embraced RLHF early (Anthropic, OpenAI) built lasting advantages
- The field went through a skepticism phase ("SFT is enough") that delayed open-source progress

---

## Chapter 2: Key Related Works — Timeline

### Phase 1: Origins (pre-2018)
- **TAMER (2008)** — humans iteratively score agent actions → reward model → policy. The proto-RLHF
- **Christiano et al. 2017** — the primary reference. RLHF on Atari trajectories. Showed preference feedback can beat direct environment interaction
- Key shift: reward models proposed as tool for studying **alignment**, not just solving RL tasks (Leike et al. 2018)

### Phase 2: Language Models (2019-2022)
- **Ziegler et al. 2019** — first RLHF on LMs. Already had reward models, KL penalties, feedback loops — strikingly similar to modern work
- Applied to: summarization (Stiennon 2020), instruction following (InstructGPT/Ouyang 2022), web QA (WebGPT), dialogue (Sparrow)
- Foundational concepts established: reward model over-optimization (Gao et al.), red teaming (Ganguli et al.)

### Phase 3: ChatGPT Era (2023+)
- ChatGPT explicitly credited RLHF. Used in Claude, Llama 2/3, Nemotron, Tülu 3
- DPO (May 2023) → didn't take off until fall 2023 when the right learning rates were found (Zephyr-Beta, Tülu 2)
- Field expanding into: process reward models, direct alignment algorithms, RLVR for reasoning

---

## Chapter 3: Training Overview — Core Concepts

### The RLHF objective
Standard RL: maximize expected reward over trajectories
RLHF simplification: no state transitions, response-level (bandit) rewards, learned reward model

$$J(\pi) = \mathbb{E}[r_\theta(x, y)] - \beta \cdot D_{\text{KL}}(\pi \parallel \pi_{\text{ref}})$$

Three key differences from standard RL:
1. **Reward function → reward model** (learned, not environmental)
2. **No state transitions** (prompt in, completion out — single step)
3. **Response-level rewards** (bandit-style, not per-timestep)

### The KL penalty
- Prevents the policy from drifting too far from the reference (initial) model
- β controls the trade-off: too low → over-optimization/reward hacking, too high → no learning
- The "KL budget" concept — how much deviation from the base model are you willing to spend?

### Three canonical recipes (increasing complexity)

**InstructGPT (2022):** SFT (10K) → Reward Model (100K pairs) → RLHF (100K prompts)
Simple, three-stage. The template everything builds on.

**Tülu 3 (2024):** SFT (1M synthetic) → On-policy DPO (1M pairs) → RLVR (10K prompts)
Much more data, synthetic data-heavy, adds RLVR for reasoning.

**DeepSeek R1 (2025):** Cold-start SFT (100K+ reasoning) → Large-scale RLVR → Rejection sampling → Mixed RL
Reasoning-first. RL is the centerpiece, not an afterthought. Represents the current frontier.

The trend: **more stages, more data, more RL, reasoning-centric.**

---

## Discussion Questions

1. **The Elicitation Theory vs. Superficial Alignment:** Lambert argues post-training extracts deep capabilities, not just surface style. But LIMA showed you can get surprisingly far with 1K examples. Where's the truth? Is there a threshold where more preference data stops helping and you need a fundamentally different signal (like RLVR)?

2. **Why did it take so long for DPO to work?** The math was published in May 2023 but the first good models weren't until fall 2023 — and the fix was just a lower learning rate. What does this tell us about the gap between theory and practice in post-training? What other "obvious in hindsight" practical details might be hiding in current methods?

3. **The three recipes (InstructGPT → Tülu 3 → DeepSeek R1) show a clear trend toward more RL.** Is RLHF (preference-based) becoming less important relative to RLVR (verifiable rewards)? Or do they serve fundamentally different purposes — preferences for style/safety, verifiable rewards for capabilities?

4. **The KL penalty is doing a lot of work.** It's the main thing preventing reward hacking and model collapse. But it also limits how far the model can improve. How should we think about setting β? Is there a principled way, or is it mostly empirical?

5. **Open vs. closed gap:** Lambert notes companies that embraced RLHF early (Anthropic) built lasting advantages, and that open-source was stuck in a "SFT is enough" phase. As of 2025/2026, has the open-source community closed this gap? What's still missing?

---

## Key Equations to Know

**The RLHF objective** — maximize reward while staying close to the reference policy:

$$J(\pi) = \mathbb{E}[r_\theta(x,y)] - \beta \, D_{\text{KL}}(\pi \parallel \pi_{\text{ref}})$$

**Trajectory distribution in standard RL** — contrast with the simplified RLHF bandit setup:

$$p_\pi(\tau) = \rho_0(s_0) \prod_{t} \pi(a_t \mid s_t) \, p(s_{t+1} \mid s_t, a_t)$$

---

## Notes

![The early three-stage RLHF process: SFT, reward model, then RL optimization](../rlhf-book/book/images/rlhf-basic.png)

- **Core intuition for RLHF:** It's a way to bake in "human taste" into AI systems. These preferences are hard to specify — you can't write them as a simple loss function like next-token prediction. But humans *can* compare outputs ("A is better than B"), and RLHF turns that comparative signal into a training objective via reward models. It's "I know it when I see it" turned into a gradient.

---

## Action Items

- [ ] Everyone confirms `rlhf-book/code` setup works (`uv sync`) before next week
- [ ] Assign Ch 4+5 presenter
- [ ] Next week: read Ch 4 (Instruction Tuning) + Ch 5 (Reward Models) — Ch 4 is short conceptual setup, Ch 5 is the first real technical deep-dive
- [ ] Hands-on: try running a reward model training before the meeting:
  ```bash
  cd rlhf-book/code
  WANDB_MODE=disabled uv run python -m reward_models.train_orm --samples 400 --epochs 2
  ```
- [ ] Optional: skim the InstructGPT paper (Ouyang et al. 2022) and Christiano et al. 2017 for deeper context
