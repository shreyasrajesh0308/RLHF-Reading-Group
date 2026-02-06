# UCLA ECE RLHF Reading Group

A group of UCLA ECE grad students studying [*Reinforcement Learning from Human Feedback*](https://rlhfbook.com) by Nathan Lambert.

**Goal:** Understand LLM post-training deeply enough to do LLM post training research.

**Book:** [rlhfbook.com](https://rlhfbook.com) | **Code:** [natolambert/rlhf-book](https://github.com/natolambert/rlhf-book) | **Reference runs:** [wandb.ai/natolambert/rlhf-book](https://wandb.ai/natolambert/rlhf-book)

**When:** Thursdays, 5:00 PM | **Where:** One of the rooms in E4 5th or 6th floor. 

---

## Schedule

| Week | Date | Chapter(s) | Topic | Exercise | Presenter |
|------|------|-----------|-------|----------|-----------|
| 1 | Feb 12@Faraday | Ch 1-3 | Introduction, history, training overview | Setup: clone repo, `uv sync`, run one training job | Shreyas |
| 2 | Feb 19 | Ch 4 | Instruction fine-tuning (SFT) | Write a simple SFT loop from scratch | TBD |
| 3 | Feb 26 | Ch 5 | Reward models (Bradley-Terry, ORM) | Train preference RM + ORM, compare curves | TBD |
| 4 | Mar 5 | Ch 5 cont | Process reward models | Train PRM, discuss ORM vs PRM tradeoffs | TBD |
| 5 | Mar 12 | Ch 6 | Policy gradients (REINFORCE, RLOO) | Implement REINFORCE loss by hand, run RLOO | TBD |
| 6 | Mar 19 | Ch 6 cont | PPO, GRPO | Run PPO vs GRPO, discuss clipping and variance | TBD |
| 7 | Mar 26 | Ch 7 | Reasoning & inference-time scaling | RLVR experiments | TBD |
| 8 | Apr 2 | Ch 8 | DPO, IPO | Implement DPO loss by hand, run IPO | TBD |
| 9 | Apr 9 | Ch 8 cont | SimPO, KTO, ORPO | Compare reference-free methods | TBD |
| 10 | Apr 16 | Ch 9-10 | Rejection sampling, nature of preferences | Discussion-heavy week | TBD |
| 11+ | Apr 23+ | Ch 11-17 / papers | Advanced topics + research directions | Pick a question, design an experiment | TBD |

## Setup

1. **Fork this repo** for your own implementations and experiments
2. Clone the upstream book code:

```bash
git clone https://github.com/natolambert/rlhf-book.git
cd rlhf-book/code
uv sync
```

3. Verify it works:

```bash
uv run python -m reward_models.train_orm --samples 100 --epochs 1
```

See [CLAUDE.md](CLAUDE.md) for full architecture notes and development commands.

## Format

- **Weekly, ~1.5-2 hours**
- One person presents the chapter (~20 min summary + 2-3 discussion questions)
- Everyone reads beforehand
- Second half is hands-on: run or modify code together
- Its important to implement core algorithms by hand, to understand them well enough before handing them over to agents. 
- All implementations go in your personal fork — compare approaches during meetings

## Meeting Notes

- [Week 1 — Feb 12: Introduction & Setup](notes/week01.md)

## Papers & Resources

*Papers referenced in discussion, supplementary reading, useful blog posts.*

| Paper / Resource | Relevant Chapter | Added By |
|-----------------|-----------------|----------|
| | | |

## Research Questions

*When something sparks a "what if..." — write it here. These become experiment ideas.*

-

## Week 1 Goals

- [ ] Everyone: fork repo, clone book code, install dependencies, confirm training runs
- [ ] Read chapters 1-3
- [ ] Assign Ch 4 presenter for week 2

## Join Us

Interested in joining? Email us at [shreyasrajesh38@ucla.edu](mailto:shreyasrajesh38@ucla.edu).
