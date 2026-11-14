# MigrationGym

[![Checks](https://github.com/tabishsyed777/Migration-gym-RL-/actions/workflows/ci.yml/badge.svg)](https://github.com/tabishsyed777/Migration-gym-RL-/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A verifiable reinforcement learning benchmark for staged database migrations, built with SQLite, Gymnasium, and a NumPy Double DQN agent.

**Train an RL policy to migrate a live database column without losing acknowledged writes.**

MigrationGym is a runnable research prototype, not a production migration tool. It uses a real in-memory SQLite database, deterministic synthetic application traffic, a custom Gymnasium environment, an independently implemented outcome verifier, and a neural Double DQN policy trained from scratch in NumPy.

The task migrates `accounts.balance` to `accounts.balance_cents`. Both columns store integer cents: this is a schema rename / staged deployment benchmark, not a currency conversion. The policy must expand the schema, deploy dual writes, backfill data, retire legacy clients, switch reads, remove the old column, and submit a verified result.

## Highlights

- Real in-memory SQLite schema changes with synthetic writes between agent actions.
- Ten migration actions and twelve telemetry features in a Gymnasium environment.
- Double DQN implemented in NumPy, with trained checkpoints included.
- Independent outcome checks against an accepted-write ledger.
- Random and scripted baselines, reproducible evaluation seeds, tests, and Docker packaging.

**Explore:** [Quick start](#run-the-included-trained-model) · [Training](#train-and-evaluate) · [Results](#measured-results) · [Model card](MODEL_CARD.md) · [Experiment history](artifacts/EXPERIMENTS.md) · [Limitations](#boundaries-and-next-experiments)

## Run the included trained model

Python 3.10+ and SQLite 3.35+ are required. Python 3.11 is recommended. No AI API, GPU, Ollama, database server, or paid service is required.

```bash
git clone https://github.com/tabishsyed777/Migration-gym-RL-.git
cd Migration-gym-RL-
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
migrationgym demo
migrationgym demo --stress --seed 200003
pytest -q
```

On Windows activate `.venv\Scripts\activate` instead. Run commands from this repository's root.

## Use the Gymnasium environment

```python
import gymnasium as gym
import migrationgym  # Registers MigrationGym-v0.

with gym.make("MigrationGym-v0") as env:
    observation, info = env.reset(seed=42)
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()  # Replace with your policy.
        observation, reward, terminated, truncated, info = env.step(action)
    print(info)
```

Random actions usually fail; use `migrationgym demo` to inspect the included trained policy.

## Train and evaluate

```bash
OPENBLAS_NUM_THREADS=1 migrationgym train --steps 240000 --seed 7 --mixed-workloads --output artifacts
migrationgym benchmark --episodes 200 --seed-base 300000 --output artifacts/benchmark.json
```

`artifacts/policy.npz` is the selected validation checkpoint, `last_policy.npz` is the final checkpoint, `training.json` records training configuration and validation history, and `benchmark.json` contains held-out results. Checkpoints contain network weights only; they support inference but not exact training resumption. The train command overwrites these artifact paths; use a different `--output` directory to preserve the included run.

## RL task

| Property | Specification |
|---|---|
| Observation | 12 normalized features: schema expanded, dual writes, backfill coverage, new reads, active old clients, retirement started, old column dropped, time, request errors, row count, load, invalid-action count |
| Actions | Expand; dual-write; backfill 4 or 16 rows; switch reads; retire clients; wait; drop legacy; submit; roll back reads |
| Episode | Up to 40 actions; submission or any request/data-read error terminates; time limit truncates |
| Standard workload | 8–24 accounts, 1–4 old clients, 1–3 writes per action |
| Larger workload (`stress`) | 25–48 accounts, 4–7 old clients, 3–6 writes per action; included in mixed training |
| Traffic | Writes interleave after each action; values and accounts are randomized; one retiring client leaves per step |
| Success | Exact agreement with independent accepted-write ledger, no extra/missing rows, target schema complete, old clients retired, new reads active, zero request errors |

Observation features expose aggregate telemetry, not the expected account values in the verifier ledger. The NumPy learner receives only observation vectors and rewards, not database handles or info fields.

**Reward:** -0.15 per action, -0.8 per invalid action, -0.12 × load for a large backfill, -2 per new request error, +30 for verified success or -25 for failed submission / error termination / timeout. Large-backfill latency is a modeled cost, not measured SQLite latency. Potential shaping adds `0.98 Φ(next) − Φ(current)`, with zero terminal potential. Φ is twice the sum of expansion, dual writes, dual-write backfill coverage, retirement started, retired-client fraction relative to eight, new reads, and legacy removal. Using a potential prevents repeatedly collecting a milestone reward by toggling read mode. Evaluation reports this shaped return; success and request errors remain separate metrics.

## Neural model

The Q network has 12 inputs, a 64-unit ReLU hidden layer, and 10 action values. It uses Double DQN targets, a replay buffer of 30,000 transitions, batch size 64, Huber loss, Adam at 0.0007, discount 0.98, and a target-network copy every 250 updates. Learning starts after 1,000 steps and updates every four steps. Epsilon decays from approximately 1 to 0.08 over 65% of the training budget. Actions are **not masked**, and no scripted demonstrations are inserted into replay. The included model trains on a 50/50 mixture of standard and larger workloads (`--mixed-workloads`). Without that flag, training uses standard workloads only. Training uses a safe-prefix curriculum: reset to a randomly chosen prefix (depth 1–24, stopping before submission) of the scripted policy with probability `0.9 × max(0, 1 − step/(0.6 × training_budget))`. The learner never receives expert transitions in replay, but these expert-generated initial states are supervision and must be disclosed. Curriculum starts receive a fresh episode clock to avoid teaching the learner to wait for the time at which an expert prefix would finish. After 60% of training, all episodes start from untouched databases. Validation and evaluation always start from untouched databases.

The implementation includes explicit gradients and optimizer state so the learning mechanism is inspectable. It is not a pretrained language model. Gradient descent updates network parameters to reduce temporal-difference prediction error; the reward defines the task, while the independent verifier decides whether the migration actually succeeded.

## Evaluation protocol

Validation uses 40 fixed standard seeds starting at 50,000 and, for mixed training, 40 larger-workload seeds starting at 60,000. The checkpoint maximizes average return across the two validation suites. Final standard evaluation starts at 300,000; larger-workload evaluation starts at 400,000. Both distributions are included in mixed training: these are fresh scenarios, not an out-of-distribution test. All three policies see the same initial seeds. Action-dependent traffic consumption means trajectories can diverge after different actions. Compare success, failures, steps, and returns—not reward alone.

Baselines are random actions and a conservative scripted expand/dual-write/retire/backfill/switch/drop policy. A learned policy need not beat that script: the script encodes a known safe strategy and is a strong reference. The initial run without curriculum stalled at zero validation success; its history is preserved in `artifacts/initial_no_curriculum.json`. The included main experiment is one training seed, not evidence of robust multi-seed algorithm superiority. Episode-level Wilson intervals do not measure training-seed uncertainty. Evaluation seeds are disjoint, but public and inspectable; they are not a confidential benchmark.

## Verifier and resistance to gaming

`migrationgym/verifier.py` derives success from actual SQLite schema/data plus the ledger, client count, read mode, and cumulative errors. It does not trust an agent's "migration complete" claim or the backfill progress feature. Tests cover early submission, premature drop, incomplete backfill, value corruption, missing/extra rows, ignored errors, and active old clients.

The policy controls only the enumerated actions. The verifier and ledger are not exposed through that action interface. **This is logical API separation, not a security boundary against arbitrary Python code:** environment and verifier currently share a process, and some lifecycle counters are supplied by the environment. A malicious program with host access could alter either. Independent-process verification and a durable external ledger are future work. The verifier is deterministic; no LLM judges success.

## Measured results

Included run: 240,000 environment steps, training seed 7, mixed workloads, selected checkpoint at step 210,000. Each cell below uses 200 fresh held-out episodes.

| Policy | Standard success | Larger-workload success | Standard mean actions | Larger mean actions |
|---|---:|---:|---:|---:|
| Random | 0% | 0% | 7.02 | 6.945 |
| Scripted | 100% | 100% | 9.145 | 11.73 |
| Double DQN | **100%** | **100%** | **7.47** | **9.07** |

The trained policy had zero observed request errors and no failed verifier verdicts in these 400 episodes. Both success intervals are approximately 98.1–100% (episode-level 95% Wilson). These are measurements on the defined workload distributions, not a guarantee of reliability or algorithm-wide confidence intervals. The neural policy uses fewer actions than the conservative script, but its larger-workload return is slightly lower because it uses costlier large backfills. It does not dominate that baseline or establish superiority over optimized scripts.

The standard-only model succeeded on all 200 original standard episodes but only 32% of larger cases. Mixed training with a curriculum long enough to reach later stages addressed that observed gap on new in-distribution cases. Failed runs and old results remain in `artifacts/`, with a configuration history in `artifacts/EXPERIMENTS.md`. These exploratory experiments share a training seed and are not a controlled ablation study. Final suites starting at 300000/400000 were reserved for the final mixed-workload checkpoint and were not used for selection. Multiple training seeds and additional unseen migration families remain necessary before broader reliability claims.

## Docker

```bash
docker build -t migrationgym .
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --memory 256m --cpus 1 \
  --pids-limit 64 migrationgym
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges migrationgym demo --stress
```

The container runs as a non-root user. This isolates the entire application from the host; it does not isolate the policy from the verifier inside the container. Local Docker validation was blocked by a network certificate error during dependency installation; see `artifacts/BUILD_STATUS.md` for checks actually run for this release.

## Boundaries and next experiments

SQLite traffic is serially interleaved between actions. It does not reproduce PostgreSQL lock contention, actual concurrent clients, replication lag, or network services. Old clients are represented by a lifecycle model; they do not run as separate processes. All row values are audited each step, giving stronger detection than typical production monitoring. The error budget is zero and unrecoverable, so rollback is intentionally not useful after an error. These choices make the first environment reproducible but simplify real operations.

Next substantive milestones: a PostgreSQL integration backend; actual old/new client processes; transaction rollback and reversible incidents; migration families beyond rename; evaluator isolation; multiple training seeds and ablations of shaping and observations. Do not describe these as implemented.

## Repository

- `migrationgym/env.py`: database, actions, traffic, observations, reward
- `migrationgym/verifier.py`: independent schema and ledger checks
- `migrationgym/agent.py`: neural network, Double DQN, replay, Adam
- `migrationgym/cli.py`: training, baseline evaluation, demo
- `tests/`: environment contract, data integrity, reproducibility, learner tests
- `artifacts/`: trained checkpoints and measured reports

## References

- [Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/)
- [Double DQN paper](https://arxiv.org/abs/1509.06461)
- [Potential-based reward shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

## License

Licensed under the [MIT License](LICENSE). Copyright © 2026 Tabish Syed.
