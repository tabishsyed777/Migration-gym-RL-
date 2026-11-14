# Explaining MigrationGym

## What did you build?

A bounded RL environment for learning the order and timing of a staged schema migration. It executes actual SQLite statements while application writes continue between migration actions. A separate verifier compares the final database with an accepted-write ledger and checks schema and lifecycle requirements. The project includes a NumPy neural Double DQN implementation, training curriculum, baselines, tests, Docker packaging, and a checkpoint.

Describe your own contribution accurately. If AI assisted implementation, say so and be ready to modify and explain the code live. Do not claim PostgreSQL, real concurrent services, or process-isolated verification until implemented and tested.

## What makes this reinforcement learning?

The model chooses actions from observations, receives rewards from their consequences, and changes its parameters using replayed transitions. It is not a fixed migration script. The script is a separate baseline and generates some initial states during curriculum training. Evaluation starts from scratch without scripted help.

## What is gradient descent doing?

The network estimates discounted future return for each action. Double DQN chooses the next action with the online network and values that action with a delayed target network. The difference between the current prediction and the bootstrapped target is the temporal-difference error. Huber loss limits the gradient of large errors, and Adam updates the network weights using those gradients. This is value-based learning; the policy is the action with the greatest estimated value.

## Why isn't reward enough to prove correctness?

A reward is an optimization signal, not proof. A learner can exploit mistakes in reward design. Here success is based on actual final database contents, schema, client retirement, read mode, and recorded request failures. The reward uses that verdict. The same-process implementation is still trusted infrastructure: arbitrary code execution would invalidate its trust boundary.

## Why use a curriculum?

The initial no-curriculum run learned to avoid failure by waiting until timeout. Starting some episodes near later migration stages lets the learner encounter successful completions and propagate their value backward. The probability of curriculum resets decreases to zero. This uses expert knowledge in the training distribution and is explicitly disclosed.

## Why not just use the script?

For this constrained migration family, the script is an excellent practical solution. The purpose is an inspectable environment and training benchmark, not a claim that RL is necessary or superior. The research question becomes more meaningful as workload tradeoffs, multiple migration families, uncertain telemetry, and reversible faults are added.

## What are the most important limitations?

Serial SQLite traffic, aggregate client lifecycle simulation, all-row read audits, a single rename family, one primary training seed, modeled rather than measured large-batch latency, and no adversarial isolation between learner and verifier. Results only support this benchmark's defined task distribution.

## Practice exercises

1. Add a new workload distribution and evaluate without retraining.
2. Remove backfill coverage from observations and explain the resulting partial observability.
3. Implement another independent ledger-mismatch test.
4. Compare small versus large batches under different modeled load costs.
5. Trace one TD update and identify the online and target-network roles.
6. Implement a second migration family and explain which observations/actions change.
