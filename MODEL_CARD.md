# MigrationGym policy v0.1

**Type:** a 12 → 64 ReLU → 10 neural action-value network, trained with Double DQN. It is not an LLM and does not generate SQL. Each output selects a bounded, preimplemented migration action.

**Training:** 240,000 environment steps, seed 7, 50/50 standard/larger workload mixture. Initial network weights are random. Curriculum uses expert-generated initial states (prefix depths 1–24) but no expert transitions in replay. Curriculum probability decreases to zero after 144,000 steps. The episode clock resets after the prefix. Validation selects the checkpoint at step 210,000. The final training checkpoint is also provided and is not the selected model.

**Evaluation:** selected policy fixed before testing fresh seeds 300000–300199 and 400000–400199. 200/200 verified completions per suite, zero request errors, mean 7.47/9.07 actions. The scripted reference also completes every case. See raw `artifacts/benchmark.json`, the README, and experiment history for limitations and earlier failures.

**Intended use:** learning, inspecting a custom RL environment, comparing migration-control policies, and reproducing a bounded research experiment.

**Not validated for:** production migrations, PostgreSQL locks, real distributed clients, arbitrary SQL actions, untrusted agent code, or other migration families. No claims about real service uptime, wall-clock recovery speed, or production data safety follow from these results.

**Trust model:** the model receives observations and rewards; the action interface does not expose the accepted-write ledger. Verifier implementation is separate from policy logic but runs in the same trusted process. Whole-container restrictions do not isolate the evaluator from malicious policy code.

**Reproducibility:** package dependencies are specified in `pyproject.toml`; exact build runtime versions and weight SHA-256 appear in `artifacts/runtime.json`. Checkpoints contain weights only. Use the README training command to rerun from scratch. Small numerical differences across NumPy/BLAS/platform versions can alter training trajectories; one seed is not a robust statistical study.
