# Experiment history

All exploratory runs used training seed 7. These are development observations, not a controlled comparison across multiple random seeds.

1. `initial_no_curriculum.json`: 120,000 steps; no expert initial states; hard target every 250 learner updates, Adam 0.0007. Validation success remained zero.
2. `hard_target_experiment/`: 160,000 steps; expert-prefix initial states retained elapsed prefix time; hard target every 250 updates, Adam 0.0007. Best validation success 82.5%; evaluated standard success 76%, larger-workload success 39.5%.
3. `soft_target_experiment/`: 240,000 steps; same elapsed-prefix-time curriculum; Adam 0.0002 and target Polyak rate 0.01. Validation success remained zero.
4. `standard_only_experiment/`: 160,000 steps; fresh clock after expert prefix; hard target every 250 updates, Adam 0.0007. Standard success 100%, larger-workload success 32% on the original evaluation suites. Demonstrates a meaningful generalization gap despite standard-task success.
5. `mixed_short_curriculum_experiment/`: 240,000 steps, both workloads; prefix depths 1–12 did not cover many late stages of larger migrations. Validation remained unstable and unsuccessful for much of training.
6. Main `training.json`: 240,000 steps, mixed workloads, prefix depths 1–24 and a fresh clock; select by average validation return across both distributions. The selected step-210000 checkpoint completed 100% of both fresh 200-episode suites. The last checkpoint regressed on standard validation, illustrating the need for checkpoint selection. Larger-workload evaluation is now in-distribution, not an out-of-distribution generalization claim.

Original evaluation suites began at seeds 100000 and 200000. The mixed-workload experiment's final evaluation uses newly reserved suites starting at 300000 and 400000, not used for model selection. Validation suites begin at 50000 and 60000. Scenario seeds control initialization and traffic; they do not make distributions confidential.

The original no-curriculum and older exploratory code versions are not separately packaged. Their configuration changes are recorded here; the main runnable code implements the final curriculum and supports standard-only or mixed-workload training. Checkpoints are inference snapshots, not full optimizer/replay checkpoints.
