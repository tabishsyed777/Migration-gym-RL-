# Build and validation status

Built 2026-11-14.

- Python package installed and imported successfully.
- 19 tests passed, including Gymnasium environment checker, real SQLite integrity checks, deterministic resets, checkpoint round-trip, and a gradient-learning check.
- Source compiled successfully.
- Trained the included neural policy for 240,000 steps with seed 7.
- Ran standard and stress evaluations, 200 episodes per policy per distribution (1,200 total evaluation episodes).
- Ran both successful standard and larger-workload demos, plus a preserved failure trace from the standard-only experiment; traces included.
- Upload verification: Docker build attempted; dependency installation was blocked by the local network certificate chain (CERTIFICATE_VERIFY_FAILED). Container execution remains unverified locally; GitHub Actions includes build and runtime checks.
- No PostgreSQL backend, real concurrent client processes, or independently sandboxed verifier implemented.
- Repository license: MIT; the existing LICENSE file is preserved.

Upload verification on Python 3.12: editable installation, all 19 tests, README Gymnasium example, both demos, and all 1,200 benchmark episodes passed. The rerun benchmark exactly matched the supplied report. Training was not repeated during upload.
