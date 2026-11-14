"""Real in-memory SQLite, bounded actions, synthetic interleaved application traffic."""
from enum import IntEnum
import sqlite3
import gymnasium as gym
import numpy as np
from .verifier import verify

class Action(IntEnum):
    EXPAND = 0
    DUAL_WRITE = 1
    BACKFILL_SMALL = 2
    BACKFILL_LARGE = 3
    SWITCH_READS = 4
    RETIRE_CLIENTS = 5
    WAIT = 6
    DROP_LEGACY = 7
    SUBMIT = 8
    ROLLBACK_READS = 9

class MigrationEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_steps=40, stress=False, render_mode=None):
        self.action_space = gym.spaces.Discrete(len(Action))
        self.observation_space = gym.spaces.Box(0, 1, (12,), dtype=np.float32)
        self.max_steps, self.stress, self.render_mode = max_steps, stress, render_mode
        self.db = None
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        options = options or {}
        self.n = int(options.get("rows", self.np_random.integers(8, 25) if not self.stress else self.np_random.integers(25, 49)))
        if not 1 <= self.n <= 64:
            raise ValueError("rows must be in [1, 64]")
        self.initial_clients = int(options.get("old_clients", self.np_random.integers(1, 5) if not self.stress else self.np_random.integers(4, 8)))
        if not 0 <= self.initial_clients <= 8:
            raise ValueError("old_clients must be in [0, 8]")
        self.old_clients = self.initial_clients
        self.load = int(self.np_random.integers(1, 4) if not self.stress else self.np_random.integers(3, 7))
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE accounts(id INTEGER PRIMARY KEY, balance INTEGER NOT NULL)")
        self.ledger = {i: int(self.np_random.integers(100, 100000)) for i in range(self.n)}
        self.db.executemany("INSERT INTO accounts VALUES (?, ?)", self.ledger.items())
        self.expanded = self.dual = self.read_new = self.retiring = self.dropped = False
        self.steps = self.errors = self.invalid = self.large_batches = 0
        self.done = False
        return self._obs(), {}

    def _missing(self):
        if not self.expanded:
            return self.n
        return self.db.execute("SELECT count(*) FROM accounts WHERE balance_cents IS NULL").fetchone()[0]

    def _obs(self):
        return np.array([self.expanded, self.dual, 1-self._missing()/self.n,
                         self.read_new, self.old_clients/8, self.retiring, self.dropped,
                         self.steps/self.max_steps, min(self.errors/10, 1), self.n/64,
                         self.load/6, min(self.invalid/10, 1)], dtype=np.float32)

    def _traffic(self):
        # Accepted writes are independently recorded only after a successful DB write.
        for _ in range(self.load):
            key = int(self.np_random.integers(self.n))
            delta = int(self.np_random.integers(-50, 51))
            if self.dropped and (self.old_clients or not self.dual):
                self.errors += 1
                continue
            if self.dropped:
                self.db.execute("UPDATE accounts SET balance_cents=balance_cents+? WHERE id=?", (delta, key))
            elif self.dual:
                self.db.execute("UPDATE accounts SET balance=balance+?, balance_cents=balance+? WHERE id=?", (delta, delta, key))
            else:
                self.db.execute("UPDATE accounts SET balance=balance+? WHERE id=?", (delta, key))
            self.ledger[key] += delta
        # Audit reads, including rows that were backfilled before dual-write deployment.
        col = "balance_cents" if self.read_new else "balance"
        if col == "balance" and self.dropped:
            self.errors += 1
        elif col == "balance_cents" and not self.expanded:
            self.errors += 1
        else:
            rows = dict(self.db.execute(f"SELECT id, {col} FROM accounts"))
            self.errors += sum(rows.get(k) != v for k, v in self.ledger.items())

    def _potential(self):
        # Potential shaping: progress cannot be harvested repeatedly by toggling reads.
        return 2.0 * (self.expanded + self.dual + (1-self._missing()/self.n)*self.dual + self.retiring + (1-self.old_clients/8) + self.read_new + self.dropped)

    def step(self, action):
        if self.done:
            raise RuntimeError("reset() required after episode end")
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        potential_before = self._potential()
        a = Action(int(action))
        self.steps += 1
        reward, valid = -0.15, True
        if a == Action.EXPAND:
            if self.expanded: valid = False
            else:
                self.db.execute("ALTER TABLE accounts ADD COLUMN balance_cents INTEGER")
                self.expanded = True
        elif a == Action.DUAL_WRITE:
            if not self.expanded or self.dual or self.dropped: valid = False
            else: self.dual = True
        elif a in (Action.BACKFILL_SMALL, Action.BACKFILL_LARGE):
            if not self.expanded or self.dropped or self._missing() == 0: valid = False
            else:
                batch = 4 if a == Action.BACKFILL_SMALL else 16
                self.db.execute("UPDATE accounts SET balance_cents=balance WHERE id IN (SELECT id FROM accounts WHERE balance_cents IS NULL ORDER BY id LIMIT ?)", (batch,))
                if a == Action.BACKFILL_LARGE:
                    self.large_batches += 1
                    reward -= 0.12 * self.load  # modeled lock/latency cost, not measured wall time
        elif a == Action.SWITCH_READS:
            if not self.expanded or self.read_new: valid = False
            else: self.read_new = True
        elif a == Action.RETIRE_CLIENTS:
            if self.retiring: valid = False
            else: self.retiring = True
        elif a == Action.WAIT:
            pass
        elif a == Action.DROP_LEGACY:
            if not self.expanded or self.dropped: valid = False
            else:
                self.db.execute("ALTER TABLE accounts DROP COLUMN balance")
                self.dropped = True
        elif a == Action.ROLLBACK_READS:
            if not self.read_new or self.dropped: valid = False
            else: self.read_new = False
        if not valid:
            reward -= 0.8
            self.invalid += 1
        before = self.errors
        if self.retiring and self.old_clients:
            self.old_clients -= 1
        self._traffic()
        reward -= 2 * (self.errors-before)
        verdict = None
        terminated = a == Action.SUBMIT or self.errors > 0
        if terminated:
            verdict = verify(self.db, self.ledger, old_clients=self.old_clients,
                             read_new=self.read_new, request_errors=self.errors)
            reward += 30 if verdict.success else -25
        truncated = self.steps >= self.max_steps and not terminated
        if truncated: reward -= 25
        reward += 0.98 * (0 if terminated or truncated else self._potential()) - potential_before
        self.done = bool(terminated or truncated)
        info = {"success": bool(verdict and verdict.success), "request_errors": self.errors,
                "invalid_actions": self.invalid, "action": a.name}
        if verdict: info["verdict"] = verdict.to_dict()
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return f"step={self.steps} missing={self._missing()} old_clients={self.old_clients} errors={self.errors}"

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
