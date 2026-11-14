"""Outcome checks independent of policy claims and migration progress flags."""
from dataclasses import dataclass, asdict
import sqlite3

@dataclass(frozen=True)
class Verdict:
    success: bool
    reasons: tuple[str, ...]
    mismatches: int

    def to_dict(self):
        return asdict(self)


def verify(db: sqlite3.Connection, ledger: dict[int, int], *, old_clients: int,
           read_new: bool, request_errors: int) -> Verdict:
    columns = {r[1] for r in db.execute("PRAGMA table_info(accounts)")}
    reasons = []
    if "balance_cents" not in columns:
        reasons.append("target_column_missing")
        actual = {}
    else:
        actual = dict(db.execute("SELECT id, balance_cents FROM accounts"))
    mismatches = sum(actual.get(k) != v for k, v in ledger.items())
    mismatches += len(set(actual) - set(ledger))
    if mismatches:
        reasons.append("ledger_mismatch")
    if "balance" in columns:
        reasons.append("legacy_column_present")
    if old_clients:
        reasons.append("legacy_clients_active")
    if not read_new:
        reasons.append("new_reads_disabled")
    if request_errors:
        reasons.append("request_failures")
    return Verdict(not reasons, tuple(reasons), mismatches)
