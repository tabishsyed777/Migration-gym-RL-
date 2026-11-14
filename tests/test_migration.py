import sqlite3
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from migrationgym import MigrationEnv
from migrationgym.env import Action as A
from migrationgym.baselines import scripted
from migrationgym.verifier import verify
from migrationgym.agent import DQN

@pytest.fixture
def env():
    e=MigrationEnv();e.reset(seed=7)
    yield e
    e.close()

def finish(e, policy=scripted):
    while True:
        obs=e._obs()
        _,_,t,tr,info=e.step(int(policy(obs)))
        if t or tr:return info

def test_gym_contract():
    e=MigrationEnv();check_env(e,skip_render_check=True);e.close()

@pytest.mark.parametrize('stress',[False,True])
def test_scripted_migrates_under_traffic(stress):
    e=MigrationEnv(stress=stress)
    try:
        for seed in range(30):
            e.reset(seed=seed)
            assert finish(e)['success']
    finally:e.close()

def test_premature_submission(env):
    *_,info=env.step(A.SUBMIT)
    assert not info['success']
    assert 'target_column_missing' in info['verdict']['reasons']

def test_drop_before_drain_fails(env):
    env.step(A.EXPAND);env.step(A.DUAL_WRITE)
    _,_,terminated,_,info=env.step(A.DROP_LEGACY)
    assert terminated and not info['success'] and info['request_errors']>0

def test_read_before_backfill_fails(env):
    env.step(A.EXPAND)
    _,_,t,_,info=env.step(A.SWITCH_READS)
    assert t and info['request_errors']>0

def test_duplicate_action_penalized(env):
    env.step(A.EXPAND)
    *_,info=env.step(A.EXPAND)
    assert info['invalid_actions']==1

def test_timeout_is_truncation():
    e=MigrationEnv(max_steps=2);e.reset(seed=0)
    e.step(A.WAIT)
    _,_,t,tr,info=e.step(A.WAIT)
    assert tr and not t and not info['success']
    with pytest.raises(RuntimeError):e.step(A.WAIT)
    e.close()

def test_seed_reproducibility():
    a=MigrationEnv();b=MigrationEnv()
    try:
        oa,_=a.reset(seed=123);ob,_=b.reset(seed=123)
        assert np.array_equal(oa,ob)
        for action in [A.EXPAND,A.DUAL_WRITE,A.RETIRE_CLIENTS,A.BACKFILL_LARGE]:
            ra=a.step(action);rb=b.step(action)
            assert np.array_equal(ra[0],rb[0]) and ra[1:]==rb[1:]
            assert a.ledger==b.ledger
    finally:a.close();b.close()

def migrated_db():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE accounts(id INTEGER PRIMARY KEY, balance_cents INTEGER)')
    db.execute('INSERT INTO accounts VALUES(1,42)')
    return db

@pytest.mark.parametrize('sql',["UPDATE accounts SET balance_cents=0", "DELETE FROM accounts", "INSERT INTO accounts VALUES(2,42)"])
def test_verifier_rejects_data_tampering(sql):
    db=migrated_db();db.execute(sql)
    v=verify(db,{1:42},old_clients=0,read_new=True,request_errors=0)
    assert not v.success and v.mismatches>0
    db.close()

def test_verifier_rejects_suppressed_errors():
    db=migrated_db()
    assert not verify(db,{1:42},old_clients=0,read_new=True,request_errors=1).success
    db.close()

def test_verifier_requires_retired_clients():
    db=migrated_db()
    assert not verify(db,{1:42},old_clients=1,read_new=True,request_errors=0).success
    db.close()

def test_checkpoint_roundtrip(tmp_path):
    a=DQN(3);p=tmp_path/'policy.npz';a.save(p);b=DQN.load(p)
    assert np.array_equal(a.values(np.ones(12)),b.values(np.ones(12)))

def test_gradient_improves_terminal_target():
    a=DQN(5);s=np.ones((32,12))*.3;actions=np.zeros(32,dtype=int)
    target=np.ones(32)*5
    before=abs(a.values(s)[0,0]-5)
    for _ in range(100):a.learn((s,actions,target,s,np.ones(32)))
    assert abs(a.values(s)[0,0]-5)<before

def test_invalid_action(env):
    with pytest.raises(ValueError):env.step(100)

def test_backfill_then_legacy_write_is_detected(env):
    env.step(A.EXPAND)
    env.step(A.BACKFILL_LARGE)
    # Force a legacy-only update to a filled row; progress telemetry remains complete.
    env.db.execute('UPDATE accounts SET balance=balance+100 WHERE id=0')
    env.ledger[0]+=100
    _,_,t,_,info=env.step(A.SWITCH_READS)
    assert t and not info['success'] and info['request_errors']>0

def test_complete_flags_do_not_override_corrupted_data(env):
    info=finish(env)
    assert info['success']
    env.db.execute('UPDATE accounts SET balance_cents=0 WHERE id=0')
    verdict=verify(env.db,env.ledger,old_clients=env.old_clients,read_new=env.read_new,request_errors=env.errors)
    assert not verdict.success and 'ledger_mismatch' in verdict.reasons
