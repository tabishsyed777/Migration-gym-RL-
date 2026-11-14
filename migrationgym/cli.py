import argparse
import json
from pathlib import Path
import numpy as np
from .env import MigrationEnv, Action
from .agent import DQN, Replay
from .baselines import scripted

def evaluate(policy, episodes=100, seed=100000, stress=False):
    env = MigrationEnv(stress=stress)
    returns, successes, errors, lengths = [], [], [], []
    for i in range(episodes):
        obs,_ = env.reset(seed=seed+i)
        total=0
        while True:
            obs,r,t,tr,info=env.step(int(policy(obs)))
            total+=r
            if t or tr: break
        returns.append(total);successes.append(info['success']);errors.append(info['request_errors']);lengths.append(env.steps)
    env.close()
    # Wilson interval, explicitly an episode-level interval for this fixed policy.
    p=float(np.mean(successes));n=episodes;z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {"episodes":episodes,"success_rate":p,"success_95pct_wilson":[max(0.,center-half),min(1.,center+half)],
            "mean_return":float(np.mean(returns)),"mean_steps":float(np.mean(lengths)),
            "mean_request_errors":float(np.mean(errors))}

def train(args):
    env=MigrationEnv();agent=DQN(args.seed);replay=Replay()
    def reset_training(step, episode):
        env.stress = bool(args.mixed_workloads and agent.rng.random() < .5)
        obs, _ = env.reset(seed=args.seed+episode)
        probability = .9 * max(0, 1-step/(args.steps*.6))
        if agent.rng.random() < probability:
            # Curriculum changes initial states only; no expert transitions enter replay.
            depth = int(agent.rng.integers(1, 25))
            for _ in range(depth):
                action = scripted(obs)
                if action == Action.SUBMIT:
                    break
                obs, _, t, tr, _ = env.step(int(action))
                if t or tr:
                    raise RuntimeError('curriculum produced invalid initial state')
        # Remove expert-prefix elapsed time: each curriculum start gets a fresh horizon.
        env.steps = 0
        obs = env._obs()
        return obs
    obs=reset_training(0,0)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    history=[];episodes=0;best=-float('inf')
    for step in range(1,args.steps+1):
        eps=max(.08,1-step/(args.steps*.65))
        a=agent.act(obs,eps)
        ns,r,t,tr,info=env.step(a)
        # Finite horizon is part of the observed task, so timeouts are terminal for training.
        replay.add(obs,a,r,ns,t or tr)
        obs=ns
        if step>1000 and step%4==0:
            agent.learn(replay.sample(agent.rng))
        if t or tr:
            episodes+=1
            obs=reset_training(step,episodes)
        if step%10000==0 or step==args.steps:
            result=evaluate(agent.act,40,seed=50000)
            result.update(step=step,epsilon=eps,episodes_trained=episodes)
            history.append(result)
            if args.mixed_workloads:
                result['large_workload_validation'] = evaluate(agent.act,40,seed=60000,stress=True)
            score=(result['mean_return'] + result.get('large_workload_validation',result)['mean_return'])/2
            if score>best:
                best=score;agent.save(out/'policy.npz')
            print(json.dumps(result),flush=True)
    agent.save(out/'last_policy.npz')
    (out/'training.json').write_text(json.dumps({"seed":args.seed,"steps":args.steps,"mixed_workloads":args.mixed_workloads,"algorithm":"Double DQN", "learning_rate":0.0007,"target_period":250, "curriculum":"safe-prefix resets with probability .9*(1-step/(.6*budget)), clipped at zero; prefix depth sampled 1..24; fresh clock after prefix; no expert replay transitions","selection":"best average return on standard validation seeds 50000..50039 and, when mixed, larger-workload seeds 60000..60039","history":history},indent=2))
    env.close()

def benchmark(args):
    agent=DQN.load(args.model);rng=np.random.default_rng(907)
    results={"protocol":{"standard_seed_start":args.seed_base,"larger_seed_start":args.seed_base+100000,"episodes_per_policy_per_suite":args.episodes,"model":str(args.model)}}
    for stress in (False,True):
        results['stress' if stress else 'standard']={name:evaluate(fn,args.episodes,args.seed_base if not stress else args.seed_base+100000,stress) for name,fn in [('random',lambda o:int(rng.integers(10))),('scripted',scripted),('double_dqn',agent.act)]}
    text=json.dumps(results,indent=2)
    Path(args.output).write_text(text)
    print(text)

def demo(args):
    agent=DQN.load(args.model);env=MigrationEnv(stress=args.stress)
    obs,_=env.reset(seed=args.seed)
    print('SQLite live migration: balance -> balance_cents')
    print(env.render())
    while True:
        a=agent.act(obs)
        obs,r,t,tr,info=env.step(a)
        print(f'{Action(a).name:20} reward={r:7.2f} {env.render()}')
        if t or tr:
            print(json.dumps(info,indent=2));break
    env.close()

def main():
    p=argparse.ArgumentParser(description='MigrationGym: train, benchmark, or inspect a migration')
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--steps',type=int,default=160000);t.add_argument('--seed',type=int,default=7);t.add_argument('--output',default='artifacts');t.add_argument('--mixed-workloads',action='store_true');t.set_defaults(fn=train)
    b=sub.add_parser('benchmark');b.add_argument('--model',default='artifacts/policy.npz');b.add_argument('--episodes',type=int,default=200);b.add_argument('--output',default='artifacts/benchmark.json');b.add_argument('--seed-base',type=int,default=300000);b.set_defaults(fn=benchmark)
    d=sub.add_parser('demo');d.add_argument('--model',default='artifacts/policy.npz');d.add_argument('--seed',type=int,default=100001);d.add_argument('--stress',action='store_true');d.set_defaults(fn=demo)
    args=p.parse_args()
    if hasattr(args,'episodes') and args.episodes<1: p.error('--episodes must be positive')
    if hasattr(args,'steps') and args.steps<1:p.error('--steps must be positive')
    args.fn(args)

if __name__=='__main__':main()
