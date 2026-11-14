from .env import Action

def scripted(obs):
    expanded, dual, coverage, reads, clients, retiring, dropped = obs[:7]
    if not expanded: return Action.EXPAND
    if not dual: return Action.DUAL_WRITE
    if not retiring: return Action.RETIRE_CLIENTS
    if coverage < 0.999: return Action.BACKFILL_SMALL
    if not reads: return Action.SWITCH_READS
    if clients > 0: return Action.WAIT
    if not dropped: return Action.DROP_LEGACY
    return Action.SUBMIT
