"""MigrationGym: a bounded migration control benchmark."""
from gymnasium.envs.registration import register
from .env import MigrationEnv
register(id="MigrationGym-v0", entry_point="migrationgym:MigrationEnv")
__all__ = ["MigrationEnv"]
