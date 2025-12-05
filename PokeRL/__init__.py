"""
Regulation I BSS - Masked PPO Training

This package implements a reinforcement learning agent for Pokémon Showdown
Battle Stadium Singles (BSS) Regulation I/J using Masked PPO.
"""

from .reg_i_player import RegIPlayer
from .reg_i_team_builder import RegITeamBuilder

__all__ = ["RegIPlayer", "RegITeamBuilder"]
