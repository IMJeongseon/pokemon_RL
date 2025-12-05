"""
Custom SinglesEnv that allows different teams for player and opponent.

This extends RegIPlayer (SinglesEnv) to support dual teams:
- agent1 (player): Uses team1
- agent2 (opponent): Uses team2
"""

from reg_i_player import RegIPlayer
from poke_env.teambuilder import Teambuilder
from typing import Union, Optional


class DualTeamRegIPlayer(RegIPlayer):
    """
    Extended RegIPlayer that allows different teams for agent1 and agent2.

    This is a workaround for the limitation that PokeEnv passes the same
    team to both agents by default.
    """

    def __init__(
        self,
        team1: Union[str, Teambuilder],
        team2: Union[str, Teambuilder],
        **kwargs
    ):
        """
        Args:
            team1: Team for agent1 (player being trained)
            team2: Team for agent2 (opponent)
            **kwargs: Other arguments passed to RegIPlayer/SinglesEnv
        """
        # Initialize with team1
        super().__init__(team=team1, **kwargs)

        # Replace agent2's team with team2
        # This works because _team is set in Player.__init__
        if isinstance(team2, str):
            from poke_env.teambuilder import ConstantTeambuilder
            self.agent2._team = ConstantTeambuilder(team2)
        else:
            self.agent2._team = team2

        print(f"[DualTeam] agent1 team: {type(self.agent1._team).__name__}")
        print(f"[DualTeam] agent2 team: {type(self.agent2._team).__name__}")
