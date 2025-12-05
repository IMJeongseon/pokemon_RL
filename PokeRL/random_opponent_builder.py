"""
Random Team Builder for opponent training.
Selects randomly from a pool of pre-built competitive teams.
"""

import numpy as np
from poke_env.teambuilder import Teambuilder


class RandomOpponentTeamBuilder(Teambuilder):
    """
    Opponent team builder that randomly selects from a pool of competitive teams.

    This ensures the training agent faces diverse team compositions
    while opponent teams remain competitive.
    """

    def __init__(self):
        # Import teams from team1.py
        from team.team1 import team1, team2, team3

        # Store raw team strings for debugging
        self.raw_teams = [team1, team2, team3]

        # Parse and store all teams
        self.teams = [
            self.join_team(self.parse_showdown_team(team1)),
            self.join_team(self.parse_showdown_team(team2)),
            self.join_team(self.parse_showdown_team(team3)),
        ]

        self.team_count = len(self.teams)
        self.call_count = 0
        print(f"[RandomOpponentTeamBuilder] Loaded {self.team_count} teams for opponent pool")

        # Debug: print first pokemon of each team
        for i, raw_team in enumerate(self.raw_teams):
            first_line = raw_team.strip().split('\n')[0].strip()
            print(f"  Team {i+1}: {first_line}")

    def yield_team(self):
        """
        Return a random team from the pool.
        Called each time a new battle is created.
        """
        self.call_count += 1
        team_idx = np.random.randint(0, self.team_count)
        selected_team = self.teams[team_idx]

        # Debug log (overwrite mode for cleaner output)
        first_pokemon = self.raw_teams[team_idx].strip().split('\n')[0].strip()
        log_msg = f"[Battle #{self.call_count}] Team {team_idx+1}: {first_pokemon}"
        # Use \r to overwrite the same line
        print(f"\r{log_msg}", end='', flush=True)

        # Write to file for debugging
        with open("/tmp/opponent_team_selection.log", "a") as f:
            f.write(log_msg + "\n")

        return selected_team
