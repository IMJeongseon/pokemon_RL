"""
Regulation I / J Team Builder
6 Pokemon BSS team: Koraidon / Calyrex-Shadow / Flutter Mane / Chien-Pao / Dondozo / Ting-Lu
"""

from poke_env.teambuilder import Teambuilder


class RegITeamBuilder(Teambuilder):
    """
    Regulation I BSS Team Builder.
    Based on the team defined in configs/REG J.yaml
    """

    def __init__(self):
        self.team = """
Koraidon @ Choice Band
Ability: Orichalcum Pulse
Level: 50
Tera Type: Fire
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Flare Blitz
- Close Combat
- Flame Charge
- U-turn

Calyrex-Shadow @ Choice Scarf
Ability: As One (Spectrier)
Level: 50
Tera Type: Fairy
EVs: 140 HP / 12 Def / 196 SpA / 4 SpD / 156 Spe
Timid Nature
IVs: 0 Atk
- Astral Barrage
- Trick
- Psychic
- Draining Kiss

Flutter Mane @ Choice Specs
Ability: Protosynthesis
Level: 50
Tera Type: Normal
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Moonblast
- Perish Song
- Shadow Ball
- Mystical Fire

Chien-Pao @ Focus Sash
Ability: Sword of Ruin
Level: 50
Tera Type: Electric
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Icicle Crash
- Swords Dance
- Sucker Punch
- Tera Blast

Dondozo (F) @ Covert Cloak
Ability: Unaware
Level: 50
Tera Type: Fairy
EVs: 212 HP / 252 Def / 44 SpD
Impish Nature
- Curse
- Rest
- Wave Crash
- Fissure

Ting-Lu @ Assault Vest
Ability: Vessel of Ruin
Level: 50
Tera Type: Fire
EVs: 36 HP / 252 Atk / 220 SpD
Adamant Nature
- Earthquake
- Ruination
- Heavy Slam
- Payback
"""
        self.parsed_team = self.join_team(self.parse_showdown_team(self.team))

    def yield_team(self):
        """Return the team in packed format"""
        return self.parsed_team
