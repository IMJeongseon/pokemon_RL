"""
Regulation I BSS Player with SAR (State-Action-Reward) system integration.

This player implements:
- State: Simplified version of SAR/state.py
- Action: 10 discrete actions from SAR/action.py (4 moves + 4 tera moves + 2 switches)
- Reward: Progressive implementation of SAR/reward_function.py
- Action Masking: For Masked PPO
"""

import logging
import os
import sys
import numpy as np
from gymnasium.spaces import Box, Discrete
from typing import Optional, Dict, List
from poke_env.environment import SinglesEnv
from poke_env.player import Player
from poke_env.battle import AbstractBattle, Pokemon
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.effect import Effect

# Ensure SAR package is importable when running from reg_i_rl
SAR_PATH = os.path.join(os.path.dirname(__file__), "..", "SAR")
if SAR_PATH not in sys.path:
    sys.path.insert(0, SAR_PATH)

from sar_adapter import BattleStateAdapter, TurnInfoAdapter
from reward_function import BETA_STATUS, BETA_STAGE


class RegIPlayer(SinglesEnv):
    """
    Regulation I BSS Player for Masked PPO training.

    Team: Koraidon / Calyrex-Shadow / Flutter Mane / Chien-Pao / Dondozo / Ting-Lu
    Battle Format: gen9bssregj (BSS 3v3, Regulation J)
    """

    # Action space constants (from SAR/action.py)
    NUM_MOVES = 4
    NUM_BENCH = 2  # BSS 3v3: 1 active + 2 bench
    ACTION_SPACE_SIZE = 10  # 4 moves + 4 tera moves + 2 switches

    # Pokemon attack type classification (based on moveset)
    # Physical pokemon: reward only Attack boosts
    # Special pokemon: reward only Special Attack boosts
    POKEMON_ATTACK_TYPE = {
        "koraidon": "physical",      # Flare Blitz, Close Combat, Flame Charge, U-turn
        "calyrexshadow": "special",  # Astral Barrage, Psychic, Draining Kiss
        "fluttermane": "special",    # Moonblast, Shadow Ball, Mystical Fire
        "chienpao": "physical",      # Icicle Crash, Sucker Punch, Swords Dance
        "dondozo": "physical",       # Wave Crash, Fissure, Body Press
        "tinglu": "physical",        # Earthquake, Heavy Slam, Payback (Ruination은 특수지만 주력은 물리)
    }

    WEATHER_TRACKED = [
        Weather.SUNNYDAY,
        Weather.RAINDANCE,
        Weather.SANDSTORM,
        Weather.HAIL,
        Weather.SNOW,
        Weather.DESOLATELAND,
        Weather.PRIMORDIALSEA,
        Weather.DELTASTREAM,
    ]
    TERRAIN_TRACKED = [
        Field.ELECTRIC_TERRAIN,
        Field.GRASSY_TERRAIN,
        Field.MISTY_TERRAIN,
        Field.PSYCHIC_TERRAIN,
    ]
    WEATHER_SUPPRESS_ABILITIES = {"airlock", "cloudnine"}
    CHOICE_ITEMS = {"choiceband", "choicespecs", "choicescarf"}
    CHOICE_LOCK_EFFECT = Effect.LOCKED_MOVE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Track battle state for reward calculation
        self._reward_buffer: Dict[AbstractBattle, float] = {}
        self._prev_hp_fractions: Dict[AbstractBattle, Dict[str, float]] = {}
        self._prev_boosts: Dict[AbstractBattle, dict] = {}  # Track stat stage history for diminishing returns
        self._turn_info: Dict[AbstractBattle, dict] = {}
        self._state_adapter = BattleStateAdapter()
        self._turn_info_adapter = TurnInfoAdapter()
        self._beta_status = BETA_STATUS
        self._beta_stage = BETA_STAGE
        self._logger = logging.getLogger(self.__class__.__name__)

        # Override action spaces with our custom 10-action space
        # (SinglesEnv sets it to 26 for gen9 by default)
        act_space = Discrete(self.ACTION_SPACE_SIZE)
        self.action_spaces = {
            agent: act_space for agent in self.possible_agents
        }

        # Define observation spaces for both agents (required by PettingZoo)
        obs_space = self.describe_embedding()
        self.observation_spaces = {
            agent: obs_space for agent in self.possible_agents
        }

    # ============================================================
    # STATE REPRESENTATION (from SAR/state.py)
    # ============================================================

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Convert battle state to observation vector.

        Simplified version of SAR/state.py's BattleState.to_vector()

        Features:
        - Active pokemon info (mine + opponent): species, HP, status, stats
        - Team info: alive count, fainted
        - Move info: base power, type effectiveness, accuracy
        - Battle flow: turn, tera used
        - Stat stages (boosts)
        """
        features = []

        # ===== MY ACTIVE POKEMON =====
        if battle.active_pokemon:
            features.extend(self._encode_pokemon(battle.active_pokemon, battle))
        else:
            features.extend(self._encode_empty_pokemon())

        # ===== OPPONENT ACTIVE POKEMON =====
        if battle.opponent_active_pokemon:
            features.extend(self._encode_pokemon(battle.opponent_active_pokemon, battle, is_opponent=True))
        else:
            features.extend(self._encode_empty_pokemon())

        # ===== MOVE INFORMATION =====
        features.extend(self._encode_moves(battle))

        # ===== TEAM INFORMATION =====
        features.extend(self._encode_team_info(battle))

        # ===== BENCH INFORMATION =====
        features.extend(self._encode_bench_info(battle))

        # ===== HAZARD INFORMATION =====
        features.extend(self._encode_hazard_flags(battle))

        # ===== BATTLE FLOW =====
        features.extend(self._encode_battle_flow(battle))

        return np.array(features, dtype=np.float32)

    def _encode_pokemon(self, pokemon: Pokemon, battle: AbstractBattle, is_opponent: bool = False) -> List[float]:
        """Encode a single pokemon's state."""
        features = []

        # HP fraction
        features.append(pokemon.current_hp_fraction)

        # Status (one-hot: none, burn, paralysis, sleep, poison, freeze, toxic)
        status_vec = [0.0] * 7
        if pokemon.status is None:
            status_vec[0] = 1.0
        elif pokemon.status.name == "BRN":
            status_vec[1] = 1.0
        elif pokemon.status.name == "PAR":
            status_vec[2] = 1.0
        elif pokemon.status.name == "SLP":
            status_vec[3] = 1.0
        elif pokemon.status.name == "PSN":
            status_vec[4] = 1.0
        elif pokemon.status.name == "FRZ":
            status_vec[5] = 1.0
        elif pokemon.status.name == "TOX":
            status_vec[6] = 1.0
        features.extend(status_vec)

        # Base stats (normalized by 255)
        base_stats = pokemon.base_stats
        features.append(base_stats['hp'] / 255.0)
        features.append(base_stats['atk'] / 255.0)
        features.append(base_stats['def'] / 255.0)
        features.append(base_stats['spa'] / 255.0)
        features.append(base_stats['spd'] / 255.0)
        features.append(base_stats['spe'] / 255.0)

        # Stat boosts (normalized by 6: -6 to +6 -> -1 to +1)
        boosts = pokemon.boosts
        features.append(boosts.get('atk', 0) / 6.0)
        features.append(boosts.get('def', 0) / 6.0)
        features.append(boosts.get('spa', 0) / 6.0)
        features.append(boosts.get('spd', 0) / 6.0)
        features.append(boosts.get('spe', 0) / 6.0)
        features.append(boosts.get('accuracy', 0) / 6.0)
        features.append(boosts.get('evasion', 0) / 6.0)

        # Is fainted
        features.append(1.0 if pokemon.fainted else 0.0)

        # Tera info
        if not is_opponent:
            features.append(1.0 if battle.can_tera else 0.0)
            features.append(1.0 if pokemon.is_terastallized else 0.0)
        else:
            # opponent_can_tera doesn't exist, use "not used_tera" instead
            features.append(1.0 if not battle.opponent_used_tera else 0.0)
            features.append(1.0 if pokemon.is_terastallized else 0.0)

        # Choice item indicator
        item_name = (pokemon.item or "").replace(" ", "").lower()
        features.append(1.0 if item_name in self.CHOICE_ITEMS else 0.0)

        # Choice lock / move lock status
        effects = getattr(pokemon, "effects", {})
        is_locked = 1.0 if effects and self.CHOICE_LOCK_EFFECT in effects else 0.0
        features.append(is_locked)

        # Pending Yawn (will fall asleep next turn if not switched)
        is_yawned = 1.0 if effects and Effect.YAWN in effects else 0.0
        features.append(is_yawned)

        # Perish Song count (0-3, normalized by 3)
        # When Perish Song is active, counter starts at 3 and decrements each turn
        # Pokemon faints when counter reaches 0
        perish_count = 0.0
        if effects and Effect.PERISH3 in effects:
            perish_count = 3.0 / 3.0
        elif effects and Effect.PERISH2 in effects:
            perish_count = 2.0 / 3.0
        elif effects and Effect.PERISH1 in effects:
            perish_count = 1.0 / 3.0
        elif effects and Effect.PERISH0 in effects:
            perish_count = 0.0  # About to faint
        features.append(perish_count)

        return features

    def _encode_empty_pokemon(self) -> List[float]:
        """Encode empty pokemon slot."""
        # 1 (HP) + 7 (status) + 6 (base stats) + 7 (boosts) + 1 (fainted) + 2 (tera) + 2 (choice info) + 1 (yawn) + 1 (perish) = 28
        return [0.0] * 28

    def _encode_moves(self, battle: AbstractBattle) -> List[float]:
        """Encode move information."""
        features = []

        # For each of the 4 move slots
        for i in range(4):
            if i < len(battle.available_moves):
                move = battle.available_moves[i]

                # Base power (normalized by 150)
                features.append(move.base_power / 150.0 if move.base_power else 0.0)

                # Type effectiveness
                if battle.opponent_active_pokemon:
                    effectiveness = move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                        type_chart=battle.opponent_active_pokemon._data.type_chart,
                    )
                    features.append(effectiveness)
                else:
                    features.append(1.0)

                # Accuracy (True -> 1.0, otherwise actual value / 100)
                if move.accuracy is True:
                    features.append(1.0)
                elif move.accuracy:
                    features.append(move.accuracy / 100.0)
                else:
                    features.append(0.0)

                # Current PP / Max PP
                features.append(move.current_pp / move.max_pp if move.max_pp else 0.0)
            else:
                # Empty move slot
                features.extend([0.0, 0.0, 0.0, 0.0])

        return features

    def _encode_team_info(self, battle: AbstractBattle) -> List[float]:
        """Encode team composition and status."""
        features = []

        # My team
        my_alive = len([p for p in battle.team.values() if not p.fainted])
        my_fainted = 3 - my_alive
        features.append(my_alive / 3.0)
        features.append(my_fainted / 3.0)

        # Opponent team
        opp_alive = len([p for p in battle.opponent_team.values() if not p.fainted])
        opp_fainted = 3 - opp_alive
        features.append(opp_alive / 3.0)
        features.append(opp_fainted / 3.0)

        # Alive difference
        features.append((my_alive - opp_alive) / 3.0)

        return features

    def _encode_bench_info(self, battle: AbstractBattle) -> List[float]:
        """Encode bench pokemon (up to 2) for both teams."""
        my_bench = self._encode_bench_team(battle.team, battle.active_pokemon)
        opp_bench = self._encode_bench_team(battle.opponent_team, battle.opponent_active_pokemon)
        return my_bench + opp_bench

    def _encode_bench_team(self, team: Dict[str, Pokemon], active: Optional[Pokemon]) -> List[float]:
        bench_slots = []
        for mon in team.values():
            if mon is None or mon is active:
                continue
            bench_slots.append(mon)
        bench_slots = bench_slots[:2]
        while len(bench_slots) < 2:
            bench_slots.append(None)
        encoded: List[float] = []
        for mon in bench_slots:
            encoded.extend(self._encode_bench_slot(mon))
        return encoded

    def _encode_bench_slot(self, pokemon: Optional[Pokemon]) -> List[float]:
        status_vec = [0.0] * 7
        if pokemon is None:
            status_vec[0] = 1.0
            return [0.0, 1.0, *status_vec]

        hp = pokemon.current_hp_fraction
        fainted = 1.0 if pokemon.fainted else 0.0
        if pokemon.status is None:
            status_vec[0] = 1.0
        else:
            name = pokemon.status.name
            if name == "BRN":
                status_vec[1] = 1.0
            elif name == "PAR":
                status_vec[2] = 1.0
            elif name == "SLP":
                status_vec[3] = 1.0
            elif name == "PSN":
                status_vec[4] = 1.0
            elif name == "FRZ":
                status_vec[5] = 1.0
            elif name == "TOX":
                status_vec[6] = 1.0
        return [hp, fainted, *status_vec]

    def _encode_hazard_flags(self, battle: AbstractBattle) -> List[float]:
        opp_conditions = getattr(battle, "opponent_side_conditions", {}) or {}
        my_conditions = getattr(battle, "side_conditions", {}) or {}

        def _has(condition_map, side_condition):
            return 1.0 if side_condition in condition_map else 0.0

        opp_features = [
            _has(opp_conditions, SideCondition.STEALTH_ROCK),
            _has(opp_conditions, SideCondition.SPIKES),
            _has(opp_conditions, SideCondition.TOXIC_SPIKES),
            _has(opp_conditions, SideCondition.STICKY_WEB),
        ]
        my_features = [
            _has(my_conditions, SideCondition.STEALTH_ROCK),
            _has(my_conditions, SideCondition.SPIKES),
            _has(my_conditions, SideCondition.TOXIC_SPIKES),
            _has(my_conditions, SideCondition.STICKY_WEB),
        ]
        return opp_features + my_features

    def _encode_battle_flow(self, battle: AbstractBattle) -> List[float]:
        """Encode battle-level information."""
        features = []

        # Turn number (normalized by 20, typical max turns)
        features.append(min(battle.turn, 40) / 40.0)

        # Tera availability
        features.append(1.0 if battle.can_tera else 0.0)
        features.append(1.0 if not battle.opponent_used_tera else 0.0)

        # Force switch
        features.append(1.0 if battle.force_switch else 0.0)

        # Weather / terrain / fields / hazards
        features.extend(self._encode_weather_features(battle))
        features.extend(self._encode_terrain_features(battle))
        features.extend(self._encode_field_flags(battle))
        features.extend(self._encode_side_conditions(battle))

        return features

    def _encode_weather_features(self, battle: AbstractBattle) -> List[float]:
        """One-hot weather encoding + duration + suppression flag."""
        size = len(self.WEATHER_TRACKED) + 1  # extra slot for "none/other"
        weather_vec = [0.0] * size
        duration = 0.0

        if battle.weather:
            weather_enum, start_turn = next(iter(battle.weather.items()))
            idx = 0
            for i, tracked in enumerate(self.WEATHER_TRACKED, start=1):
                if weather_enum == tracked:
                    idx = i
                    break
            weather_vec[idx] = 1.0
            duration = min(8.0, max(0.0, battle.turn - start_turn + 1)) / 8.0
        else:
            weather_vec[0] = 1.0

        suppressed = 1.0 if battle.weather and self._is_weather_suppressed(battle) else 0.0
        return weather_vec + [duration, suppressed]

    def _encode_terrain_features(self, battle: AbstractBattle) -> List[float]:
        """Encode electric/grassy/misty/psychic terrain with duration."""
        size = len(self.TERRAIN_TRACKED) + 1
        terrain_vec = [0.0] * size
        duration = 0.0

        terrain_field = None
        if battle.fields:
            for field, start_turn in battle.fields.items():
                if field in self.TERRAIN_TRACKED:
                    terrain_field = (field, start_turn)
                    break

        if terrain_field:
            field, start_turn = terrain_field
            idx = self.TERRAIN_TRACKED.index(field) + 1
            terrain_vec[idx] = 1.0
            duration = min(8.0, max(0.0, battle.turn - start_turn + 1)) / 8.0
        else:
            terrain_vec[0] = 1.0

        return terrain_vec + [duration]

    def _encode_field_flags(self, battle: AbstractBattle) -> List[float]:
        """Binary flags for room-like field states."""
        fields = battle.fields or {}
        return [
            1.0 if Field.TRICK_ROOM in fields else 0.0,
            1.0 if Field.GRAVITY in fields else 0.0,
            1.0 if Field.MAGIC_ROOM in fields else 0.0,
            1.0 if Field.WONDER_ROOM in fields else 0.0,
        ]

    def _encode_side_conditions(self, battle: AbstractBattle) -> List[float]:
        """Encode hazards/screens for both sides."""
        my_conditions = battle.side_conditions or {}
        opp_conditions = battle.opponent_side_conditions or {}
        return (
            self._encode_single_side_conditions(my_conditions)
            + self._encode_single_side_conditions(opp_conditions)
        )

    def _encode_single_side_conditions(self, side_conditions: Dict[SideCondition, int]) -> List[float]:
        """Normalize hazard/screen info for one side."""
        def _has(condition: SideCondition) -> float:
            return 1.0 if condition in side_conditions else 0.0

        def _layers(condition: SideCondition, max_layers: int) -> float:
            return min(float(side_conditions.get(condition, 0)), float(max_layers)) / float(max_layers)

        return [
            _has(SideCondition.STEALTH_ROCK),
            _layers(SideCondition.SPIKES, 3),
            _layers(SideCondition.TOXIC_SPIKES, 2),
            _has(SideCondition.STICKY_WEB),
            _has(SideCondition.TAILWIND),
            _has(SideCondition.AURORA_VEIL),
            _has(SideCondition.REFLECT),
            _has(SideCondition.LIGHT_SCREEN),
            _has(SideCondition.SAFEGUARD),
        ]

    def _is_weather_suppressed(self, battle: AbstractBattle) -> bool:
        """Detect Air Lock / Cloud Nine style suppression."""
        for mon in (battle.active_pokemon, battle.opponent_active_pokemon):
            if not mon:
                continue
            ability = getattr(mon, "ability", None)
            if ability and self._normalize_ability_name(ability) in self.WEATHER_SUPPRESS_ABILITIES:
                return True
        return False

    @staticmethod
    def _normalize_ability_name(ability: Optional[str]) -> str:
        if not ability:
            return ""
        return ability.replace("-", "").replace(" ", "").lower()

    def describe_embedding(self) -> Box:
        """
        Define the observation space.

        Total dimensions:
        - My active: 28 (added perish song count)
        - Opp active: 28 (added perish song count)
        - Moves: 16 (4 moves * 4 features)
        - Team info: 5
        - Bench info: 36 (2 bench per side × 9 features)
        - Hazard flags: 8 (opponent/my Stealth Rock, Spikes, Toxic Spikes, Sticky Web)
        - Battle flow + field state: 43
        Total: 164 dimensions
        """
        return Box(
            low=-1.0,
            high=4.0,
            shape=(164,),
            dtype=np.float32
        )

    # ============================================================
    # ACTION SPACE & MASKING (from SAR/action.py)
    # ============================================================

    def action_space(self) -> Discrete:
        """
        Define discrete action space.

        10 actions:
        - 0-3: Use move 0-3 (no tera)
        - 4-7: Use move 0-3 with tera
        - 8-9: Switch to bench pokemon 0-1
        """
        return Discrete(self.ACTION_SPACE_SIZE)

    def _get_action_mask_for_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Internal method to compute action mask for a specific battle.

        IMPORTANT: Actions 0-3 and 4-7 correspond to MOVE SLOTS, not available_moves indices.

        Args:
            battle: AbstractBattle to compute mask for

        Returns:
            np.ndarray: Binary mask of shape (10,), where 1 = valid, 0 = invalid
        """
        mask = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)

        if battle.active_pokemon is None:
            # No active pokemon, allow switches only
            for i in range(min(len(battle.available_switches), 2)):
                mask[8 + i] = 1.0
            return mask

        # Get move slots (always 4 slots, even if some are empty/unusable)
        active_moves = battle.active_pokemon.moves

        # Build a mapping: slot_id -> Move object (only for available moves)
        available_move_ids = {move.id for move in battle.available_moves}

        # Check opponent's field hazards to avoid redundant setup moves
        opp_conditions = getattr(battle, "opponent_side_conditions", {}) or {}

        # Check opponent's status condition to avoid redundant status moves
        opp_has_status = False
        if battle.opponent_active_pokemon:
            opp_status = battle.opponent_active_pokemon.status
            if opp_status is not None:
                opp_has_status = True

        # ===== MOVES (0-3) =====
        if not battle.force_switch:
            for slot_idx, (move_id, move) in enumerate(active_moves.items()):
                if slot_idx >= 4:
                    break
                # Check if this slot's move is in available_moves
                if move.id in available_move_ids:
                    # Check if this is a hazard move that's already set up
                    move_id_lower = move.id.lower()
                    should_mask = False

                    # === HAZARD MOVES ===
                    # Stealth Rock: only once
                    if move_id_lower == "stealthrock":
                        if SideCondition.STEALTH_ROCK in opp_conditions:
                            should_mask = True

                    # Spikes: maximum 3 layers
                    elif move_id_lower == "spikes":
                        if SideCondition.SPIKES in opp_conditions:
                            spikes_layers = opp_conditions[SideCondition.SPIKES]
                            if spikes_layers >= 3:
                                should_mask = True

                    # Toxic Spikes: maximum 2 layers
                    elif move_id_lower == "toxicspikes":
                        if SideCondition.TOXIC_SPIKES in opp_conditions:
                            toxic_layers = opp_conditions[SideCondition.TOXIC_SPIKES]
                            if toxic_layers >= 2:
                                should_mask = True

                    # Sticky Web: only once
                    elif move_id_lower == "stickyweb":
                        if SideCondition.STICKY_WEB in opp_conditions:
                            should_mask = True

                    # === STATUS MOVES ===
                    # If opponent already has a status condition, mask status-inducing moves
                    if opp_has_status:
                        # Common status moves
                        status_moves = {
                            # Paralysis
                            "thunderwave", "stunspore", "glare", "nuzzle", "zapcannon",
                            # Burn
                            "willowisp", "sacredfire",
                            # Sleep
                            "sleeppowder", "spore", "hypnosis", "lovelykiss", "darkvoid", "yawn",
                            # Poison/Toxic
                            "toxic", "poisonpowder", "poisongas", "toxicthread",
                            # Freeze (rare in competitive)
                            "powder",
                        }

                        if move_id_lower in status_moves:
                            should_mask = True

                        # Also check move object's status property if available
                        # Some moves have guaranteed status effects
                        if hasattr(move, 'status') and move.status is not None:
                            should_mask = True

                    # If not masked, enable this move
                    if not should_mask:
                        mask[slot_idx] = 1.0

                        # ===== TERA MOVES (4-7) =====
                        if battle.can_tera:
                            mask[4 + slot_idx] = 1.0

        # ===== SWITCHES (8-9) =====
        for i in range(min(len(battle.available_switches), 2)):
            mask[8 + i] = 1.0

        # Ensure at least one action is available
        if mask.sum() == 0:
            # Fallback strategy (quiet mode - this is expected during transitions)
            # 1. If switches are available, use them
            if len(battle.available_switches) > 0:
                mask[8] = 1.0
            # 2. If moves are available, find first valid slot
            elif len(battle.available_moves) > 0 and battle.active_pokemon:
                for slot_idx, (move_id, move) in enumerate(active_moves.items()):
                    if slot_idx >= 4:
                        break
                    if move.id in available_move_ids:
                        mask[slot_idx] = 1.0
                        break
            # 3. Last resort: enable basic moves (battle in transition)
            else:
                mask[0:4] = 1.0

        return mask

    def get_action_mask(self) -> np.ndarray:
        """
        Compute action mask for current battle.

        This is the public interface for sb3_contrib.ActionMasker.
        It uses self.battle1 (from PokeEnv) to get the current battle state.

        Returns:
            np.ndarray: Binary mask of shape (10,), where 1 = valid, 0 = invalid
        """
        battle = getattr(self, 'battle1', None)

        # No battle or battle finished
        if battle is None or battle.finished:
            # Return default mask (first 4 moves only)
            mask = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)
            mask[0:4] = 1.0  # Enable basic moves
            return mask

        # Battle is in transition state (async timing issue)
        # This happens between turns when battle state is being updated
        if (len(battle.available_moves) == 0 and
            len(battle.available_switches) == 0 and
            battle.turn > 0 and
            not battle.finished):
            # Cache the last valid mask or return a safe default
            if hasattr(self, '_last_valid_mask'):
                return self._last_valid_mask.copy()
            else:
                # Safe default: allow basic moves only
                mask = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)
                mask[0:4] = 1.0
                return mask

        # Normal case: compute mask
        mask = self._get_action_mask_for_battle(battle)

        # Cache this mask for transition states
        if mask.sum() > 0:
            self._last_valid_mask = mask.copy()

        return mask

    # ============================================================
    # REWARD COMPUTATION (from SAR/reward_function.py)
    # ============================================================

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculate reward for the current battle state.
        This is the interface required by PokeEnv.
        """
        return self.compute_reward(battle)

    @staticmethod
    def _get_stage_reward_weight(current_stage: int) -> float:
        """
        Calculate diminishing returns weight for stat stage changes.

        Stat boosts have diminishing marginal value:
        - Stage 0 → +2: High value (100% → 200% damage)
        - Stage +2 → +4: Medium value (200% → 300% damage, +50%)
        - Stage +4 → +6: Low value (300% → 400% damage, +33%)

        Args:
            current_stage: Current stat stage (-6 to +6)

        Returns:
            Weight multiplier (0.0 to 1.0)
        """
        # At stage limits (±6), further boosts have no value
        if abs(current_stage) >= 6:
            return 0.0

        # Linear decay: weight decreases as |stage| increases
        # Stage 0: weight = 1.0
        # Stage ±3: weight = 0.5
        # Stage ±6: weight = 0.0
        abs_stage = abs(current_stage)
        weight = max(0.0, 1.0 - abs_stage / 6.0)
        return weight

    def _get_pokemon_attack_type(self, pokemon: Pokemon) -> str:
        """
        Get attack type (physical/special) for a pokemon.

        Args:
            pokemon: Pokemon object

        Returns:
            "physical" or "special"
        """
        species = pokemon.species.lower().replace("-", "")
        return self.POKEMON_ATTACK_TYPE.get(species, "physical")  # Default to physical

    def _compute_stat_stage_reward(self, battle: AbstractBattle) -> float:
        """
        Compute reward for stat stage changes with diminishing returns.

        Only rewards relevant stat boosts:
        - Physical attackers: only Attack boosts
        - Special attackers: only Special Attack boosts

        Uses diminishing returns: higher current stage = lower reward for further boosts.

        Args:
            battle: Current battle state

        Returns:
            Stat stage reward
        """
        # Initialize previous boosts tracking
        if battle not in self._prev_boosts:
            self._prev_boosts[battle] = {}
            # Initialize with current boosts
            if battle.active_pokemon:
                self._prev_boosts[battle]["my_active"] = dict(battle.active_pokemon.boosts)
            if battle.opponent_active_pokemon:
                self._prev_boosts[battle]["opp_active"] = dict(battle.opponent_active_pokemon.boosts)
            return 0.0  # No reward on first observation

        reward = 0.0

        # === MY POKEMON STAT CHANGES ===
        if battle.active_pokemon:
            species = battle.active_pokemon.species
            attack_type = self._get_pokemon_attack_type(battle.active_pokemon)
            current_boosts = battle.active_pokemon.boosts
            prev_boosts = self._prev_boosts[battle].get("my_active", {})

            # Determine which stat to track
            relevant_stat = "spa" if attack_type == "special" else "atk"

            current_stage = current_boosts.get(relevant_stat, 0)
            prev_stage = prev_boosts.get(relevant_stat, 0)
            stage_change = current_stage - prev_stage

            if stage_change != 0:
                # Apply diminishing returns based on PREVIOUS stage (before the boost)
                weight = self._get_stage_reward_weight(prev_stage)
                stage_reward = self._beta_stage * stage_change * weight
                reward += stage_reward

                if self._logger.level <= logging.DEBUG:
                    self._logger.debug(
                        f"{species} ({attack_type}): {relevant_stat} {prev_stage:+d} → {current_stage:+d} "
                        f"(weight={weight:.2f}, reward={stage_reward:+.3f})"
                    )

            # Update tracking
            self._prev_boosts[battle]["my_active"] = dict(current_boosts)

        # === OPPONENT POKEMON STAT CHANGES ===
        if battle.opponent_active_pokemon:
            species = battle.opponent_active_pokemon.species
            attack_type = self._get_pokemon_attack_type(battle.opponent_active_pokemon)
            current_boosts = battle.opponent_active_pokemon.boosts
            prev_boosts = self._prev_boosts[battle].get("opp_active", {})

            # Determine which stat to track
            relevant_stat = "spa" if attack_type == "special" else "atk"

            current_stage = current_boosts.get(relevant_stat, 0)
            prev_stage = prev_boosts.get(relevant_stat, 0)
            stage_change = current_stage - prev_stage

            if stage_change != 0:
                # Apply diminishing returns based on PREVIOUS stage
                weight = self._get_stage_reward_weight(prev_stage)
                stage_penalty = self._beta_stage * stage_change * weight
                reward -= stage_penalty  # Opponent boost is negative for us

                if self._logger.level <= logging.DEBUG:
                    self._logger.debug(
                        f"OPP {species} ({attack_type}): {relevant_stat} {prev_stage:+d} → {current_stage:+d} "
                        f"(weight={weight:.2f}, penalty={stage_penalty:+.3f})"
                    )

            # Update tracking
            self._prev_boosts[battle]["opp_active"] = dict(current_boosts)

        return reward

    def compute_reward(self, battle: AbstractBattle) -> float:
        """
        Compute reward for current transition.

        Implements progressive reward system:
        1. Basic: Win/loss + HP changes + KO count
        2. Stat stages with diminishing returns and attack-type awareness
        3. Status effects (from SAR)

        Returns:
            float: Reward for this step
        """
        # Initialize reward buffer
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = 0.0
            self._prev_hp_fractions[battle] = {}

        reward = 0.0

        turn_info = self._compute_turn_info(battle)

        # ===== TERMINAL REWARD =====
        if battle.finished:
            if battle.won:
                reward += 10.0
            else:
                reward -= 10.0

            # Add alive count difference
            my_alive = len([p for p in battle.team.values() if not p.fainted])
            opp_alive = len([p for p in battle.opponent_team.values() if not p.fainted])
            reward += float(my_alive - opp_alive)

            self._cleanup_battle_tracking(battle)
            return reward

        # ===== STEP REWARD =====

        # Track HP changes
        current_value = 0.0

        # My team HP
        for mon in battle.team.values():
            current_value += mon.current_hp_fraction
            if mon.fainted:
                current_value -= 2.0  # Faint penalty

        # Opponent team HP (inverted)
        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction
            if mon.fainted:
                current_value += 2.0  # Opponent faint reward

        # Compute delta reward
        reward = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value

        # Add status effect reward (from SAR)
        if turn_info is not None:
            reward += self._beta_status * (
                float(turn_info.status_inflicted_on_opp) - float(turn_info.status_received_self)
            )

        # Add stat stage reward with diminishing returns and attack-type awareness
        reward += self._compute_stat_stage_reward(battle)

        return reward

    def _compute_turn_info(self, battle: AbstractBattle):
        """Convert poke-env battle to SAR TurnInfo for reward shaping."""
        if self._state_adapter is None or self._turn_info_adapter is None:
            return None
        try:
            battle_state = self._state_adapter.convert(battle)
            return self._turn_info_adapter.compute_turn_info(battle, battle_state)
        except Exception as exc:  # pragma: no cover - defensive logging
            if self._logger:
                self._logger.debug("TurnInfo conversion failed: %s", exc)
            return None

    def _cleanup_battle_tracking(self, battle: AbstractBattle):
        """Reset per-battle trackers once an episode ends."""
        self._reward_buffer.pop(battle, None)
        self._prev_hp_fractions.pop(battle, None)
        self._prev_boosts.pop(battle, None)
        battle_id = getattr(battle, "battle_tag", str(id(battle)))
        if self._turn_info_adapter:
            self._turn_info_adapter.reset_battle(battle_id)

    # ============================================================
    # ACTION EXECUTION
    # ============================================================

    @staticmethod
    def action_to_order(action, battle, fake: bool = False, strict: bool = True):
        """
        Override SinglesEnv.action_to_order for our custom 10-action space.

        IMPORTANT: Actions 0-3 and 4-7 correspond to MOVE SLOTS, not available_moves indices.

        Args:
            action: Integer in [0, 9]
            battle: Current battle state
            fake: If true, avoid returning default if possible
            strict: If true, throw error on illegal move

        Returns:
            BattleOrder object
        """
        try:
            # Moves without tera (0-3)
            if 0 <= action < 4:
                slot_idx = action
                if battle.active_pokemon is None:
                    raise ValueError("No active pokemon")

                # Get move from slot
                active_moves = battle.active_pokemon.moves
                move_list = list(active_moves.values())

                if slot_idx >= len(move_list):
                    raise ValueError(f"Move slot {slot_idx} doesn't exist (only {len(move_list)} moves)")

                target_move = move_list[slot_idx]

                # Check if this move is actually available
                if target_move not in battle.available_moves:
                    # Async/transition state: fall back to first available move if any
                    if len(battle.available_moves) > 0:
                        return Player.create_order(battle.available_moves[0])
                    # No moves available (force switch/transition): let outer fallback handle
                    raise ValueError(f"Move slot {slot_idx} ({target_move.id}) not available")

                return Player.create_order(target_move)

            # Moves with tera (4-7)
            elif 4 <= action < 8:
                slot_idx = action - 4
                if battle.active_pokemon is None:
                    raise ValueError("No active pokemon")

                if not battle.can_tera:
                    raise ValueError("Cannot terastallize")

                # Get move from slot
                active_moves = battle.active_pokemon.moves
                move_list = list(active_moves.values())

                if slot_idx >= len(move_list):
                    raise ValueError(f"Move slot {slot_idx} doesn't exist (only {len(move_list)} moves)")

                target_move = move_list[slot_idx]

                # Check if this move is actually available
                if target_move not in battle.available_moves:
                    if len(battle.available_moves) > 0:
                        return Player.create_order(battle.available_moves[0], terastallize=True)
                    raise ValueError(f"Move slot {slot_idx} ({target_move.id}) not available")

                return Player.create_order(target_move, terastallize=True)

            # Switches (8-9)
            elif 8 <= action < 10:
                switch_idx = action - 8
                if switch_idx >= len(battle.available_switches):
                    raise ValueError(f"Switch index {switch_idx} out of bounds")

                return Player.create_order(battle.available_switches[switch_idx])

            else:
                raise ValueError(f"Invalid action {action}")

        except ValueError as e:
            # In strict mode, only raise if the battle is still ongoing and legal options exist
            if strict and not battle.finished and (
                len(battle.available_moves) > 0 or len(battle.available_switches) > 0
            ):
                raise e
            # Silently fallback to random move (expected during async/terminal transitions)
            return Player.choose_random_singles_move(battle)

    @staticmethod
    def order_to_action(order, battle, fake: bool = False, strict: bool = True):
        """
        Override SinglesEnv.order_to_action for our custom 10-action space.

        IMPORTANT: Returns SLOT INDEX, not available_moves index.

        Args:
            order: BattleOrder to convert
            battle: Current battle state
            fake: If true, avoid returning default if possible
            strict: If true, throw error on illegal order

        Returns:
            Action index in [0, 9]
        """
        from poke_env.player.battle_order import (
            DefaultBattleOrder,
            ForfeitBattleOrder,
            SingleBattleOrder,
        )

        try:
            # Special orders
            if isinstance(order, DefaultBattleOrder):
                # Return first valid action
                if battle.active_pokemon and len(battle.available_moves) > 0:
                    # Find first available move slot
                    active_moves = battle.active_pokemon.moves
                    for slot_idx, move in enumerate(active_moves.values()):
                        if slot_idx >= 4:
                            break
                        if move in battle.available_moves:
                            return np.int64(slot_idx)

                if len(battle.available_switches) > 0:
                    return np.int64(8)

                raise ValueError("No valid actions available")

            elif isinstance(order, ForfeitBattleOrder):
                # Return first valid action (can't forfeit)
                if battle.active_pokemon and len(battle.available_moves) > 0:
                    active_moves = battle.active_pokemon.moves
                    for slot_idx, move in enumerate(active_moves.values()):
                        if slot_idx >= 4:
                            break
                        if move in battle.available_moves:
                            return np.int64(slot_idx)

                if len(battle.available_switches) > 0:
                    return np.int64(8)

                raise ValueError("No valid actions available")

            else:
                assert isinstance(order, SingleBattleOrder)
                assert not isinstance(order.order, str)

                # Switch
                if isinstance(order.order, Pokemon):
                    # Find the index in available_switches
                    switch_idx = None
                    for i, switch_pokemon in enumerate(battle.available_switches):
                        if switch_pokemon.base_species == order.order.base_species:
                            switch_idx = i
                            break

                    if switch_idx is None:
                        raise ValueError(
                            f"Switch pokemon {order.order.base_species} not in available_switches"
                        )

                    if switch_idx >= 2:
                        raise ValueError(f"Switch index {switch_idx} >= 2 (BSS limit)")

                    return np.int64(8 + switch_idx)

                # Move
                else:
                    assert battle.active_pokemon is not None

                    # Find the SLOT index for this move
                    active_moves = battle.active_pokemon.moves
                    slot_idx = None
                    for i, move in enumerate(active_moves.values()):
                        if i >= 4:
                            break
                        if move.id == order.order.id:
                            slot_idx = i
                            break

                    # Handle Struggle (not present in move list)
                    if slot_idx is None and order.order.id == "struggle":
                        # Map struggle to first move slot (policy will ignore mask if invalid)
                        return np.int64(0)

                    if slot_idx is None:
                        raise ValueError(f"Move {order.order.id} not found in active pokemon moves")

                    # Check for terastallize
                    if order.terastallize:
                        return np.int64(4 + slot_idx)
                    else:
                        return np.int64(slot_idx)

        except (ValueError, AssertionError) as e:
            # If battle is over or no legal actions, fall back to a safe default
            if strict and not battle.finished and (
                len(battle.available_moves) > 0 or len(battle.available_switches) > 0
            ):
                raise e
            # Silently default to action 0 (this is expected during async/terminal states)
            return np.int64(0)

    def set_team_order(self, team_order: str):
        """
        Set custom team order for the next battle.

        Args:
            team_order: Team order string (e.g., "/team 123456")
        """
        self._custom_team_order = team_order

    def teampreview(self, battle: AbstractBattle) -> str:
        """
        Choose team order for BSS 3v3.

        If a custom order was set via set_team_order(), use that.
        Otherwise, use fixed order: Koraidon, Calyrex, Flutter Mane (first 3).

        Format: "/team 123456" means use pokemon 1, 2, 3 in battle
        """
        # Check for custom order
        if hasattr(self, '_custom_team_order') and self._custom_team_order:
            order = self._custom_team_order
            # Clear after use (one-time)
            self._custom_team_order = None
            return order

        # Default fixed order: 1, 2, 3, 4, 5, 6
        # This means: Koraidon (1), Calyrex (2), Flutter Mane (3) will be used
        return "/team 123456"

    def choose_move(self, battle: AbstractBattle):
        """
        This method is called by poke-env during actual battles.
        For RL training, this will be overridden by the RL algorithm.

        Default: Random valid move
        """
        return self.choose_random_move(battle)

    # ============================================================
    # BATTLE LIFECYCLE
    # ============================================================

    def _battle_finished_callback(self, battle: AbstractBattle):
        """
        Called when a battle finishes. Clean up battle-specific state.

        This prevents memory leaks and ensures fresh state for new battles.
        """
        # Clean up reward tracking
        if battle in self._reward_buffer:
            del self._reward_buffer[battle]

        if battle in self._prev_hp_fractions:
            del self._prev_hp_fractions[battle]

        if battle in self._turn_info:
            del self._turn_info[battle]

        # Clean up cached mask
        if hasattr(self, '_last_valid_mask'):
            del self._last_valid_mask

        # Call parent callback if needed
        super()._battle_finished_callback(battle)
