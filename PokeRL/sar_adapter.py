"""
Adapter layer between poke-env and SAR (State-Action-Reward) system.

This module converts poke-env's AbstractBattle to SAR's data structures:
- AbstractBattle -> BattleState (state.py)
- AbstractBattle -> TurnInfo (reward_function.py)
- Action index -> BattleOrder
"""

import sys
import os
import numpy as np
from typing import Dict, Optional, Tuple

# Add SAR directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SAR'))

from state import BattleState, PokemonView
from action import BSSActionSpace, Action, ActionKind
from reward_function import TurnInfo, compute_turn_reward, compute_final_result_reward

from poke_env.battle import AbstractBattle, Pokemon, PokemonType, Status
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.effect import Effect


# ============================================================
# SPECIES MAPPING
# ============================================================

# Map pokemon species names to IDs
SPECIES_NAME_TO_ID: Dict[str, int] = {
    'koraidon': 1,
    'calyrexshadow': 2,
    'fluttermane': 3,
    'chienpao': 4,
    'dondozo': 5,
    'tinglu': 6,
}

# Map type names to IDs
TYPE_NAME_TO_ID: Dict[str, int] = {
    'normal': 0, 'fire': 1, 'water': 2, 'electric': 3, 'grass': 4,
    'ice': 5, 'fighting': 6, 'poison': 7, 'ground': 8, 'flying': 9,
    'psychic': 10, 'bug': 11, 'rock': 12, 'ghost': 13, 'dragon': 14,
    'dark': 15, 'steel': 16, 'fairy': 17, 'stellar': 18,
}

# Map status names to IDs
STATUS_NAME_TO_ID: Dict[str, int] = {
    None: 0,  # No status
    'brn': 1, 'par': 2, 'slp': 3, 'psn': 4, 'frz': 5, 'tox': 6,
}

WEATHER_NAME_TO_ID: Dict[str, int] = {
    'none': 0,
    **{weather.name.lower(): idx + 1 for idx, weather in enumerate(list(Weather))}
}

FIELD_NAME_TO_ID: Dict[str, int] = {
    'none': 0,
    **{field.name.lower(): idx + 1 for idx, field in enumerate(list(Field))}
}

WEATHER_SUPPRESS_ABILITIES = {'airlock', 'cloudnine'}


# ============================================================
# BATTLE STATE ADAPTER
# ============================================================

class BattleStateAdapter:
    """
    Converts poke-env's AbstractBattle to SAR's BattleState.
    """

    def __init__(self):
        self._species_cache: Dict[str, int] = {}

    def convert(self, battle: AbstractBattle) -> BattleState:
        """
        Convert poke-env battle to SAR BattleState.

        Args:
            battle: poke-env AbstractBattle

        Returns:
            BattleState: SAR state representation
        """
        # Active pokemon
        my_active = self._convert_pokemon(battle.active_pokemon, battle, is_active=True)
        opp_active = self._convert_pokemon(
            battle.opponent_active_pokemon, battle, is_active=True, is_opponent=True
        )

        # Team
        my_team = [self._convert_pokemon(p, battle) for p in battle.team.values()]
        opp_team = [self._convert_pokemon(p, battle, is_opponent=True)
                    for p in battle.opponent_team.values()]

        # Count alive
        my_alive = len([p for p in battle.team.values() if not p.fainted])
        opp_alive = len([p for p in battle.opponent_team.values() if not p.fainted])

        # Battle flow
        turn_index = battle.turn
        my_tera_used = not battle.can_tera  # If can't tera, it's already used
        opp_tera_used = battle.opponent_used_tera

        rocks_my, spikes_my = self._extract_hazard_info(battle.side_conditions)
        rocks_opp, spikes_opp = self._extract_hazard_info(battle.opponent_side_conditions)
        weather_id, weather_turns = self._extract_weather_info(battle)
        terrain_id, terrain_turns = self._extract_terrain_info(battle)
        weather_effective = self._is_weather_effective(battle)
        gravity_active = Field.GRAVITY in battle.fields
        trick_room_active = Field.TRICK_ROOM in battle.fields

        return BattleState(
            my_active=my_active,
            opp_active=opp_active,
            my_team=my_team,
            opp_team=opp_team,
            turn_index=turn_index,
            my_alive=my_alive,
            opp_alive=opp_alive,
            my_tera_used=my_tera_used,
            opp_tera_used=opp_tera_used,
            rocks_on_my_side=rocks_my,
            rocks_on_opp_side=rocks_opp,
            spikes_on_my_side=spikes_my,
            spikes_on_opp_side=spikes_opp,
            weather_id=weather_id,
            weather_turns_active=weather_turns,
            weather_effective=weather_effective,
            terrain_id=terrain_id,
            terrain_turns_active=terrain_turns,
            gravity_active=gravity_active,
            trick_room_active=trick_room_active,
            # TODO: Add field hazards, perish song, etc.
        )

    def _convert_pokemon(
        self,
        pokemon: Optional[Pokemon],
        battle: AbstractBattle,
        is_active: bool = False,
        is_opponent: bool = False
    ) -> PokemonView:
        """
        Convert poke-env Pokemon to SAR PokemonView.
        """
        if pokemon is None:
            # Return dummy pokemon
            return PokemonView(
                species_id=0,
                hp_fraction=0.0,
                is_fainted=True,
            )

        # Get species ID
        species_name = pokemon.species.lower().replace('-', '')
        species_id = SPECIES_NAME_TO_ID.get(species_name, 0)

        # HP
        hp_fraction = pokemon.current_hp_fraction

        # Status
        status_name = pokemon.status.name.lower() if pokemon.status else None
        status_id = STATUS_NAME_TO_ID.get(status_name, 0)
        status_features = self._extract_status_features(pokemon)

        # Fainted
        is_fainted = pokemon.fainted

        # Tera
        tera_used = getattr(pokemon, "is_terastallized", False)
        tera_type_id = 0
        if tera_used and pokemon.tera_type:
            tera_type_id = TYPE_NAME_TO_ID.get(pokemon.tera_type.name.lower(), 0)

        # Stat stages (boosts)
        boosts = pokemon.boosts
        atk_stage = boosts.get('atk', 0)
        def_stage = boosts.get('def', 0)
        spa_stage = boosts.get('spa', 0)
        spd_stage = boosts.get('spd', 0)
        spe_stage = boosts.get('spe', 0)
        acc_stage = boosts.get('accuracy', 0)
        eva_stage = boosts.get('evasion', 0)

        # Item detection (heuristic - poke-env doesn't always expose items)
        has_choice_lock = False
        has_focus_sash = False
        has_assault_vest = False
        sash_intact = False

        if pokemon.item:
            item_name = pokemon.item.lower()
            if 'choice' in item_name:
                has_choice_lock = True
            if 'focussash' in item_name or 'sash' in item_name:
                has_focus_sash = True
                # Sash is intact if HP is full
                sash_intact = hp_fraction >= 0.99
            if 'assaultvest' in item_name:
                has_assault_vest = True

        # Ability detection
        is_unaware = False
        if pokemon.ability and 'unaware' in pokemon.ability.lower():
            is_unaware = True

        return PokemonView(
            species_id=species_id,
            hp_fraction=hp_fraction,
            status_id=status_id,
            status_turns_remaining=status_features["status_turns_remaining"],
            status_severity=status_features["status_severity"],
            is_drowsy=status_features["is_drowsy"],
            drowsy_turns=status_features["drowsy_turns"],
            status_attack_penalty=status_features["status_attack_penalty"],
            status_speed_penalty=status_features["status_speed_penalty"],
            status_hp_loss_per_turn=status_features["status_hp_loss_per_turn"],
            is_active=is_active,
            is_fainted=is_fainted,
            tera_used=tera_used,
            tera_type_id=tera_type_id,
            atk_stage=atk_stage,
            def_stage=def_stage,
            spa_stage=spa_stage,
            spd_stage=spd_stage,
            spe_stage=spe_stage,
            acc_stage=acc_stage,
            eva_stage=eva_stage,
            has_choice_lock=has_choice_lock,
            has_focus_sash=has_focus_sash,
            sash_intact=sash_intact,
            is_unaware=is_unaware,
            has_assault_vest=has_assault_vest,
        )

    def _extract_status_features(self, pokemon: Optional[Pokemon]) -> Dict[str, float]:
        """
        Compute status-related auxiliary metrics for PokemonView.
        Returns default zeros if no status.
        """
        features = {
            "status_turns_remaining": 0.0,
            "status_severity": 0.0,
            "is_drowsy": False,
            "drowsy_turns": 0.0,
            "status_attack_penalty": 0.0,
            "status_speed_penalty": 0.0,
            "status_hp_loss_per_turn": 0.0,
        }

        if pokemon is None:
            return features

        status = pokemon.status.name.lower() if pokemon.status else None
        stats = pokemon.base_stats if pokemon.base_stats else {}
        atk = float(stats.get("atk", 0) or 0)
        spa = float(stats.get("spa", 0) or 0)
        spe = float(stats.get("spe", 0) or 0)
        total_atk = max(1.0, atk + spa)
        physical_bias = atk / total_atk
        speed_ratio = min(1.0, spe / 255.0 if spe else 0.0)

        if status == "slp":
            # Sleep lasts 1-3 turns typically. status_counter counts turns slept.
            turns_remaining = max(0.0, float(3 - pokemon.status_counter))
            features["status_turns_remaining"] = turns_remaining
            features["status_severity"] = 1.0
            features["status_attack_penalty"] = 1.0
            features["status_speed_penalty"] = 1.0
        elif status == "brn":
            features["status_turns_remaining"] = -1.0
            attack_penalty = 0.5 + 0.5 * physical_bias
            features["status_attack_penalty"] = min(1.0, attack_penalty)
            features["status_hp_loss_per_turn"] = 0.0625
            features["status_severity"] = 0.6 + 0.4 * physical_bias
        elif status == "par":
            features["status_turns_remaining"] = -1.0
            speed_penalty = 0.5 + 0.5 * speed_ratio
            features["status_speed_penalty"] = min(1.0, speed_penalty)
            features["status_attack_penalty"] = 0.1
            features["status_severity"] = 0.5 + 0.3 * speed_ratio
        elif status == "psn":
            features["status_turns_remaining"] = -1.0
            features["status_hp_loss_per_turn"] = 0.0625
            features["status_severity"] = 0.5
        elif status == "tox":
            features["status_turns_remaining"] = -1.0
            features["status_hp_loss_per_turn"] = 0.125
            features["status_severity"] = 0.7
        elif status == "frz":
            features["status_turns_remaining"] = -1.0
            features["status_attack_penalty"] = 1.0
            features["status_speed_penalty"] = 0.5
            features["status_severity"] = 0.9

        # Check Yawn / Drowsy
        try:
            effects = pokemon.effects
        except Exception:
            effects = {}

        if Effect.YAWN in effects:
            features["is_drowsy"] = True
            features["drowsy_turns"] = float(effects[Effect.YAWN])
            features["status_severity"] = max(features["status_severity"], 0.4)

        return features

    def _extract_hazard_info(self, side_conditions: Dict[SideCondition, int]) -> Tuple[bool, int]:
        if side_conditions is None:
            return False, 0
        rocks = SideCondition.STEALTH_ROCK in side_conditions
        spikes_layers = int(side_conditions.get(SideCondition.SPIKES, 0))
        return rocks, spikes_layers

    def _extract_weather_info(self, battle: AbstractBattle) -> Tuple[int, float]:
        if not battle.weather:
            return 0, 0.0
        weather, start_turn = next(iter(battle.weather.items()))
        weather_id = WEATHER_NAME_TO_ID.get(weather.name.lower(), 0)
        turns_active = float(max(0, battle.turn - start_turn + 1))
        return weather_id, turns_active

    def _extract_terrain_info(self, battle: AbstractBattle) -> Tuple[int, float]:
        if not battle.fields:
            return 0, 0.0
        for field, start_turn in battle.fields.items():
            if field.is_terrain:
                field_id = FIELD_NAME_TO_ID.get(field.name.lower(), 0)
                turns_active = float(max(0, battle.turn - start_turn + 1))
                return field_id, turns_active
        return 0, 0.0

    def _is_weather_effective(self, battle: AbstractBattle) -> bool:
        if not battle.weather:
            return False
        return not self._is_weather_suppressed(battle)

    def _is_weather_suppressed(self, battle: AbstractBattle) -> bool:
        for mon in (battle.active_pokemon, battle.opponent_active_pokemon):
            if mon is None:
                continue
            ability = getattr(mon, "ability", None)
            if ability and self._normalize_ability_name(ability) in WEATHER_SUPPRESS_ABILITIES:
                return True
        return False

    def _normalize_ability_name(self, ability: str) -> str:
        return ability.replace("-", "").replace(" ", "").lower()


# ============================================================
# TURN INFO ADAPTER
# ============================================================

class TurnInfoAdapter:
    """
    Tracks battle state changes and generates TurnInfo for reward computation.
    - 이전 BattleState와 현재 BattleState를 비교해서
      hp_loss_self / hp_loss_opp / num_self_KO / num_opp_KO 를 계산한다.
    """

    def __init__(self):
        # 각 battle_id별 직전 BattleState 저장
        self._prev_state: Dict[str, BattleState] = {}
        # 턴 카운트 (1, 2, 3, ...)
        self._turn_count: Dict[str, int] = {}
        # 랭크 상태 추적 (battle_id -> species_id -> stage value)
        self._stage_memory_self: Dict[str, Dict[int, float]] = {}
        self._stage_memory_opp: Dict[str, Dict[int, float]] = {}

    # -----------------------------
    # 내부 유틸
    # -----------------------------
    def _get_battle_id(self, battle: AbstractBattle) -> str:
        # poke-env에서는 보통 battle_tag로 유니크 ID를 제공함
        return getattr(battle, "battle_tag", str(id(battle)))

    def _total_hp_fraction(self, team) -> float:
        """
        팀 전체 HP 비율 합 (0~팀 최대 수).
        - PokemonView.hp_fraction (0~1)을 단순 합산.
        """
        if team is None:
            return 0.0
        return float(sum(max(0.0, min(1.0, p.hp_fraction)) for p in team))

    # -----------------------------
    # 메인 함수
    # -----------------------------
    def compute_turn_info(
        self,
        battle: AbstractBattle,
        current_state: BattleState,
    ) -> TurnInfo:
        """
        Compute TurnInfo by comparing current and previous battle states.

        Args:
            battle: poke-env battle object
            current_state: SAR BattleState for the *current* turn

        Returns:
            TurnInfo: filled with hp_loss_self / hp_loss_opp / KOs / turn index 등
        """
        battle_id = self._get_battle_id(battle)

        # 첫 턴인지 확인
        prev_state = self._prev_state.get(battle_id, None)
        prev_turn = self._turn_count.get(battle_id, 0)

        # 턴 카운트 업데이트
        turn_idx = prev_turn + 1
        self._turn_count[battle_id] = turn_idx

        # 이전 상태가 없는 경우 (배틀 첫 턴)
        if prev_state is None:
            # 아무런 변화 정보를 모를 때는 0으로 세팅
            turn_info = TurnInfo(
                turn_index=turn_idx,
                total_turns_estimate=turn_idx,   # 일단 현재 턴까지를 길이로 사용
                hp_loss_opp=0.0,
                hp_loss_self=0.0,
                num_opp_KO=0,
                num_self_KO=0,
            )
            # 현재 상태를 prev로 저장해두고 반환
            self._prev_state[battle_id] = current_state
            return turn_info

        # -------------------------
        # HP 감소량 계산
        # -------------------------
        # 팀 전체 HP 합 기준 (원하면 active만 쓰도록 바꿀 수 있음)
        prev_hp_self = self._total_hp_fraction(prev_state.my_team)
        prev_hp_opp = self._total_hp_fraction(prev_state.opp_team)

        curr_hp_self = self._total_hp_fraction(current_state.my_team)
        curr_hp_opp = self._total_hp_fraction(current_state.opp_team)

        # "loss" = 이전 - 현재 (0 이상만 사용)
        hp_loss_self = max(0.0, prev_hp_self - curr_hp_self)
        hp_loss_opp = max(0.0, prev_hp_opp - curr_hp_opp)

        # -------------------------
        # KO 수 계산
        # -------------------------
        # prev_alive - curr_alive = 감소한 마릿수 = 이번 턴에 죽은 마릿수
        delta_my_alive = prev_state.my_alive - current_state.my_alive
        delta_opp_alive = prev_state.opp_alive - current_state.opp_alive

        num_self_KO = max(0, int(delta_my_alive))
        num_opp_KO = max(0, int(delta_opp_alive))

        # -------------------------
        # TurnInfo 생성
        # -------------------------
        # 상태 이상 / 랭크 변화 계산
        status_inflicted = self._count_new_statuses(prev_state.opp_team, current_state.opp_team)
        status_received = self._count_new_statuses(prev_state.my_team, current_state.my_team)
        stage_adv_self = self._compute_stage_advantage(
            prev_state.my_team, current_state.my_team, battle_id, is_self=True
        )
        stage_adv_opp = self._compute_stage_advantage(
            prev_state.opp_team, current_state.opp_team, battle_id, is_self=False
        )

        turn_info = TurnInfo(
            turn_index=turn_idx,
            # total_turns_estimate는 현재로서는 실제 길이를 모름
            # → "지금까지 진행된 턴 수"를 넣고, 나중에 더 정교하게 바꿀 수 있음
            total_turns_estimate=turn_idx,
            hp_loss_opp=hp_loss_opp,
            hp_loss_self=hp_loss_self,
            num_opp_KO=num_opp_KO,
            num_self_KO=num_self_KO,
            status_inflicted_on_opp=status_inflicted,
            status_received_self=status_received,
            stage_advantage_self=stage_adv_self,
            stage_advantage_opp=stage_adv_opp,
        )

        # 현재 상태를 prev_state로 업데이트
        self._prev_state[battle_id] = current_state

        return turn_info

    # -----------------------------
    # 배틀 리셋
    # -----------------------------
    def _compute_hp_loss(
        self,
        prev_team: list,
        current_team: list
    ) -> float:
        """
        Compute total HP loss for a team.

        Args:
            prev_team: Previous team state
            current_team: Current team state

        Returns:
            Total HP loss (0~1 per pokemon, summed)
        """
        total_loss = 0.0

        # Match pokemon by species (assuming order is preserved)
        for prev_mon, curr_mon in zip(prev_team, current_team):
            if prev_mon.species_id == curr_mon.species_id:
                hp_diff = prev_mon.hp_fraction - curr_mon.hp_fraction
                if hp_diff > 0:  # Lost HP
                    total_loss += hp_diff

        return total_loss

    def _count_new_statuses(self, prev_team, current_team) -> int:
        """
        Count how many pokemon newly received a status condition.
        """
        if prev_team is None or current_team is None:
            return 0

        prev_map = self._team_to_map(prev_team)
        count = 0
        for species_id, curr_mon in self._team_to_map(current_team).items():
            if species_id == 0:
                continue
            prev_mon = prev_map.get(species_id)
            if prev_mon is None:
                continue
            if prev_mon.status_id == 0 and curr_mon.status_id != 0:
                count += 1
        return count

    def _compute_stage_advantage(self, prev_team, current_team, battle_id: str, is_self: bool) -> float:
        """
        Track per-pokemon stat multipliers and return net change for this turn.
        Resets (e.g., Haze, switching out, white-out to 0) simply reset memory
        without penalizing the player.
        """
        if prev_team is None or current_team is None:
            return 0.0

        memory = self._stage_memory_self if is_self else self._stage_memory_opp
        battle_memory = memory.setdefault(battle_id, {})
        advantage = 0.0

        curr_map = self._team_to_map(current_team)
        for species_id, curr_mon in curr_map.items():
            if species_id == 0:
                continue

            curr_value = self._aggregate_stage_value(curr_mon)
            prev_value = battle_memory.get(species_id, 0.0)

            if self._is_stage_reset(curr_mon):
                # Reset without penalty, so drop stored value and continue
                battle_memory[species_id] = 0.0
                continue

            delta = curr_value - prev_value
            if abs(delta) > 1e-6:
                advantage += delta
                battle_memory[species_id] = curr_value

        # Remove entries for pokemon no longer present to avoid stale rewards
        valid_species = set(curr_map.keys())
        to_delete = [sid for sid in battle_memory.keys() if sid not in valid_species]
        for sid in to_delete:
            del battle_memory[sid]

        return advantage

    def _aggregate_stage_value(self, mon) -> float:
        tracked_stats = [
            'atk_stage', 'def_stage', 'spa_stage',
            'spd_stage', 'spe_stage', 'acc_stage', 'eva_stage'
        ]
        total = 0.0
        for stat in tracked_stats:
            stage = getattr(mon, stat, 0)
            total += self._stage_to_multiplier(stage) - 1.0
        return total

    def _is_stage_reset(self, mon) -> bool:
        tracked_stats = [
            'atk_stage', 'def_stage', 'spa_stage',
            'spd_stage', 'spe_stage', 'acc_stage', 'eva_stage'
        ]
        return all(getattr(mon, stat, 0) == 0 for stat in tracked_stats)

    def _team_to_map(self, team):
        mapping = {}
        if team is None:
            return mapping
        for mon in team:
            if mon is None:
                continue
            species_id = getattr(mon, 'species_id', 0)
            if species_id == 0:
                continue
            mapping[species_id] = mon
        return mapping

    def _stage_to_multiplier(self, stage: int) -> float:
        """
        Convert Pokemon stat stage (-6~+6) to actual in-game multiplier.
        """
        stage = int(max(-6, min(6, stage)))
        if stage >= 0:
            return (2.0 + stage) / 2.0
        else:
            return 2.0 / (2.0 - stage)

    def reset_battle(self, battle_id: str):
        """Reset tracking for a battle."""
        if battle_id in self._prev_state:
            del self._prev_state[battle_id]
        if battle_id in self._turn_count:
            del self._turn_count[battle_id]
        if battle_id in self._stage_memory_self:
            del self._stage_memory_self[battle_id]
        if battle_id in self._stage_memory_opp:
            del self._stage_memory_opp[battle_id]


# ============================================================
# ACTION ADAPTER
# ============================================================

class ActionAdapter:
    """
    Converts between action indices and poke-env BattleOrders.
    Uses SAR's BSSActionSpace directly.
    """

    @staticmethod
    def index_to_action(action_idx: int) -> Action:
        """
        Convert action index to SAR Action.

        Args:
            action_idx: Integer in [0, 9]

        Returns:
            SAR Action object
        """
        return BSSActionSpace.index_to_action(action_idx)

    @staticmethod
    def action_to_index(action: Action) -> int:
        """
        Convert SAR Action to index.

        Args:
            action: SAR Action object

        Returns:
            Integer in [0, 9]
        """
        return BSSActionSpace.action_to_index(action)
