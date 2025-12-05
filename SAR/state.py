# state.py
"""
State representation for Reg I BSS team (Koraidon / Calyrex-S / Flutter Mane /
Chien-Pao / Dondozo / Ting-Lu).

요점:
- SD 여부, Curse 여부 같은 "스킬 단위 플래그" 대신
  -> 공/방/스공/스방/스피드/명중/회피의 랭크 스테이지(-6~+6)를 직접 상태로 사용.
- 이로 인해 Swords Dance, Dragon Dance, Calm Mind, Intimidate 등
  모든 랭크업/랭크다운 효과를 하나의 구조로 표현 가능.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# -------------------------
# 유틸: 타입/상태/종 ID 인코딩
# -------------------------

# 실제 환경에서는 종/타입/상태를 고정된 인덱스로 매핑해야 한다.
# 여기서는 placeholder 형태로 두고, 외부에서 매핑 테이블을 관리한다고 가정.

NUM_SPECIES = 1024      # 충분히 큰 상한 (실제론 더 작게 써도 됨)
NUM_TYPES = 19          # 노말~페어리 + 특수(테라 없는 경우 0 등) 포함
NUM_STATUS = 6          # (none, burn, para, sleep, poison, freeze 등)


def one_hot(index: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        v[index] = 1.0
    return v


# =========================
# 포켓몬 단위 View 구조
# =========================

@dataclass
class PokemonView:
    """
    '해당 시점에서 관측 가능한 한 포켓몬의 정보'를 담는 구조체.

    - 내 필드 / 상대 필드 / 벤치 포켓몬 모두에 사용 가능.
    - 스킬 단위 플래그 대신, stat stages 로 랭크업/다운 상태를 직접 표현한다.
    """

    # 필수 정보
    species_id: int                      # 종 ID (0 ~ NUM_SPECIES-1, 환경에서 매핑)
    hp_fraction: float                   # 현재 HP (0~1)
    status_id: int = 0                   # 상태 이상 (0~NUM_STATUS-1)
    status_turns_remaining: float = 0.0  # 남은 지속 (슬립 등), -1 = indefinite
    status_severity: float = 0.0         # 상태 전체 영향 점수 (0~1+)
    is_drowsy: bool = False              # Yawn 등으로 잠들 예정인지
    drowsy_turns: float = 0.0            # 잠들기까지 남은 턴 수
    status_attack_penalty: float = 0.0   # 예측되는 공격력 저하율 (0~1)
    status_speed_penalty: float = 0.0    # 예측되는 속도 저하율 (0~1)
    status_hp_loss_per_turn: float = 0.0 # 매 턴 HP 감소 비율 추정
    is_active: bool = False              # 현재 필드에 나와 있는지
    is_fainted: bool = False             # 이미 기절했는지

    # 테라 관련
    tera_used: bool = False
    tera_type_id: int = 0                # NUM_TYPES 중 하나 (0 = none or original)

    # 능력치 랭크 (–6 ~ +6)
    atk_stage: int = 0
    def_stage: int = 0
    spa_stage: int = 0
    spd_stage: int = 0
    spe_stage: int = 0
    acc_stage: int = 0
    eva_stage: int = 0

    # 아이템/락 관련 (Choice, etc.)
    has_choice_lock: bool = False        # 밴드/스카프/스펙 등
    locked_move_id: Optional[int] = None # 어떤 기술에 락됐는지 (없다면 None)

    # 기타 전략적으로 중요하지만 랭크로 커버되지 않는 플래그
    # 예: Focus Sash 유지 여부, Unaware 여부, Assault Vest 여부 등
    has_focus_sash: bool = False         # (대표적으로 Chien-Pao)
    sash_intact: bool = False            # 아직 Sash가 깨지지 않았는지
    is_unaware: bool = False             # Dondozo 같은 특성
    has_assault_vest: bool = False       # Ting-Lu 같은 세팅

    def clamp(self) -> None:
        """
        값 범위 보정 (환경 버그 / 노이즈 대비용).
        """
        self.hp_fraction = float(np.clip(self.hp_fraction, 0.0, 1.0))
        self.atk_stage = int(np.clip(self.atk_stage, -6, 6))
        self.def_stage = int(np.clip(self.def_stage, -6, 6))
        self.spa_stage = int(np.clip(self.spa_stage, -6, 6))
        self.spd_stage = int(np.clip(self.spd_stage, -6, 6))
        self.spe_stage = int(np.clip(self.spe_stage, -6, 6))
        self.acc_stage = int(np.clip(self.acc_stage, -6, 6))
        self.eva_stage = int(np.clip(self.eva_stage, -6, 6))

    def _stage_to_scalar(self, stage: int) -> float:
        """
        랭크 스테이지(-6 ~ +6)를 [-1, 1] 범위의 실수로 정규화.
        - 0  -> 0.0
        - +6 -> +1.0
        - -6 -> -1.0
        """
        return float(stage / 6.0)

    def to_vector(self) -> np.ndarray:
        """
        이 PokemonView를 하나의 feature 벡터로 변환.

        벡터 구성 예:
        [ species_onehot,
          hp,
          status_onehot,
          is_active,
          is_fainted,
          tera_used,
          tera_type_onehot,
          atk_stage_norm,
          def_stage_norm,
          spa_stage_norm,
          spd_stage_norm,
          spe_stage_norm,
          acc_stage_norm,
          eva_stage_norm,
          has_choice_lock,
          locked_move_onehot(optional),
          has_focus_sash,
          sash_intact,
          is_unaware,
          has_assault_vest
        ]
        """
        self.clamp()

        features: List[np.ndarray] = []

        # species
        features.append(one_hot(self.species_id, NUM_SPECIES))

        # hp
        features.append(np.array([self.hp_fraction], dtype=np.float32))

        # status
        features.append(one_hot(self.status_id, NUM_STATUS))
        features.append(np.array([
            self.status_turns_remaining,
            self.status_severity,
            float(self.is_drowsy),
            self.drowsy_turns,
            self.status_attack_penalty,
            self.status_speed_penalty,
            self.status_hp_loss_per_turn,
        ], dtype=np.float32))

        # active / fainted
        features.append(np.array(
            [float(self.is_active), float(self.is_fainted)], dtype=np.float32
        ))

        # tera
        features.append(np.array([float(self.tera_used)], dtype=np.float32))
        features.append(one_hot(self.tera_type_id, NUM_TYPES))

        # stat stages (norm)
        stages = [
            self._stage_to_scalar(self.atk_stage),
            self._stage_to_scalar(self.def_stage),
            self._stage_to_scalar(self.spa_stage),
            self._stage_to_scalar(self.spd_stage),
            self._stage_to_scalar(self.spe_stage),
            self._stage_to_scalar(self.acc_stage),
            self._stage_to_scalar(self.eva_stage),
        ]
        features.append(np.array(stages, dtype=np.float32))

        # choice lock
        features.append(np.array([float(self.has_choice_lock)], dtype=np.float32))
        # locked_move_id는 환경이 move_vocab_size를 알아야 해서 여기서는 스칼라로만 둔다
        locked_move_scalar = -1.0 if self.locked_move_id is None else float(self.locked_move_id)
        features.append(np.array([locked_move_scalar], dtype=np.float32))

        # 기타 플래그
        features.append(np.array([
            float(self.has_focus_sash),
            float(self.sash_intact),
            float(self.is_unaware),
            float(self.has_assault_vest),
        ], dtype=np.float32))

        return np.concatenate(features, axis=0)


# =========================
# 배틀 전체 State 구조
# =========================

@dataclass
class BattleState:
    """
    한 턴(t)의 RL 상태 표현.

    SD 여부를 따로 플래그로 두지 않고,
    PokemonView의 stat stages 로 모든 랭크업/다운을 표현한다.

    구성:
    - my_active: 내 필드 포켓몬
    - opp_active: 상대 필드 포켓몬
    - my_team: 내 전체 팀(최대 6마리)
    - opp_team: 상대 팀(알려진 범위 내)
    - 전투 흐름(턴, 마릿수 차이, 테라 사용 여부 등)
    """

    # 내 필드, 상대 필드 (항상 1 vs 1 가정: BSS)
    my_active: PokemonView
    opp_active: PokemonView

    # 팀 전체 뷰 (내 6 / 상대 6까지 확장 가능)
    my_team: List[PokemonView] = field(default_factory=list)
    opp_team: List[PokemonView] = field(default_factory=list)

    # 전투 흐름 정보
    turn_index: int = 1
    my_alive: int = 3
    opp_alive: int = 3
    my_tera_used: bool = False
    opp_tera_used: bool = False

    # 기타 하이레벨 전략 feature (필요 시 확장)
    rocks_on_my_side: bool = False
    rocks_on_opp_side: bool = False
    spikes_on_my_side: int = 0
    spikes_on_opp_side: int = 0
    weather_id: int = 0
    weather_turns_active: float = 0.0
    weather_effective: bool = True
    terrain_id: int = 0
    terrain_turns_active: float = 0.0
    gravity_active: bool = False
    trick_room_active: bool = False

    # Perish Song 관련 카운트
    perish_count_my_active: Optional[int] = None
    perish_count_opp_active: Optional[int] = None

    def to_vector(self) -> np.ndarray:
        """
        BattleState 전체를 하나의 큰 벡터로 변환한다.
        NN 입력으로 바로 쓸 수 있는 형태.

        구조 (예시):
        [ my_active_vector,
          opp_active_vector,
          my_team_vectors (padding 포함),
          opp_team_vectors (padding 포함),
          battle_flow_features
        ]
        """
        features: List[np.ndarray] = []

        # 1) 필드 포켓몬
        features.append(self.my_active.to_vector())
        features.append(self.opp_active.to_vector())

        # 2) 팀 전체 (고정 크기: 6마리 기준으로 패딩)
        def team_to_matrix(team: List[PokemonView]) -> np.ndarray:
            if not team:
                return np.zeros((0,), dtype=np.float32)
            vecs = [p.to_vector() for p in team]
            return np.concatenate(vecs, axis=0)

        # 팀 크기가 항상 6이라고 가정하고, 부족하면 dummy padding
        max_team_size = 6

        def pad_team(team: List[PokemonView]) -> List[PokemonView]:
            if len(team) >= max_team_size:
                return team[:max_team_size]
            dummy = PokemonView(
                species_id=0,
                hp_fraction=0.0,
                status_id=0,
                is_active=False,
                is_fainted=True,
            )
            return team + [dummy] * (max_team_size - len(team))

        my_team_padded = pad_team(self.my_team)
        opp_team_padded = pad_team(self.opp_team)

        features.append(team_to_matrix(my_team_padded))
        features.append(team_to_matrix(opp_team_padded))

        # 3) 배틀 플로우 feature
        battle_flow = np.array([
            float(self.turn_index),
            float(self.my_alive),
            float(self.opp_alive),
            float(self.my_alive - self.opp_alive),
            float(self.my_tera_used),
            float(self.opp_tera_used),
            float(self.rocks_on_my_side),
            float(self.rocks_on_opp_side),
            float(self.spikes_on_my_side),
            float(self.spikes_on_opp_side),
            float(self.weather_id),
            self.weather_turns_active,
            float(self.weather_effective),
            float(self.terrain_id),
            self.terrain_turns_active,
            float(self.gravity_active),
            float(self.trick_room_active),
            float(-1 if self.perish_count_my_active is None else self.perish_count_my_active),
            float(-1 if self.perish_count_opp_active is None else self.perish_count_opp_active),
        ], dtype=np.float32)
        features.append(battle_flow)

        return np.concatenate(features, axis=0)


if __name__ == "__main__":
    # 간단 sanity check
    my_active = PokemonView(
        species_id=1,    # 예: Koraidon
        hp_fraction=0.8,
        status_id=0,
        is_active=True,
        atk_stage=2,     # 예: Swords Dance 1회 (+2)
        spe_stage=1,     # 예: Flame Charge 1회 (+1)
        has_choice_lock=True,
    )
    opp_active = PokemonView(
        species_id=2,    # 예: Ting-Lu
        hp_fraction=0.5,
        status_id=0,
        is_active=True,
        def_stage=1,
        spd_stage=2,
    )

    state = BattleState(
        my_active=my_active,
        opp_active=opp_active,
        my_team=[my_active],   # 예시로 1마리만 넣음 (나머지는 패딩)
        opp_team=[opp_active],
        turn_index=3,
        my_alive=3,
        opp_alive=2,
        my_tera_used=False,
        opp_tera_used=True,
    )

    vec = state.to_vector()
    print("State vector shape:", vec.shape)
