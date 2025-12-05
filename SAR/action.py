# actions.py
"""
Action space definition for BSS 3v3 environment using the
Reg I Koraidon / Calyrex-S / Flutter Mane / Chien-Pao / Dondozo / Ting-Lu team.

핵심 설계:
- 매 턴 에이전트가 선택하는 것은 다음 3가지 중 하나:
  1) MOVE       : 테라 없이 기술 사용 (4개 중 1개)
  2) TERA_MOVE  : 이번 턴 테라를 사용하면서 기술 사용 (4개 중 1개)
  3) SWITCH     : 벤치 포켓몬으로 교체 (2마리 중 1마리)

Flatten된 discrete action space:
- index 0~3  : MOVE_0 ~ MOVE_3
- index 4~7  : TERA_MOVE_0 ~ TERA_MOVE_3
- index 8~9  : SWITCH_BENCH_0 ~ SWITCH_BENCH_1

환경 쪽에서 할 일:
- 현재 턴 상황에 따라 "불가능한 액션" 을 mask 처리
  (예: 이미 테라 사용 → 4~7 mask, 해당 기술 PP 0 → 해당 index mask,
   교체할 포켓몬이 기절 → 해당 SWITCH index mask, 등)
- policy network는 항상 [0, ..., 9] 중 하나를 고르고,
  mask를 적용한 뒤 샘플/argmax를 수행하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionKind(str, Enum):
    """
    고수준 액션 타입 정의.

    - MOVE      : 테라 없이 기술 사용
    - TERA_MOVE : 테라를 사용하면서 기술 사용
    - SWITCH    : 벤치 포켓몬으로 교체
    """
    MOVE = "move"
    TERA_MOVE = "tera_move"
    SWITCH = "switch"


@dataclass
class Action:
    """
    High-level action 표현.

    kind:
        - ActionKind.MOVE      : move_index 사용
        - ActionKind.TERA_MOVE : move_index 사용
        - ActionKind.SWITCH    : switch_index 사용

    move_index:
        - 0 ~ 3
        - kind가 MOVE 또는 TERA_MOVE일 때만 의미 있음

    switch_index:
        - 0 ~ 1 (BSS 3v3 기준 벤치 슬롯 index)
        - kind가 SWITCH일 때만 의미 있음

    이 구조체를 환경 엔진에서 해석해서:
        - MOVE / TERA_MOVE → 실제 기술 선택 + (필요 시) 테라 처리
        - SWITCH           → 실제 교체 수행
    """

    kind: ActionKind
    move_index: Optional[int] = None
    switch_index: Optional[int] = None

    def __repr__(self) -> str:
        if self.kind in (ActionKind.MOVE, ActionKind.TERA_MOVE):
            return f"Action(kind={self.kind}, move_index={self.move_index})"
        else:
            return f"Action(kind={self.kind}, switch_index={self.switch_index})"


class BSSActionSpace:
    """
    BSS 3 vs 3 기준 Flattened Discrete Action Space.

    설계:
        - NUM_MOVES = 4   : 포켓몬은 항상 최대 4개의 기술을 가진다고 가정
        - NUM_BENCH = 2   : 3마리 중 필드 1 + 벤치 2 (BSS 3v3)

    인덱스 매핑:
        - 0 ~ 3 : MOVE_0 ~ MOVE_3
        - 4 ~ 7 : TERA_MOVE_0 ~ TERA_MOVE_3
        - 8 ~ 9 : SWITCH_BENCH_0 ~ SWITCH_BENCH_1

    사용 예시:
        - action_dim = BSSActionSpace.size()
        - logits = policy_network(state_vec)  # shape: [action_dim]
        - mask = env.compute_action_mask(state)  # shape: [action_dim], {0 or 1}
        - masked_logits = logits + log(mask)    # 불가능한 액션은 -inf
        - idx = sample(masked_logits) or argmax
        - high_level_action = BSSActionSpace.index_to_action(idx)
    """

    NUM_MOVES: int = 4
    NUM_BENCH: int = 2  # BSS 3v3: bench = 2

    @classmethod
    def size(cls) -> int:
        """
        전체 action space 크기 반환.
        BSS 3v3 기준: 4(MOVE) + 4(TERA_MOVE) + 2(SWITCH) = 10
        """
        return cls.NUM_MOVES + cls.NUM_MOVES + cls.NUM_BENCH

    # =========================
    # Index ↔ Action 변환
    # =========================

    @classmethod
    def index_to_action(cls, idx: int) -> Action:
        """
        정수 action index -> High-level Action 변환.

        0 <= idx < size() 인덱스를 받으면,
        해당하는 MOVE / TERA_MOVE / SWITCH 액션 객체를 반환한다.
        """
        if not (0 <= idx < cls.size()):
            raise ValueError(f"Invalid action index: {idx}")

        # 0~3: MOVE_0~3
        if 0 <= idx < cls.NUM_MOVES:
            return Action(kind=ActionKind.MOVE, move_index=idx)

        # 4~7: TERA_MOVE_0~3
        if cls.NUM_MOVES <= idx < cls.NUM_MOVES * 2:
            move_idx = idx - cls.NUM_MOVES
            return Action(kind=ActionKind.TERA_MOVE, move_index=move_idx)

        # 8~9: SWITCH_BENCH_0~1
        bench_idx = idx - cls.NUM_MOVES * 2
        return Action(kind=ActionKind.SWITCH, switch_index=bench_idx)

    @classmethod
    def action_to_index(cls, action: Action) -> int:
        """
        High-level Action -> 정수 index.

        정책 네트워크가 내는 것은 항상 "0~size()-1" 범위의 정수이므로,
        이 함수는 roll-out, debug 등에서 역방향 매핑용으로 사용 가능하다.
        """
        if action.kind == ActionKind.MOVE:
            if action.move_index is None:
                raise ValueError("MOVE action requires move_index.")
            if not (0 <= action.move_index < cls.NUM_MOVES):
                raise ValueError(f"Invalid move_index: {action.move_index}")
            return int(action.move_index)

        if action.kind == ActionKind.TERA_MOVE:
            if action.move_index is None:
                raise ValueError("TERA_MOVE action requires move_index.")
            if not (0 <= action.move_index < cls.NUM_MOVES):
                raise ValueError(f"Invalid move_index: {action.move_index}")
            return cls.NUM_MOVES + int(action.move_index)

        if action.kind == ActionKind.SWITCH:
            if action.switch_index is None:
                raise ValueError("SWITCH action requires switch_index.")
            if not (0 <= action.switch_index < cls.NUM_BENCH):
                raise ValueError(f"Invalid switch_index: {action.switch_index}")
            return cls.NUM_MOVES * 2 + int(action.switch_index)

        raise ValueError(f"Unknown action kind: {action.kind}")


# =========================
# 간단한 self-test
# =========================

if __name__ == "__main__":
    print("Action space size:", BSSActionSpace.size())

    # 모든 index에 대해 양방향 매핑이 잘 되는지 확인
    for idx in range(BSSActionSpace.size()):
        act = BSSActionSpace.index_to_action(idx)
        back_idx = BSSActionSpace.action_to_index(act)
        print(f"idx={idx} -> {act} -> back_idx={back_idx}")
        assert back_idx == idx, f"Round-trip mismatch: {idx} != {back_idx}"

    print("All action index <-> action mappings are consistent.")
