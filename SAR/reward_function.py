# reward_function.py
"""
Reward function for the Reg I Koraidon / Calyrex-S / Flutter Mane /
Chien-Pao / Dondozo / Ting-Lu team.

이 모듈은 "환경이 이미 전투 로그를 해석해서 준 상태"를 입력으로 받아
리워드를 계산하는 쪽에 집중한다.
즉, RL 환경에서 전투 로그를 파싱해서 TurnInfo에 값/플래그를 채워 넣고,
여기서는 그 숫자/플래그를 가지고 점수만 계산하는 구조다.

핵심 구성:
- 에피소드 리워드:
    R_total = R_result + sum_t r_t
- R_result: 승패 + 잔여 마릿수
- r_t: 데미지/KO 기반 기본 리워드 + 역할/테라 기반 shaping 리워드
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# =========================
# 기본 파라미터
# =========================

ALPHA_DAMAGE = 1.0  # damage shaping factor
BETA_STATUS = 0.5   # reward for inflicting / penalizing for receiving status
BETA_STAGE = 0.2    # reward scale for rank (stat stage) advantage changes


# =========================
# 데이터 구조 정의
# =========================

@dataclass
class TurnInfo:
    """
    한 턴에서 리워드를 계산하기 위해 필요한 고수준 이벤트 정보.

    환경 쪽에서 "전투 로그 → TurnInfo" 로 변환해서 넘겨주어야 한다.
    이 구조체에 값이 제대로 들어오기만 하면,
    아래의 reward 계산은 그대로 동작한다.

    hp_* 값은 0~1 범위로 정규화된 값이어야 한다.
    """

    # ---- 공통 / 기본 통계 ----
    turn_index: int                  # 1-based 턴 번호
    total_turns_estimate: int        # 에피소드 전체 길이의 추정치 (초반/후반 구분용)

    hp_loss_opp: float               # 이 턴에 상대 전체 HP 감소량 (0~1 합)
    hp_loss_self: float              # 이 턴에 내 전체 HP 감소량 (0~1 합)
    num_opp_KO: int                  # 이 턴에 잡은 상대 마릿수
    num_self_KO: int                 # 이 턴에 내가 잃은 마릿수

    # ---- Status / Rank events ----
    status_inflicted_on_opp: int = 0         # 새로 건 상태 이상 수
    status_received_self: int = 0            # 내가 새로 걸린 상태 이상 수
    stage_advantage_self: float = 0.0        # 내 능력치 랭크 총합 변화(>0 상승, <0 하락)
    stage_advantage_opp: float = 0.0         # 상대 능력치 랭크 총합 변화

    # ====================
    # Koraidon 관련 플래그
    # ====================
    koraidon_on_field: bool = False

    # 초반 핵심 벽(팅루/도도조/하마/가고일류)을 크게 긁었는지
    koraidon_hit_key_wall_this_turn: bool = False
    koraidon_cumulative_key_wall_damage: float = 0.0  # 누적 비율 (0~1), 환경에서 관리

    # Flame Charge 스피드 상승
    koraidon_used_flame_charge_this_turn: bool = False
    koraidon_has_speed_boost_from_flame: bool = False
    koraidon_KOs_after_flame_boost_this_turn: int = 0

    # Fire Tera 관련
    koraidon_used_tera_fire_this_turn: bool = False
    koraidon_is_tera_fire: bool = False
    koraidon_KOs_with_flare_blitz_after_tera_this_turn: int = 0
    koraidon_blocked_burn_due_to_fire_this_turn: bool = False  # Will-O-Wisp/Scald 등을 맞았는데 화상 X

    # ====================
    # Calyrex-Shadow 관련
    # ====================
    calyrex_on_field: bool = False

    calyrex_used_trick_on_wall_this_turn: bool = False

    calyrex_used_tera_fairy_this_turn: bool = False
    calyrex_is_tera_fairy: bool = False
    calyrex_got_hit_by_ghost_or_dark_this_turn: bool = False
    calyrex_survived_ghost_or_dark_hit_with_hp_ge_20: bool = False

    calyrex_used_draining_kiss_this_turn: bool = False
    calyrex_heal_fraction_from_draining_kiss: float = 0.0  # 0~1
    calyrex_will_survive_next_turn_after_kiss: bool = False

    calyrex_wasted_tera_fairy_this_turn: bool = False  # 환경에서 판단(조건 안 맞는 테라 후 곧 사망 등)

    # ====================
    # Flutter Mane 관련
    # ====================
    flutter_on_field: bool = False

    flutter_used_perish_song_this_turn_vs_boosted_or_bp: bool = False

    flutter_used_tera_normal_this_turn: bool = False
    flutter_is_tera_normal: bool = False
    flutter_immune_ghost_attack_this_turn: bool = False

    flutter_used_perish_after_tera_and_forced_endgame: bool = False
    flutter_wasted_tera_normal_this_turn: bool = False

    # ====================
    # Chien-Pao 관련
    # ====================
    chienpao_on_field: bool = False
    chienpao_hp_fraction_before_action: float = 1.0  # 0~1

    chienpao_used_swords_dance_this_turn: bool = False

    chienpao_used_tera_electric_this_turn: bool = False
    chienpao_is_tera_electric: bool = False
    chienpao_blocked_twave_or_nuzzle_this_turn: bool = False

    chienpao_tera_blast_electric_KO_big_water_or_hooh: bool = False
    chienpao_tera_blast_electric_big_hit_fraction_on_water_or_hooh: float = 0.0  # 0~1

    chienpao_wasted_tera_electric_this_turn: bool = False

    # ====================
    # Dondozo 관련
    # ====================
    dondozo_on_field: bool = False
    dondozo_hp_fraction_before_action: float = 1.0
    dondozo_hp_fraction_after_action: float = 1.0

    dondozo_used_curse_this_turn: bool = False

    dondozo_used_tera_fairy_this_turn: bool = False
    dondozo_is_tera_fairy: bool = False
    dondozo_safely_sustained_with_tera: bool = False  # Tera 후 2턴 동안 공격 맞고 HP>=0.5 유지/회복
    dondozo_clutch_survive_and_action_with_tera: bool = False  # 원래 죽을 공격을 Tera로 버티고 Rest/Curse

    dondozo_wasted_tera_fairy_this_turn: bool = False

    # ====================
    # Ting-Lu 관련
    # ====================
    tinglu_on_field: bool = False

    tinglu_used_ruination_this_turn: bool = False
    tinglu_target_hp_after_ruination: Optional[float] = None  # 0~1, None이면 사용 안 함

    tinglu_used_tera_fire_this_turn: bool = False
    tinglu_is_tera_fire: bool = False
    tinglu_tanked_special_hits_with_tera: bool = False
    tinglu_ruination_plus_survive_with_tera: bool = False

    tinglu_wasted_tera_fire_this_turn: bool = False


# =========================
# 결과 리워드 계산
# =========================

def compute_final_result_reward(win: bool, my_alive: int, opp_alive: int) -> float:
    """
    에피소드 종료 시 한 번만 계산하는 최종 결과 리워드.

    win: True면 승리, False면 패배
    my_alive: 내 남은 포켓몬 수 (0~3)
    opp_alive: 상대 남은 포켓몬 수 (0~3)
    """
    sign = 1 if win else -1
    return 10.0 * sign + float(my_alive - opp_alive)


# =========================
# 턴 단위 리워드 계산
# =========================

def compute_turn_reward(info: TurnInfo) -> float:
    """
    한 턴에 대한 총 리워드를 계산한다.
    기본 데미지/KO 리워드 + 역할/테라 shaping 리워드의 합.
    """
    r = 0.0

    # ----- 1) 기본 데미지/KO 리워드 -----
    r += ALPHA_DAMAGE * (info.hp_loss_opp - info.hp_loss_self)
    r += 3.0 * info.num_opp_KO
    r -= 3.0 * info.num_self_KO

    # ----- 2) 상태 이상 / 랭크 변화 리워드 -----
    r += BETA_STATUS * float(info.status_inflicted_on_opp)
    r -= BETA_STATUS * float(info.status_received_self)
    r += BETA_STAGE * info.stage_advantage_self
    r -= BETA_STAGE * info.stage_advantage_opp

    # ----- 3) 역할/테라 shaping -----
    r += _reward_koraidon(info)
    r += _reward_calyrex(info)
    r += _reward_flutter(info)
    r += _reward_chienpao(info)
    r += _reward_dondozo(info)
    r += _reward_tinglu(info)

    return r


# =========================
# 개별 포켓몬별 shaping
# =========================

def _reward_koraidon(info: TurnInfo) -> float:
    r = 0.0

    # (1) 초반 핵심 벽 긁기 (turn ≤ 5, 누적 데미지 ≥ 0.5)
    if info.koraidon_on_field:
        if info.turn_index <= 5 and info.koraidon_hit_key_wall_this_turn:
            if info.koraidon_cumulative_key_wall_damage >= 0.5:
                r += 2.0

        # (2) Flame Charge -> 스윕
        if info.koraidon_has_speed_boost_from_flame:
            # 이 턴에 Flame Charge 직후 KO 횟수 (환경에서 카운트)
            if info.koraidon_KOs_after_flame_boost_this_turn > 0:
                # 최대 2킬까지만 보상
                r += 2.0 * min(info.koraidon_KOs_after_flame_boost_this_turn, 2)

        # (3) Fire Tera 관련
        if info.koraidon_used_tera_fire_this_turn or info.koraidon_is_tera_fire:
            # Flare Blitz로 KO
            if info.koraidon_KOs_with_flare_blitz_after_tera_this_turn > 0:
                r += 2.0 * info.koraidon_KOs_with_flare_blitz_after_tera_this_turn

            # 화상 무력화
            if info.koraidon_blocked_burn_due_to_fire_this_turn:
                r += 1.0

            # Fire Tera를 썼는데 아무 영향 없이 다음 턴 전 사망했다면
            # -> 이건 환경 쪽에서 플래그/조건으로 설정해도 됨.
            # 여기서는 "wasted" 플래그 대신, 외부에서 판단하도록 열어 둔다.

    return r


def _reward_calyrex(info: TurnInfo) -> float:
    r = 0.0

    if info.calyrex_on_field:
        # (1) Trick로 벽 뚫기
        if info.calyrex_used_trick_on_wall_this_turn:
            r += 3.0

        # (2) Tera Fairy 관련
        if info.calyrex_used_tera_fairy_this_turn or info.calyrex_is_tera_fairy:
            # Ghost/Dark 기술을 맞고도 HP >= 20% 생존
            if info.calyrex_survived_ghost_or_dark_hit_with_hp_ge_20:
                r += 1.5

            # Draining Kiss 회복 + 생존
            if info.calyrex_used_draining_kiss_this_turn:
                if info.calyrex_heal_fraction_from_draining_kiss >= 0.3 and \
                        info.calyrex_will_survive_next_turn_after_kiss:
                    r += 1.5

            # 완전 헛테라 (환경에서 플래그로 판정)
            if info.calyrex_wasted_tera_fairy_this_turn:
                r -= 2.0

    return r


def _reward_flutter(info: TurnInfo) -> float:
    r = 0.0

    if info.flutter_on_field:
        # (1) Perish Song으로 세팅/지구전 억제
        if info.flutter_used_perish_song_this_turn_vs_boosted_or_bp:
            r += 2.0

        # (2) Normal Tera 관련
        if info.flutter_used_tera_normal_this_turn or info.flutter_is_tera_normal:
            # Ghost 공격 완전 무효
            if info.flutter_immune_ghost_attack_this_turn:
                r += 2.5

            # Tera Normal + Perish로 엔드게임 강제
            if info.flutter_used_perish_after_tera_and_forced_endgame:
                r += 1.5

            # 헛테라
            if info.flutter_wasted_tera_normal_this_turn:
                r -= 2.0

    return r


def _reward_chienpao(info: TurnInfo) -> float:
    r = 0.0

    if info.chienpao_on_field:
        # (1) Swords Dance (HP >= 0.5에서 안전하게 쌓기)
        if info.chienpao_used_swords_dance_this_turn and \
                info.chienpao_hp_fraction_before_action >= 0.5:
            r += 1.0

        # (2) Electric Tera 관련
        if info.chienpao_used_tera_electric_this_turn or info.chienpao_is_tera_electric:
            # Thunder Wave / Nuzzle 무력화
            if info.chienpao_blocked_twave_or_nuzzle_this_turn:
                r += 1.5

            # 물/비행 막이 브레이크
            if info.chienpao_tera_blast_electric_KO_big_water_or_hooh:
                r += 3.0
            else:
                if info.chienpao_tera_blast_electric_big_hit_fraction_on_water_or_hooh >= 0.7:
                    r += 1.5

            # 헛테라
            if info.chienpao_wasted_tera_electric_this_turn:
                r -= 1.5

    return r


def _reward_dondozo(info: TurnInfo) -> float:
    r = 0.0

    if info.dondozo_on_field:
        # (1) Curse 쌓기 (HP >= 0.6일 때만 보상)
        if info.dondozo_used_curse_this_turn and info.dondozo_hp_fraction_before_action >= 0.6:
            r += 0.5

        # (2) Fairy Tera 관련 — 보상 작게
        if info.dondozo_used_tera_fairy_this_turn or info.dondozo_is_tera_fairy:
            if info.dondozo_safely_sustained_with_tera:
                # 테라 덕분에 2턴 버티며 HP를 50% 이상으로 관리
                r += 0.5

            if info.dondozo_clutch_survive_and_action_with_tera:
                # 원래 죽을 공격을 테라로 버티고 Curse/Rest를 쓴 클러치 상황
                r += 1.0

            if info.dondozo_wasted_tera_fairy_this_turn:
                r -= 2.0

    return r


def _reward_tinglu(info: TurnInfo) -> float:
    r = 0.0

    if info.tinglu_on_field:
        # (1) Ruination으로 상대 HP를 50% 이하로 낮췄는지
        if info.tinglu_used_ruination_this_turn and info.tinglu_target_hp_after_ruination is not None:
            if info.tinglu_target_hp_after_ruination <= 0.5:
                r += 0.5

        # (2) Fire Tera 관련 — 보상 작게
        if info.tinglu_used_tera_fire_this_turn or info.tinglu_is_tera_fire:
            if info.tinglu_tanked_special_hits_with_tera:
                r += 0.5

            if info.tinglu_ruination_plus_survive_with_tera:
                r += 0.5

            if info.tinglu_wasted_tera_fire_this_turn:
                r -= 1.5

    return r


# =========================
# 사용 예시 (환경에서)
# =========================
if __name__ == "__main__":
    # 간단한 예시: 아무 일도 안 일어난 턴
    dummy_turn = TurnInfo(
        turn_index=1,
        total_turns_estimate=10,
        hp_loss_opp=0.0,
        hp_loss_self=0.0,
        num_opp_KO=0,
        num_self_KO=0,
    )
    print("Example empty turn reward:", compute_turn_reward(dummy_turn))

    # 예시: 코라이돈이 초반에 벽을 크게 긁은 턴
    example_turn = TurnInfo(
        turn_index=3,
        total_turns_estimate=10,
        hp_loss_opp=0.3,
        hp_loss_self=0.1,
        num_opp_KO=0,
        num_self_KO=0,
        koraidon_on_field=True,
        koraidon_hit_key_wall_this_turn=True,
        koraidon_cumulative_key_wall_damage=0.6,
    )
    print("Example Koraidon wall-break turn reward:", compute_turn_reward(example_turn))
