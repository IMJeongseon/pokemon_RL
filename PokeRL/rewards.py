import logging
from typing import Dict, Optional, Tuple, Any
from poke_env.battle import AbstractBattle
from poke_env.battle.effect import Effect
from poke_env.battle.side_condition import SideCondition
from poke_env.data import GenData

class RegIRewardEngine:
    """
    Regulation I Reward Engine (Final)

    [반영된 로직]
    1. Potential-based 랭크 시스템 (초기화/복구 개념 적용)
    2. 상태이상/하품 대처 로직 (교체 유도)
    3. 턴 비용(Turn Cost) 적용 (스톨링 방지, 먹밥 회복량 고려하여 -0.05 설정)
    4. 타입 상성 패널티 (무효 기술 사용 시 -0.5 패널티)
    5. 방어 교체/테라 보상 (상대 기술을 교체/테라로 회피 시 보상)
    """

# 가중치 설정 (Revised)
    W_WIN = 100.0          
    W_LOSS = -100.0        
    W_KO = 5.0             
    W_HP = 1.0             
    W_RANK = 0.8           # 1.0 -> 0.8 (랭크업 남발 방지, 공격 유도)
    W_STATUS = 2.0         # 3.0 -> 2.0 (상태이상보다 KO가 더 중요함)
    W_HAZARD = 1.5         # 2.0 -> 1.5 (장판보다 깡딜이 더 중요할 때가 많음)
    
    # 패널티 설정
    P_TURN_LOST = -2.0
    P_RISKY_SETUP = -3.0
    P_YAWN_SLEEP = -5.0
    P_TURN_COST = -0.05
    P_INEFFECTIVE_MOVE = -2.0  # 무효 타입 기술 사용 패널티

    # 보상 설정
    R_IMMUNITY_SWITCH = 1.0  # 상성 교체로 공격 회피 보상
    R_IMMUNITY_TERA = 1.5    # 테라스탈로 공격 회피 보상

    # 패널티 설정 (추가)
    P_PERISH_FAINT = -3.0   # 멸망의 노래로 기절 시 큰 패널티

    # 임계값
    HP_DANGER_THRESHOLD = 0.25 # 0.35 -> 0.25 (딸피 역전 가능성 열어둠)
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self._battle_states = {}

    def reset_battle(self, battle_id: str):
        if battle_id in self._battle_states:
            del self._battle_states[battle_id]

    def compute_reward(self, battle: AbstractBattle) -> float:
        battle_id = getattr(battle, "battle_tag", str(id(battle)))
        
        # 1. 초기 상태 설정
        if battle_id not in self._battle_states:
            self._init_battle_state(battle_id, battle)
            return 0.0

        prev_state = self._battle_states[battle_id]
        total_reward = 0.0
        
        # 2. 승패 보상
        if battle.finished:
            # 이겼을 때 빨리 이기는 게 좋으므로 턴 비용 누적을 피하기 위해 여기서 반환
            total_reward = self.W_WIN if battle.won else self.W_LOSS
            self.reset_battle(battle_id)
            return total_reward

        # === 세부 보상 계산 ===

        # [HP]
        total_reward += self._calc_hp_reward(battle, prev_state)

        # [Rank]
        total_reward += self._calc_rank_reward(battle, prev_state)

        # [Status & Turn]
        total_reward += self._calc_turn_loss_penalty(battle, prev_state)
        total_reward += self._calc_status_reward(battle, prev_state)
        total_reward += self._calc_yawn_logic(battle, prev_state)

        # [Risk]
        total_reward += self._calc_risky_play_penalty(battle, prev_state)
        
        # [Field]
        total_reward += self._calc_field_reward(battle, prev_state)

        # [Immunity & Type Effectiveness]
        total_reward += self._calc_immunity_switch_reward(battle, prev_state)
        total_reward += self._calc_ineffective_move_penalty(battle, prev_state)

        # [Perish Song]
        # total_reward += self._calc_perish_song_logic(battle, prev_state)

        # [Time Cost] ★ 여기가 추가된 부분입니다! ★
        # 매 턴 고정 비용을 부과하여 "아무것도 안 하면 손해"임을 학습시킴
        total_reward += self.P_TURN_COST

        # 상태 업데이트
        self._update_state(battle_id, battle)

        return total_reward

    def _init_battle_state(self, battle_id, battle):
        """초기 상태 저장"""
        self._battle_states[battle_id] = {
            "my_hp": {mon.species: mon.current_hp_fraction for mon in battle.team.values()},
            "opp_hp": {mon.species: mon.current_hp_fraction for mon in battle.opponent_team.values()},

            "rank_score": self._get_total_rank_score(battle),

            "opp_hazards": 0,
            "active_species": battle.active_pokemon.species if battle.active_pokemon else None,
            "yawn_turn": None,

            # [수정] KeyError 방지를 위해 초기화 키 추가
            "last_switch_turn": -1,

            # 타입 상성 체크를 위한 추가 정보
            "my_last_move": None,           # 내가 마지막으로 사용한 기술
            "opp_last_move": None,          # 상대가 마지막으로 사용한 기술
            "prev_active_types": None,      # 이전 턴 내 포켓몬 타입 (교체 감지용)
            "prev__terastallized": False,    # 이전 턴 테라스탈 상태

            # 멸망의 노래 추적
            "my_perish_count": 0,           # 내 활성 포켓몬의 멸망의 노래 카운트
            "opp_perish_count": 0,          # 상대 활성 포켓몬의 멸망의 노래 카운트
        }

    def _update_state(self, battle_id, battle):
        """현재 턴 정보를 저장"""
        state = self._battle_states[battle_id]

        # 교체 여부 확인 (이번 턴에 포켓몬이 바뀌었으면 last_switch_turn 갱신)
        current_species = battle.active_pokemon.species if battle.active_pokemon else None
        if current_species != state["active_species"]:
            state["last_switch_turn"] = battle.turn
        state["active_species"] = current_species

        state["my_hp"] = {mon.species: mon.current_hp_fraction for mon in battle.team.values()}
        state["opp_hp"] = {mon.species: mon.current_hp_fraction for mon in battle.opponent_team.values()}

        state["rank_score"] = self._get_total_rank_score(battle)
        state["opp_hazards"] = self._count_hazards(battle.opponent_side_conditions)

        active = battle.active_pokemon
        if active and Effect.YAWN in getattr(active, "effects", {}):
            if state["yawn_turn"] is None:
                state["yawn_turn"] = battle.turn
        else:
            state["yawn_turn"] = None

        # 타입 상성 체크를 위한 정보 업데이트
        if active:
            state["prev_active_types"] = active.types
            state["prev__terastallized"] = active._terastallized

        # 멸망의 노래 카운트 업데이트
        # state["my_perish_count"] = self._get_perish_count(battle.active_pokemon)
        # state["opp_perish_count"] = self._get_perish_count(battle.opponent_active_pokemon)

        # 사용된 기술 추적 (battle history에서 가져옴)
        # poke-env에서는 move history를 직접 제공하지 않으므로
        # 이 부분은 battle 객체에서 가능한 정보를 활용
        # Note: 실제 사용된 move는 턴 단위로 추적이 어려우므로,
        # 패널티는 action 선택 시점이 아닌 결과 시점에서 판단

    # === Logic Implementation ===

    def _calc_rank_reward(self, battle, prev_state):
        current_score = self._get_total_rank_score(battle)
        prev_score = prev_state["rank_score"]
        delta = current_score - prev_score
        return delta * self.W_RANK

    def _get_total_rank_score(self, battle):
        my_score = 0.0
        opp_score = 0.0
        
        if battle.active_pokemon and not battle.active_pokemon.fainted:
            my_score = self._calculate_diminishing_rank_value(battle.active_pokemon.boosts)
            
        if battle.opponent_active_pokemon and not battle.opponent_active_pokemon.fainted:
            opp_score = self._calculate_diminishing_rank_value(battle.opponent_active_pokemon.boosts)
            
        return my_score - opp_score

    def _calculate_diminishing_rank_value(self, boosts: Dict[str, int]) -> float:
        total_value = 0.0
        for stat, stage in boosts.items():
            if stage == 0: continue
            
            value = 0.0
            for i in range(1, abs(stage) + 1):
                value += 1.0 / i 
            
            if stage < 0:
                total_value -= value
            else:
                total_value += value
                
        return total_value

    def _calc_hp_reward(self, battle, prev_state):
        reward = 0.0
        # 내 체력
        for mon in battle.team.values():
            curr = mon.current_hp_fraction
            prev = prev_state["my_hp"].get(mon.species, curr)
            reward += (curr - prev) * self.W_HP

        # 상대 체력
        for mon in battle.opponent_team.values():
            curr = mon.current_hp_fraction
            prev = prev_state["opp_hp"].get(mon.species, curr)
            reward -= (curr - prev) * self.W_HP * 1.5 
            
            if prev > 0 and mon.fainted:
                reward += self.W_KO
        return reward

    def _calc_turn_loss_penalty(self, battle, prev_state):
        active = battle.active_pokemon
        if not active or not active.status: return 0.0
        
        status_name = active.status.name
        if status_name in ["SLP", "PAR", "FRZ"]:
            opp_hp_delta = 0
            for mon in battle.opponent_team.values():
                prev = prev_state["opp_hp"].get(mon.species, mon.current_hp_fraction)
                opp_hp_delta += (mon.current_hp_fraction - prev)
            
            if opp_hp_delta >= 0: 
                return self.P_TURN_LOST
        return 0.0

    def _calc_status_reward(self, battle, prev_state):
        return 0.0

    def _calc_yawn_logic(self, battle, prev_state):
        reward = 0.0
        prev_yawn = (prev_state["yawn_turn"] is not None)
        active = battle.active_pokemon
        
        if not active: return 0.0
        current_slp = (active.status and active.status.name == "SLP")
        
        if prev_yawn and active.species != prev_state["active_species"]:
            reward += 2.0
        if prev_yawn and current_slp and active.species == prev_state["active_species"]:
            reward += self.P_YAWN_SLEEP
        return reward

    def _calc_risky_play_penalty(self, battle, prev_state):
        active = battle.active_pokemon
        if not active: return 0.0
        
        if active.current_hp_fraction <= self.HP_DANGER_THRESHOLD:
            curr_score = self._get_total_rank_score(battle)
            prev_score = prev_state["rank_score"]
            if curr_score > prev_score:
                return self.P_RISKY_SETUP
        return 0.0

    def _calc_field_reward(self, battle, prev_state):
        reward = 0.0
        curr_hazards = self._count_hazards(battle.opponent_side_conditions)
        if curr_hazards > prev_state["opp_hazards"]:
            reward += self.W_HAZARD * (curr_hazards - prev_state["opp_hazards"])
        return reward

    def _count_hazards(self, side_conditions):
        count = 0
        if SideCondition.STEALTH_ROCK in side_conditions: count += 1
        if SideCondition.SPIKES in side_conditions: count += side_conditions[SideCondition.SPIKES]
        if SideCondition.TOXIC_SPIKES in side_conditions: count += 1
        if SideCondition.STICKY_WEB in side_conditions: count += 1
        return count
    
    # === 타입 상성 로직 ===

    def _calc_ineffective_move_penalty(self, battle, prev_state):
        """
        무효 타입 기술 사용 시 패널티 적용
        예: 땅타입에게 전기기술, 불타입에게 불타입 공격기, 페어리에게 드래곤기술 등
        """
        reward = 0.0
        active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon

        if not active or not opp_active:
            return 0.0

        # 내 포켓몬의 HP가 감소했는지 확인 (공격을 시도했다는 신호)
        prev_hp = prev_state["my_hp"].get(active.species, active.current_hp_fraction)

        # 상대 포켓몬의 HP 변화 확인
        prev_opp_hp = prev_state["opp_hp"].get(opp_active.species, opp_active.current_hp_fraction)
        curr_opp_hp = opp_active.current_hp_fraction

        # 상대 HP가 변하지 않았고, 내가 행동했다면 (턴이 진행됨)
        # 이는 무효 타입이거나, miss거나, 또는 비공격 기술을 사용했을 가능성
        # 정확한 판단을 위해서는 실제 사용한 기술 정보가 필요하지만,
        # poke-env의 제약으로 인해 간접적인 방법 사용

        # 내 포켓몬이 가진 공격 기술들을 확인하고, 상대 타입에 무효인 기술이 있는지 체크
        if active.moves:
            for move_id, move in active.moves.items():
                if move.base_power > 0:  # 공격 기술인 경우
                    effectiveness = self._get_type_effectiveness(move, opp_active)

                    # 만약 이 기술이 무효(0배)라면, 그리고 상대 HP가 변하지 않았다면
                    if effectiveness == 0 and curr_opp_hp == prev_opp_hp:
                        # 상대가 기절하지 않았고, 내 HP도 크게 변하지 않았다면
                        # (반동 데미지 등 예외 케이스 제외)
                        if not opp_active.fainted and abs(active.current_hp_fraction - prev_hp) < 0.3:
                            # 무효 기술 사용으로 추정
                            reward += self.P_INEFFECTIVE_MOVE
                            break  # 한 번만 패널티 적용

        return reward

    def _calc_immunity_switch_reward(self, battle, prev_state):
        """
        상대방의 기술을 교체 또는 테라스탈로 회피/감소시켰을 때 보상
        """
        reward = 0.0
        active = battle.active_pokemon

        if not active:
            return 0.0

        # 1. 교체로 인한 회피 감지
        if battle.turn == prev_state.get("last_switch_turn", -1):
            # 이번 턴에 교체했음
            # 이전 포켓몬의 타입과 현재 포켓몬의 타입을 비교
            prev_types = prev_state.get("prev_active_types")

            if prev_types and active.types:
                # 교체 후 타입이 바뀌었고, 상대의 공격을 더 잘 받을 수 있게 되었는지 확인
                # 간단한 휴리스틱: 교체 후 살아있고, HP가 90% 이상이면 성공적인 교체로 간주
                if active.current_hp_fraction > 0.9:
                    reward += self.R_IMMUNITY_SWITCH

        # 2. 테라스탈로 인한 타입 변경 감지
        prev_tera = prev_state.get("prev__terastallized", False)
        curr_tera = active._terastallized

        if not prev_tera and curr_tera:
            # 이번 턴에 테라스탈을 사용했음
            # 테라스탈 후 타입이 바뀌어 방어 상성이 개선되었는지 확인
            # 간단한 휴리스틱: 테라 후 HP 손실이 적으면 성공적인 방어로 간주
            prev_hp = prev_state["my_hp"].get(active.species, active.current_hp_fraction)
            hp_loss = prev_hp - active.current_hp_fraction

            if hp_loss < 0.3:  # 테라 턴에 큰 피해를 받지 않았다면
                reward += self.R_IMMUNITY_TERA

        return reward

    def _get_type_effectiveness(self, move, target_pokemon):
        """
        기술의 타입 상성 계산
        Returns: 배수 (0, 0.25, 0.5, 1, 2, 4)
        """
        if not move or not target_pokemon:
            return 1.0

        move_type = move.type
        if not move_type:
            return 1.0

        # 타겟의 타입들 가져오기
        target_types = target_pokemon.types
        if not target_types:
            return 1.0

        try:
            # poke-env의 damage_multiplier 메서드 사용
            # Move 객체에 직접 damage_multiplier 메서드가 있을 수 있음
            if hasattr(move, 'damage_multiplier'):
                return move.damage_multiplier(target_pokemon)

            # 또는 타입별로 직접 계산
            effectiveness = 1.0
            for target_type in target_types:
                if target_type and move_type:
                    # PokemonType 객체의 damage_multiplier 메서드 사용
                    if hasattr(move_type, 'damage_multiplier'):
                        multiplier = move_type.damage_multiplier(target_type)
                        effectiveness *= multiplier

            return effectiveness

        except Exception as e:
            # 타입 차트 접근 실패 시 기본값 사용
            if self.verbose:
                print(f"[Type Effectiveness] Error calculating effectiveness: {e}")
            return 1.0