# 포켓몬 강화학습 (Regulation I, Masked PPO)

이 디렉터리는 포켓몬 스카렛/바이올렛 배틀 스타디움 싱글(3v3, Regulation I)을 위한 강화학습 파이프라인입니다. poke-env로 쇼다운 서버와 통신하고, sb3-contrib의 MaskablePPO로 불법 액션을 마스킹하며 학습합니다.

## 들어가기 앞서
  ```bash
  ```node pokemon-showdown start --no-security```
  ```
 를 통하여 서버에 접속을 해야 학습이 가능합니다.


## 핵심 개념
- **상태(164차원)**: `reg_i_player.py`의 `embed_battle`가 생성. 액티브/벤치 HP·상태·랭크, 기술 위력/상성/명중/PP, 테라 가능 여부, 함정/필드/날씨, 팀 생존 수, 턴 정보 등을 포함합니다. 상세는 `state_vector.md` 참조.
- **행동(10 이산)**: 4개 기술, 4개 테라+기술, 2개 교체. `get_action_mask`로 불법 액션(강제 교체, 중복 함정/상태기, 테라 불가 등)을 0으로 마스킹합니다. 상세는 `action.md` 참조.
- **보상**: `rewards.py`의 `RegIRewardEngine`을 사용해 승패/HP/KO, 랭크 변화, 상태/하품, 함정, 무효 상성 패널티, 턴 비용 등을 포함한 shaping을 수행합니다. 상세는 `reward.md` 참조.
- **정책**: `train2.py`에서 MaskablePPO(2층 MLP 256-256)를 사용. 텐서보드 로그와 체크포인트는 `outputs/`에 저장됩니다. 개요는 `policy.md` 참조.

## 팀 구성
`reg_i_team_builder.py`에서 정의한 고정 파티(6마리). 기본 팀프리뷰는 `/team 123456`으로 1~3번 슬롯을 출전시킵니다. 필요하면 `RegIPlayer.set_team_order("/team ...")`로 한 번에 한해 커스텀 순서를 지정할 수 있습니다.

## 주요 스크립트
- **학습**: `train.py`  
  ```bash
  CUDA_VISIBLE_DEVICES=0 python train.py \
    --log-frequency 100   # 에피소드 통계 출력 주기
  ```
  - Self-play 환경 + ActionMasker + CheckpointCallback + EpisodeStatsCallback 사용.
  - 체크포인트/텐서보드 경로: `outputs/checkpoints_regi`, `outputs/tensorboard_regi`

- **랜덤 상대 평가**: `evaluate_model.py`  
  ```bash
  python evaluate_model.py --model outputs/checkpoints_regi_v2/train2_final_model.zip --n-battles 200
  ```
  - 고정 팀 vs 랜덤 팀/랜덤 플레이어.

- **모델 vs 모델 평가**: `evaluate.py` (한 방향)  
  ```bash
  python evaluate.py \
    --model outputs/checkpoints_regi_v2/train2_final_model.zip \
    --opponent-model outputs/checkpoints_regi_v2/train2_model_50000_steps.zip \
    --n-battles 200
  ```
  - 동일 팀으로 한쪽이 환경, 한쪽이 상대 모델로 동작.

- **배틀 관전**: `watch_battles.py`  
  - 체크포인트를 로드해 랜덤/모델 상대와 배틀을 진행하며 로그 출력.

## 환경 구성
1) Python 패키지  
   ```bash
   pip install poke-env stable-baselines3 sb3-contrib torch tensorboard gymnasium
   ```
2) 쇼다운 서버 실행(포그라운드 예시)  
   ```bash
   cd ../pokemon-showdown
   node pokemon-showdown start --no-security
   ```
3) 텐서보드  
   ```bash
   tensorboard --logdir reg_i_rl/outputs/tensorboard_regi_v2 --port 6006 --bind_all
   ```
   브라우저에서 `http://<호스트>:6006` 접속.

## 파일 구조 (주요)
```
reg_i_rl/
├── train2.py                 # 학습 엔트리포인트 (Masked PPO + self-play)
├── reg_i_player.py           # 상태/행동/마스킹/보상 계산을 포함한 환경 래퍼
├── reg_i_team_builder.py     # 팀 정의
├── rewards.py                # 고급 보상 엔진
├── evaluate.py               # 모델 vs 모델 단방향 평가
├── evaluate_model.py         # 랜덤 상대 평가
└── watch_battles.py          # 배틀 디버그
```