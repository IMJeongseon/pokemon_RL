"""
Training script for Regulation I BSS agent using Masked PPO.
Uses rewards.py engine with train1.py's stable structure.
"""

import os
import argparse
from typing import Callable, Union
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from reg_i_player import RegIPlayer
from reg_i_team_builder import RegITeamBuilder
from self_play_opponent import SelfPlayOpponent
from poke_env.player import RandomPlayer
from poke_env.environment import SingleAgentWrapper
from poke_env.teambuilder import Teambuilder, ConstantTeambuilder

# Import the advanced reward engine
from rewards import RegIRewardEngine


class TrainingConfig:
    BATTLE_FORMAT = "gen9bssregj"

    # Training hyperparameters
    TOTAL_TIMESTEPS = 500_000
    LEARNING_RATE = 2e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.1
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    POLICY_KWARGS = dict(net_arch=[256, 256])

    # Logging / saving
    SAVE_DIR = "./outputs/checkpoints_regi"
    TENSORBOARD_LOG = "./outputs/tensorboard_regi"
    CHECKPOINT_FREQ = 50_000
    CHECKPOINT_NAME = "train2_model"

    # Verbose logging
    LOG_EPISODE_STATS = True
    LOG_FREQUENCY = 100
    POKE_ENV_LOG_LEVEL = 25

    # Self-play settings
    SELF_PLAY_UPDATE_FREQ = 10_000


# =====================================================================
#  Custom Callback for Episode Logging
# =====================================================================

class EpisodeStatsCallback(BaseCallback):
    """Callback for logging detailed episode statistics."""

    def __init__(self, log_frequency: int = 100, total_timesteps: int = 500_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.total_timesteps = total_timesteps

        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_outcomes = []

        # Current episode
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        # Progress bar
        self.pbar = None

    def _on_training_start(self) -> None:
        """Initialize progress bar at training start."""
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", unit="steps",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    def _on_step(self) -> bool:
        """Called at every step."""
        if self.pbar is not None:
            self.pbar.update(1)
            if self.episode_count > 0:
                recent_n = min(self.log_frequency, len(self.episode_outcomes))
                if recent_n > 0:
                    win_rate = sum(self.episode_outcomes[-recent_n:]) / recent_n * 100
                    avg_reward = sum(self.episode_rewards[-recent_n:]) / recent_n
                    self.pbar.set_postfix({
                        'Episodes': self.episode_count,
                        'Win%': f'{win_rate:.1f}',
                        'AvgR': f'{avg_reward:.2f}'
                    })

        infos = self.locals.get('infos', [])

        for info in infos:
            self.current_episode_reward += self.locals.get('rewards', [0])[0]
            self.current_episode_length += 1

            if self.locals.get('dones', [False])[0]:
                self.episode_count += 1

                outcome = 1 if self.current_episode_reward > 0 else 0

                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.episode_outcomes.append(outcome)

                if self.episode_count % self.log_frequency == 0:
                    self._print_episode_stats()

                self.current_episode_reward = 0.0
                self.current_episode_length = 0

        return True

    def _on_training_end(self) -> None:
        """Close progress bar at training end."""
        if self.pbar is not None:
            self.pbar.close()

    def _print_episode_stats(self):
        """Print statistics for recent episodes."""
        recent_n = min(self.log_frequency, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-recent_n:]
        recent_lengths = self.episode_lengths[-recent_n:]
        recent_outcomes = self.episode_outcomes[-recent_n:]

        win_rate = sum(recent_outcomes) / len(recent_outcomes) * 100 if recent_outcomes else 0.0
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
        avg_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0.0

        tqdm.write("\n" + "="*60)
        tqdm.write(f"EPISODE {self.episode_count} STATS (last {recent_n} episodes)")
        tqdm.write("="*60)
        tqdm.write(f"Win Rate:       {win_rate:.1f}% ({sum(recent_outcomes)}/{len(recent_outcomes)})")
        tqdm.write(f"Avg Reward:     {avg_reward:.2f}")
        tqdm.write(f"Avg Length:     {avg_length:.1f} steps")
        tqdm.write(f"Reward Range:   [{min(recent_rewards):.2f}, {max(recent_rewards):.2f}]")
        tqdm.write(f"Total Episodes: {self.episode_count}")
        tqdm.write(f"Total Steps:    {self.num_timesteps:,}")
        tqdm.write("="*60 + "\n")


# =====================================================================
#  Self-Play Callback
# =====================================================================

class SelfPlayCallback(BaseCallback):
    """Callback for updating opponent's model during self-play training."""

    def __init__(self, opponent: SelfPlayOpponent, update_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.opponent = opponent
        self.update_freq = update_freq
        self.last_update = 0

    def _on_step(self) -> bool:
        """Called at every step. Update opponent model periodically."""
        if self.num_timesteps - self.last_update >= self.update_freq:
            self.opponent.update_model(self.model)
            self.last_update = self.num_timesteps

            if self.verbose > 0:
                tqdm.write(f"\n[SelfPlay] Updated opponent model at step {self.num_timesteps}")

        return True


# =====================================================================
#  SmartRewardRegIPlayer - Uses RegIRewardEngine
# =====================================================================

class SmartRewardRegIPlayer(RegIPlayer):
    """
    RegIPlayer with advanced reward engine from rewards.py.
    Uses potential-based ranks, turn cost, status penalties, etc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_engine = RegIRewardEngine(verbose=False)

    def compute_reward(self, battle):
        """Override to use reward engine."""
        return self.reward_engine.compute_reward(battle)

    def _cleanup_battle_tracking(self, battle):
        """Clean up battle tracking in both base and reward engine."""
        super()._cleanup_battle_tracking(battle)
        battle_id = getattr(battle, "battle_tag", str(id(battle)))
        self.reward_engine.reset_battle(battle_id)


class DualTeamSmartPlayer(SmartRewardRegIPlayer):
    """
    Dual-team variant (from train1.py) that allows agent1 and agent2 to use different teams.
    """

    def __init__(
        self,
        team1: Union[str, Teambuilder],
        team2: Union[str, Teambuilder],
        **kwargs,
    ):
        super().__init__(team=team1, **kwargs)
        if isinstance(team2, str):
            opponent_team = ConstantTeambuilder(team2)
        else:
            opponent_team = team2
        self.agent2._team = opponent_team
        print(f"[DualTeam] agent1 team: {type(self.agent1._team).__name__}")
        print(f"[DualTeam] agent2 team: {type(self.agent2._team).__name__}")


# =====================================================================
#  Gymnasium Wrapper
# =====================================================================

class GymnasiumWrapper(gym.Env):
    """Wrap poke-env SingleAgentWrapper to Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(self, single_agent_env: SingleAgentWrapper):
        super().__init__()
        self.env = single_agent_env
        self.observation_space = single_agent_env.env.observation_spaces[
            single_agent_env.env.possible_agents[0]
        ]
        self.action_space = single_agent_env.env.action_spaces[
            single_agent_env.env.possible_agents[0]
        ]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        return self.env.env.get_action_mask()


# =====================================================================
#  Environment Factory
# =====================================================================

_global_opponent = None


def make_env(opponent_type: str = "selfplay", with_masking: bool = True) -> Callable[[], gym.Env]:
    """
    Factory for a single SB3-compatible environment.
    """

    def _init() -> gym.Env:
        global _global_opponent

        from random_opponent_builder import RandomOpponentTeamBuilder

        # Player team (fixed)
        player_team = RegITeamBuilder()

        # Opponent team (random)
        opponent_team = RandomOpponentTeamBuilder()

        # Training player (DualTeam for different teams)
        train_player = DualTeamSmartPlayer(
            team1=player_team,
            team2=opponent_team,
            battle_format=TrainingConfig.BATTLE_FORMAT,
            log_level=TrainingConfig.POKE_ENV_LOG_LEVEL,
            start_listening=True,
            strict=False,
        )

        # Opponent
        if opponent_type == "selfplay":
            opponent = SelfPlayOpponent(
                env_for_embed=train_player,
                battle_format=TrainingConfig.BATTLE_FORMAT,
                log_level=TrainingConfig.POKE_ENV_LOG_LEVEL,
                start_listening=False,
            )
            _global_opponent = opponent
            print("[SelfPlay] Created self-play opponent with diverse teams")
        else:
            opponent = RandomPlayer(
                battle_format=TrainingConfig.BATTLE_FORMAT,
                log_level=TrainingConfig.POKE_ENV_LOG_LEVEL,
                start_listening=False,
            )
            print("[Opponent] Created random opponent with diverse teams")

        # Wrap environment
        single_agent_env = SingleAgentWrapper(train_player, opponent)
        base_env = GymnasiumWrapper(single_agent_env)

        # Apply action masking
        if with_masking:
            base_env = ActionMasker(base_env, mask_fn)

        return base_env

    return _init


def mask_fn(env: gym.Env) -> np.ndarray:
    """Action mask function for ActionMasker."""
    if hasattr(env, "get_action_mask"):
        return env.get_action_mask()
    return np.ones(10, dtype=np.float32)


# =====================================================================
#  Training Loop
# =====================================================================

def train_masked_ppo():
    """Main training loop for Masked PPO."""
    print("=" * 60)
    print("Train2 - Advanced Reward Engine + Stable Structure")
    print("=" * 60)

    # Create directories
    os.makedirs(TrainingConfig.SAVE_DIR, exist_ok=True)
    os.makedirs(TrainingConfig.TENSORBOARD_LOG, exist_ok=True)

    # Environment setup
    print("\n[Env] Creating environment with Self-Play opponent...")
    env = DummyVecEnv([make_env("selfplay", with_masking=True)])

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=TrainingConfig.CHECKPOINT_FREQ // env.num_envs,
        save_path=TrainingConfig.SAVE_DIR,
        name_prefix=TrainingConfig.CHECKPOINT_NAME,
    )

    episode_stats_callback = EpisodeStatsCallback(
        log_frequency=TrainingConfig.LOG_FREQUENCY,
        total_timesteps=TrainingConfig.TOTAL_TIMESTEPS,
        verbose=1,
    )

    # Self-play callback
    selfplay_callback = None
    if _global_opponent is not None:
        selfplay_callback = SelfPlayCallback(
            opponent=_global_opponent,
            update_freq=TrainingConfig.SELF_PLAY_UPDATE_FREQ,
            verbose=1,
        )
        print(f"[SelfPlay] Opponent model will update every {TrainingConfig.SELF_PLAY_UPDATE_FREQ} steps")

    callbacks = [checkpoint_callback]
    if TrainingConfig.LOG_EPISODE_STATS:
        callbacks.append(episode_stats_callback)
    if selfplay_callback is not None:
        callbacks.append(selfplay_callback)

    # Model creation
    print("\n[RL] Creating MaskablePPO model...")
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=TrainingConfig.LEARNING_RATE,
        n_steps=TrainingConfig.N_STEPS,
        batch_size=TrainingConfig.BATCH_SIZE,
        n_epochs=TrainingConfig.N_EPOCHS,
        gamma=TrainingConfig.GAMMA,
        gae_lambda=TrainingConfig.GAE_LAMBDA,
        clip_range=TrainingConfig.CLIP_RANGE,
        ent_coef=TrainingConfig.ENT_COEF,
        vf_coef=TrainingConfig.VF_COEF,
        max_grad_norm=TrainingConfig.MAX_GRAD_NORM,
        policy_kwargs=TrainingConfig.POLICY_KWARGS,
        tensorboard_log=TrainingConfig.TENSORBOARD_LOG,
        verbose=0,
    )

    # Training start
    print(f"\n[RL] Starting training for {TrainingConfig.TOTAL_TIMESTEPS} timesteps...")
    print(f"[RL] Episode stats will be printed every {TrainingConfig.LOG_FREQUENCY} episodes")
    print("[RL] Using rewards.py engine: Potential ranks, Turn cost, Status penalties")

    try:
        model.learn(
            total_timesteps=TrainingConfig.TOTAL_TIMESTEPS,
            callback=callbacks,
        )
    except KeyboardInterrupt:
        print("\n[Train2] Training interrupted manually.")
    finally:
        # Save final model
        final_path = os.path.join(TrainingConfig.SAVE_DIR, "final_model")
        model.save(final_path)
        print(f"\n[RL] Training finished. Final model saved to: {final_path}")


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Regulation I BSS agent (Train2)")
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=100,
        help="Print episode stats every N episodes (default: 100)",
    )
    parser.add_argument(
        "--no-episode-stats",
        action="store_true",
        help="Disable episode statistics logging",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose battle logging",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=f"Directory to store checkpoints (default: {TrainingConfig.SAVE_DIR})",
    )

    args = parser.parse_args()

    # Update config based on arguments
    if args.log_frequency:
        TrainingConfig.LOG_FREQUENCY = args.log_frequency
    if args.no_episode_stats:
        TrainingConfig.LOG_EPISODE_STATS = False
    if args.verbose:
        TrainingConfig.POKE_ENV_LOG_LEVEL = 20
        print("[Verbose] Enabled verbose battle logging")
    if args.save_dir:
        TrainingConfig.SAVE_DIR = args.save_dir

    train_masked_ppo()


if __name__ == "__main__":
    main()
