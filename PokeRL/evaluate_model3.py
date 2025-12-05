"""
Evaluate trained Masked PPO model with RANDOM TEAMS (both player and opponent).

This tests if the learned strategy can generalize to completely random team matchups.

Usage:
    python evaluate_model3.py --model outputs/checkpoints_regi_v2/final_model --n-battles 100
"""

import argparse
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from reg_i_player import RegIPlayer
from random_opponent_builder import RandomOpponentTeamBuilder
from dual_team_env import DualTeamRegIPlayer
from poke_env.environment import SingleAgentWrapper
from poke_env.player import RandomPlayer


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
        """Get action mask from underlying RegIPlayer environment."""
        return self.env.env.get_action_mask()


def mask_fn(env):
    """Action mask function for ActionMasker."""
    return env.get_action_mask()


def evaluate_model(model_path: str, n_battles: int = 100, verbose: bool = True):
    """
    Evaluate a trained model with RANDOM teams (both player and opponent).

    Args:
        model_path: Path to saved model (.zip file)
        n_battles: Number of battles to play
        verbose: Print detailed statistics

    Returns:
        dict: Evaluation statistics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_path}")
    print(f"Player Team: RANDOM (from pool)")
    print(f"Opponent Team: RANDOM (from pool)")
    print(f"{'='*60}\n")

    # Create evaluation environment (same structure as training)
    print("[Eval] Creating environment...")

    # Player team (RANDOM)
    player_team = RandomOpponentTeamBuilder()

    # Opponent team (RANDOM)
    opponent_team = RandomOpponentTeamBuilder()

    # Create DualTeam environment
    dual_env = DualTeamRegIPlayer(
        team1=player_team,
        team2=opponent_team,
        battle_format="gen9bssregj",
        log_level=30,  # WARNING level
        start_listening=True,
        strict=False,
    )

    # Create opponent (RandomPlayer)
    opponent = RandomPlayer(
        battle_format="gen9bssregj",
        log_level=30,
        start_listening=False,
    )

    # Wrap with SingleAgentWrapper
    single_agent_env = SingleAgentWrapper(dual_env, opponent)

    # Wrap with Gymnasium wrapper
    gym_env = GymnasiumWrapper(single_agent_env)

    # Wrap with ActionMasker
    test_env = ActionMasker(gym_env, mask_fn)

    # Load model
    print(f"[Eval] Loading model from {model_path}...")
    model = MaskablePPO.load(model_path)

    # Evaluation loop
    print(f"[Eval] Starting {n_battles} battles...\n")

    wins = 0
    losses = 0
    episode_rewards = []
    episode_lengths = []
    alive_diffs = []

    for i in range(n_battles):
        obs, info = test_env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            # Get action mask
            action_masks = test_env.action_masks()

            # Predict action with mask
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

            # Take step
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        # Record results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check win/loss from reward
        # Terminal reward: +10 for win, -10 for loss
        # So positive final reward means win
        if episode_reward > 0:
            wins += 1
            # Estimate alive diff from reward
            # reward = 10 + alive_diff, so alive_diff ≈ reward - 10
            alive_diff = max(0, int(episode_reward - 10))
        else:
            losses += 1
            # reward = -10 + alive_diff, so alive_diff ≈ reward + 10
            alive_diff = min(0, int(episode_reward + 10))

        alive_diffs.append(alive_diff)

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_battles} battles - "
                  f"Win rate: {wins/(i+1)*100:.1f}% - "
                  f"Avg reward: {np.mean(episode_rewards):.2f}")

    # Calculate statistics
    win_rate = wins / n_battles
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_alive_diff = np.mean(alive_diffs)

    # Print results
    if verbose:
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Team Setup:       Both RANDOM")
        print(f"Total Battles:    {n_battles}")
        print(f"Wins:             {wins}")
        print(f"Losses:           {losses}")
        print(f"Win Rate:         {win_rate*100:.2f}%")
        print(f"Avg Reward:       {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Avg Episode Len:  {avg_length:.1f} steps")
        print(f"Avg Alive Diff:   {avg_alive_diff:+.2f} Pokémon")
        print(f"Reward Range:     [{min(episode_rewards):.2f}, {max(episode_rewards):.2f}]")
        print(f"{'='*60}\n")

        # Interpretation
        if win_rate > 0.55:
            print(f"✓ Model shows strong generalization ({win_rate*100:.1f}% > 55%)")
        elif win_rate > 0.50:
            print(f"~ Model shows moderate generalization ({win_rate*100:.1f}% > 50%)")
        else:
            print(f"⚠ Model struggles with random teams ({win_rate*100:.1f}% ≤ 50%)")
        print(f"{'='*60}\n")

    # Clean up
    test_env.close()

    return {
        'n_battles': n_battles,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'avg_alive_diff': avg_alive_diff,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model with random teams")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/jeongseon39/MLLAB/pokemon/poke_RL/reg_i_rl/outputs/checkpoints_regi_v2/train2_final_model.zip",
        help="Path to model file (without .zip extension)"
    )
    parser.add_argument(
        "--n-battles",
        type=int,
        default=100,
        help="Number of battles to evaluate"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        n_battles=args.n_battles,
        verbose=not args.quiet
    )
