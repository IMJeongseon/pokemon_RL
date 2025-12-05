"""
Evaluate a trained MaskablePPO model against another checkpointed model.

Usage:
    python evaluate.py --model outputs/checkpoints_regi_v2/train2_final_model.zip \
                       --opponent-model outputs/checkpoints_regi_v2/train2_model_50000_steps.zip \
                       --n-battles 200
"""

import argparse
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from reg_i_player import RegIPlayer
from reg_i_team_builder import RegITeamBuilder
from dual_team_env import DualTeamRegIPlayer
from poke_env.environment import SingleAgentWrapper
from poke_env.player import Player
from poke_env.battle import AbstractBattle


class ModelOpponent(Player):
    """Opponent controlled by a loaded MaskablePPO checkpoint."""

    def __init__(self, model_path: str, team, **kwargs):
        super().__init__(team=team, **kwargs)
        self.model = MaskablePPO.load(model_path)
        # A RegIPlayer helper to embed battles and map actions -> orders
        self.reg_player = RegIPlayer(team=team, battle_format=kwargs.get("battle_format", "gen9bssregj"))

    def choose_move(self, battle: AbstractBattle):
        # Embed current battle state
        obs = self.reg_player.embed_battle(battle)
        # Action mask for this specific battle
        action_mask = self.reg_player._get_action_mask_for_battle(battle)
        # Predict deterministic action
        action, _states = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        # Convert to BattleOrder (lenient to avoid async issues)
        return self.reg_player.action_to_order(int(action), battle, strict=False)


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


def mask_fn(env):
    return env.get_action_mask()


def evaluate_model_vs_checkpoint(
    model_path: str,
    opponent_model_path: str,
    n_battles: int = 100,
    battle_format: str = "gen9bssregj",
    verbose: bool = True,
):
    """
    Evaluate model_path agent against opponent_model_path agent.

    Both sides use the same team builder (RegITeamBuilder).
    """
    print(f"\n{'='*60}")
    print("EVALUATION: MODEL vs MODEL")
    print(f"{'='*60}")
    print(f"Agent model:    {model_path}")
    print(f"Opponent model: {opponent_model_path}")
    print(f"Battles:        {n_battles}")
    print(f"{'='*60}\n")

    # Team for both players
    team = RegITeamBuilder()

    # Environment player (agent under evaluation)
    env_player = DualTeamRegIPlayer(
        team1=team,
        team2=team,  # opponent's team is handled by opponent player
        battle_format=battle_format,
        log_level=30,
        start_listening=True,
        strict=False,
    )

    # Opponent controlled by checkpoint
    opponent = ModelOpponent(
        model_path=opponent_model_path,
        team=team,
        battle_format=battle_format,
        log_level=30,
        start_listening=False,
    )

    # Wrap environment
    single_agent_env = SingleAgentWrapper(env_player, opponent)
    gym_env = GymnasiumWrapper(single_agent_env)
    test_env = ActionMasker(gym_env, mask_fn)

    # Load agent model
    model = MaskablePPO.load(model_path)

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
            action_masks = test_env.action_masks()
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_reward > 0:
            wins += 1
            alive_diff = max(0, int(episode_reward - 10))
        else:
            losses += 1
            alive_diff = min(0, int(episode_reward + 10))
        alive_diffs.append(alive_diff)

        if (i + 1) % 10 == 0:
            print(
                f"  Progress: {i+1}/{n_battles} - "
                f"Win rate: {wins/(i+1)*100:.1f}% - "
                f"Avg reward: {np.mean(episode_rewards):.2f}"
            )

    win_rate = wins / n_battles
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_alive_diff = np.mean(alive_diffs)

    if verbose:
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total Battles:    {n_battles}")
        print(f"Wins:             {wins}")
        print(f"Losses:           {losses}")
        print(f"Win Rate:         {win_rate*100:.2f}%")
        print(f"Avg Reward:       {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Avg Episode Len:  {avg_length:.1f} steps")
        print(f"Avg Alive Diff:   {avg_alive_diff:+.2f} Pokémon")
        print(f"Reward Range:     [{min(episode_rewards):.2f}, {max(episode_rewards):.2f}]")
        print(f"{'='*60}\n")

    test_env.close()

    return {
        "n_battles": n_battles,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
        "avg_alive_diff": avg_alive_diff,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO model vs another PPO checkpoint")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to agent model (.zip)",
    )
    parser.add_argument(
        "--opponent-model",
        type=str,
        required=True,
        help="Path to opponent model (.zip)",
    )
    parser.add_argument(
        "--n-battles",
        type=int,
        default=200,
        help="Number of battles to evaluate (single side)",
    )
    parser.add_argument(
        "--battle-format",
        type=str,
        default="gen9bssregj",
        help="Battle format string",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()

    evaluate_model_vs_checkpoint(
        model_path=args.model,
        opponent_model_path=args.opponent_model,
        n_battles=args.n_battles,
        battle_format=args.battle_format,
        verbose=not args.quiet,
    )
