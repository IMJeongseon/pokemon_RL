"""
Watch trained agent play battles with detailed logging.

Usage:
    python watch_battles.py --model checkpoints_regi/regi_masked_ppo_final --n-battles 5
"""

import argparse
import numpy as np
import logging
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from reg_i_player import RegIPlayer
from reg_i_team_builder import RegITeamBuilder
from random_opponent_builder import RandomOpponentTeamBuilder
from dual_team_env import DualTeamRegIPlayer
from poke_env.environment import SingleAgentWrapper
from poke_env.player import RandomPlayer


# Enable detailed battle logging
logging.basicConfig(level=logging.INFO)


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


ACTION_NAMES = [
    "Move 0", "Move 1", "Move 2", "Move 3",
    "Move 0 (TERA)", "Move 1 (TERA)", "Move 2 (TERA)", "Move 3 (TERA)",
    "Switch to bench 0", "Switch to bench 1",
]


def watch_battles(model_path: str, n_battles: int = 5):
    """
    Watch trained agent play battles with detailed logging.

    Args:
        model_path: Path to saved model
        n_battles: Number of battles to watch
    """
    print(f"\n{'='*80}")
    print(f"WATCHING AGENT PLAY - MODEL: {model_path}")
    print(f"{'='*80}\n")

    # Create environment (same structure as training)
    player_team = RegITeamBuilder()
    opponent_team = RandomOpponentTeamBuilder()

    dual_env = DualTeamRegIPlayer(
        team1=player_team,
        team2=opponent_team,
        battle_format="gen9bssregj",
        log_level=20,  # INFO level for detailed logs
        start_listening=True,
        strict=False,
    )

    # Create opponent (RandomPlayer)
    opponent = RandomPlayer(
        battle_format="gen9bssregj",
        log_level=20,
        start_listening=False,
    )

    single_agent_env = SingleAgentWrapper(env=dual_env, opponent=opponent)
    gym_env = GymnasiumWrapper(single_agent_env)
    env = ActionMasker(gym_env, mask_fn)

    # Load model
    print(f"Loading model from {model_path}...\n")
    model = MaskablePPO.load(model_path)

    # Watch battles
    for battle_num in range(1, n_battles + 1):
        print(f"\n{'='*80}")
        print(f"BATTLE #{battle_num}")
        print(f"{'='*80}\n")

        obs, info = env.reset()
        done = False
        turn = 0
        episode_reward = 0.0

        while not done:
            turn += 1

            # Get battle state - access through dual_env.agent1
            # dual_env.agent1 is the actual Player instance
            battle = list(dual_env.agent1.battles.values())[0] if dual_env.agent1.battles else None
            if not battle:
                break

            # Print turn header
            print(f"\n--- Turn {turn} ---")
            if battle.active_pokemon:
                print(f"My Active:  {battle.active_pokemon.species} "
                      f"(HP: {battle.active_pokemon.current_hp_fraction*100:.1f}%)")
            else:
                print("My Active:  None")

            if battle.opponent_active_pokemon:
                print(f"Opp Active: {battle.opponent_active_pokemon.species} "
                      f"(HP: {battle.opponent_active_pokemon.current_hp_fraction*100:.1f}%)")
            else:
                print("Opp Active: None")

            # Get action mask
            action_masks = env.action_masks()
            valid_actions = [i for i, mask in enumerate(action_masks) if mask == 1]

            print(f"Valid actions: {[ACTION_NAMES[i] for i in valid_actions]}")

            # Get agent's action
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            action = int(action)

            print(f"Agent chose: {ACTION_NAMES[action]}")

            # Get move name if applicable
            if 0 <= action < 8 and battle.active_pokemon:
                move_slot = action % 4
                moves = list(battle.active_pokemon.moves.values())
                if move_slot < len(moves):
                    move = moves[move_slot]
                    tera_str = " with TERA!" if action >= 4 else ""
                    print(f"  → Using move: {move.id}{tera_str}")

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward

            if reward != 0:
                print(f"Reward this step: {reward:+.2f}")

        # Battle result - get from final battle state
        battle = list(dual_env.agent1.battles.values())[0] if dual_env.agent1.battles else None

        print(f"\n{'='*80}")
        if battle:
            my_alive = len([p for p in battle.team.values() if not p.fainted])
            opp_alive = len([p for p in battle.opponent_team.values() if not p.fainted])

            if battle.won:
                print(f"🎉 BATTLE #{battle_num} WON!")
            else:
                print(f"❌ BATTLE #{battle_num} LOST")
            print(f"Final Score: {my_alive}-{opp_alive}")
        else:
            # Use reward to determine win/loss
            if episode_reward > 0:
                print(f"🎉 BATTLE #{battle_num} WON!")
            else:
                print(f"❌ BATTLE #{battle_num} LOST")
            print(f"Final Score: (unknown)")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Battle Length: {turn} turns")
        print(f"{'='*80}")

    env.close()
    print(f"\n✅ Finished watching {n_battles} battles!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained agent play battles")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/jeongseon39/MLLAB/pokemon/poke_RL/reg_i_rl/outputs/checkpoints_regi_v2/train2_final_model.zip",
        help="Path to model file (without .zip extension)"
    )
    parser.add_argument(
        "--n-battles",
        type=int,
        default=5,
        help="Number of battles to watch"
    )

    args = parser.parse_args()

    watch_battles(
        model_path=args.model,
        n_battles=args.n_battles
    )
