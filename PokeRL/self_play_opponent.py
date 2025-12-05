"""
Self-Play Opponent for training.
Uses the same PPO model as the training agent.
"""

import numpy as np
from poke_env.player import Player
from poke_env.battle import AbstractBattle
from typing import Optional


class SelfPlayOpponent(Player):
    """
    Opponent that uses the same PPO model as the training agent.

    - Initially acts randomly (model=None)
    - Model is updated periodically via callback
    - Uses the same observation embedding and action space as training agent
    """

    def __init__(self, env_for_embed, *args, **kwargs):
        """
        Args:
            env_for_embed: RegIPlayer instance for embed_battle() and action_to_order()
        """
        super().__init__(*args, **kwargs)
        self.model = None  # PPO model (set later by callback)
        self.env_for_embed = env_for_embed
        self.random_action_count = 0
        self.model_action_count = 0

    def choose_move(self, battle: AbstractBattle):
        """
        Choose move using PPO model if available, otherwise random.
        """
        # If no model yet, act randomly
        if self.model is None:
            self.random_action_count += 1
            return self.choose_random_move(battle)

        # Use PPO model to choose action
        try:
            # Get observation from battle
            obs = self.env_for_embed.embed_battle(battle)

            # Get action mask (use internal method that accepts battle parameter)
            action_mask = self.env_for_embed._get_action_mask_for_battle(battle)

            # Predict action using model with mask
            # MaskablePPO.predict() accepts action_masks parameter
            action, _states = self.model.predict(
                obs,
                action_masks=action_mask,
                deterministic=False  # Use stochastic policy for exploration
            )

            self.model_action_count += 1

            # Convert action to BattleOrder
            return self.env_for_embed.action_to_order(
                action,
                battle,
                fake=self.env_for_embed.fake,
                strict=self.env_for_embed.strict
            )

        except Exception as e:
            # Fallback to random if anything goes wrong
            self.logger.warning(f"SelfPlayOpponent failed to use model: {e}, using random")
            self.random_action_count += 1
            return self.choose_random_move(battle)

    def update_model(self, model):
        """
        Update the PPO model used by this opponent.
        Called periodically by SelfPlayCallback.
        """
        self.model = model
        if model is not None:
            self.logger.info(
                f"[SelfPlay] Model updated. Stats: {self.model_action_count} model actions, "
                f"{self.random_action_count} random actions"
            )
