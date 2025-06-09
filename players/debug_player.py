"""
Debug Player for Catanatron: prints the natural language game state and picks a random action.
"""

import random
import logging
from typing import List

from catanatron import Game, Color, Player, Action

from game_convert import game_to_natural_language, format_playable_actions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebugPlayer(Player):
    """
    Debugging Catan player that prints the game state and picks a random action.
    """
    def __init__(self, color: Color, name: str = None):
        super().__init__(color)
        self.name = name or f"Debug-{color.name}"
        logger.info(f"Initialized {self.name}")

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        if not playable_actions:
            raise ValueError("No playable actions available")
        # Print the natural language game state
        game_state_nl = game_to_natural_language(game, self.color)
        actions_nl = format_playable_actions(playable_actions)
        print(f"\n===== {self.name} Debug Info =====")
        print("Game State (Natural Language):\n", game_state_nl)
        print("Playable Actions:")
        print(actions_nl)
        print("==================================\n")
        # Pick a random action
        chosen_action = random.choice(playable_actions)
        logger.info(f"{self.name}: Randomly chose action: {chosen_action}")
        return chosen_action

def create_debug_player(color: Color, name: str = None) -> DebugPlayer:
    """
    Create a DebugPlayer.
    Args:
        color: Player color
        name: Optional player name
    Returns:
        DebugPlayer: Configured debug player
    """
    return DebugPlayer(color, name) 