#!/usr/bin/env python3
"""
Example usage of the LLM Player for Catan.
"""

import os
import logging
from datetime import datetime
from catanatron import Game, Color, RandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.models.map import MINI_MAP_TEMPLATE, BASE_MAP_TEMPLATE, CatanMap
from players.llm_player import LLMPlayer
from util.game_convert import game_to_natural_language, format_playable_actions
from visualization.board_plot import plot_board
import json


LLM_CONFIGS = {
    "gemini": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemini-2.5-flash-preview-05-20",
        "name": "gemini2.5-flash-5-20",
    },
    "qwen3": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3-235b-a22b",
        "name": "qwen3-235b-a22b",
    },
}

def create_ab_player(color: Color) -> AlphaBetaPlayer:
    """Factory function for creating an AlphaBetaPlayer."""
    player = AlphaBetaPlayer(color)
    player.name = "AlphaBeta"
    return player


def create_vf_player(color: Color) -> ValueFunctionPlayer:
    """Factory function for creating an ValueFunctionPlayer."""
    player = ValueFunctionPlayer(color)
    player.name = "ValueFunction"
    return player


def create_random_player(color: Color) -> RandomPlayer:
    """Factory function for creating a RandomPlayer."""
    player = RandomPlayer(color)
    player.name = "Random"
    return player


PLAYER_FACTORIES = {
    "alphabeta": create_ab_player,
    "random": create_random_player,
    "value": create_vf_player,
}


def setup_logging(game_name: str = None):
    """Set up logging to file for game states and LLM outputs."""
    logger = logging.getLogger("catan_game")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = f"{game_name}.log" if game_name else "last.log"
    file_handler = logging.FileHandler(os.path.join(logs_dir, log_filename), mode="w")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def log_game_state(logger, game, current_player_color, turn_number):
    """Log the current game state."""
    separator = f"{'='*50}"
    turn = f"TURN {turn_number} - {current_player_color.name}'s turn"
    logger.info(f"\n{separator}")
    logger.info(turn)
    logger.info(separator)
    game_state_nl = game_to_natural_language(game, current_player_color)
    logger.info(f"\nGAME STATE:\n{game_state_nl}")
    return f"{separator}\n{turn}\n{separator}\n{game_state_nl}"


def log_llm_response(logger, player):
    """Log the LLM response."""
    if player.log:
        logger.info(f"\nREASONING:\n{player.log[-1]['reasoning']}")
        logger.info(f"\nRESPONSE:\n{player.log[-1]['response']}")


class LoggingGame(Game):
    """Custom Game class that logs LLM interactions."""

    def __init__(
        self,
        players,
        logger=None,
        max_turns=None,
        mini_map=False,
        vps_to_win=10,
        game_name=None,
    ):
        super().__init__(
            players,
            catan_map=CatanMap.from_template(
                MINI_MAP_TEMPLATE if mini_map else BASE_MAP_TEMPLATE
            ),
            vps_to_win=vps_to_win,
        )
        self.logger = logger
        self.turn_number = 0
        self.max_turns = max_turns
        self.game_name = game_name
        self.logs_dir = f"logs/{self.game_name}"
        os.makedirs(self.logs_dir, exist_ok=True)
        self.board_dir = f"{self.logs_dir}/boards"
        os.makedirs(self.board_dir, exist_ok=True)
        self.game_state_logs = []

    def play_tick(self):
        if self.logger:
            self.turn_number += 1
            current_player = self.state.current_player()
            # Log game state before the turn
            game_state_log = log_game_state(
                self.logger, self, current_player.color, self.turn_number
            )
            self.game_state_logs.append(game_state_log)
            # Check if it's an LLM player and log their response
            if hasattr(current_player, "llm_wrapper"):
                self.logger.info(
                    f"\nLLM PLAYER {current_player.name} is making a decision..."
                )
                # Get playable actions
                playable_actions = self.state.playable_actions
                if playable_actions:
                    actions_nl = format_playable_actions(playable_actions)
                    self.logger.info(f"\nAVAILABLE ACTIONS:\n{actions_nl}")

        prev_player = self.state.current_player()

        image_path = f"{self.board_dir}/{self.turn_number}.png"
        plot_board(self.state, image_path)
        if hasattr(prev_player, "llm_wrapper"):
            prev_player.current_board_image_path = image_path

        super().play_tick()

        if self.logger and hasattr(prev_player, "llm_wrapper"):
            log_llm_response(self.logger, prev_player)

    def play(self):
        """Override play to log game start and end."""
        if self.logger:
            self.logger.info(f"GAME STARTED at {datetime.now()}")
            self.logger.info(f"Players: {[p.color.name for p in self.state.players]}")
            if self.max_turns:
                self.logger.info(f"Max turns limit: {self.max_turns}")

        # Play the game with max turns limit
        while not self.winning_color():
            if self.max_turns and self.turn_number >= self.max_turns:
                if self.logger:
                    self.logger.info(
                        f"\nGAME ENDED: Maximum turns ({self.max_turns}) reached"
                    )
                break
            self.play_tick()

        result = self.winning_color()

        if self.logger:
            self.logger.info(f"\nGAME FINISHED at {datetime.now()}")
            if result:
                self.logger.info(f"Winner: {result}")
            else:
                self.logger.info("Game ended without a winner (max turns reached)")

        for player in self.state.players:
            if hasattr(player, "log"):
                json.dump(player.log, open(f"{self.logs_dir}/{player.name}.json", "w"))
        return result


def run_game(
    player_types: list[str], max_turns=None, mini_map=False, vps=10, game_name=None
):
    """
    Example game with configurable players.
    """
    # Set up logging
    logger = setup_logging(game_name)

    # Create players
    players = []
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    for i, player_spec in enumerate(player_types):
        if i >= len(colors):
            raise ValueError(f"Maximum number of players is {len(colors)}")

        if player_spec["type"] in PLAYER_FACTORIES:
            player_factory = PLAYER_FACTORIES[player_spec["type"]]
            players.append(player_factory(color=colors[i]))
        elif player_spec['type'] == "llm":
            llm_config = LLM_CONFIGS[player_spec["config"]]
            llm_config["prompt"] = player_spec["prompt"]
            llm_config["thinking"] = player_spec["thinking"]
            player = LLMPlayer(color=colors[i], **llm_config)
            players.append(player)
        else:
            raise ValueError(f"Unknown player type: {player_spec['type']}")

    # Create and play game with logging
    game = LoggingGame(
        players,
        logger,
        max_turns,
        mini_map=mini_map,
        vps_to_win=vps,
        game_name=game_name,
    )

    print("Starting game...")
    print(f"Players: {[p.name for p in players]}")
    if max_turns:
        print(f"Max turns: {max_turns}")
    winning_color = game.play()
    print(f"Game finished! Winner: {winning_color}")

    return winning_color


if __name__ == "__main__":
    config = {
        "player_types": [
            {
                "type": "llm",
                "config": "gemini",
                "thinking": False,
                "prompt": "basic",
            },
            {
                "type": "value",
            },
        ],
        "max_turns": None,
        "mini_map": True,
        "vps": 10,
        "game_name": "gemini-no_think-basic",
    }

    run_game(**config)