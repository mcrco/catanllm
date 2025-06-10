#!/usr/bin/env python3
"""
Example usage of the LLM Player for Catan.
"""

import os
import logging
from datetime import datetime
from catanatron import Game, Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.models.map import MINI_MAP_TEMPLATE, BASE_MAP_TEMPLATE, CatanMap
from players.llm_player import LLMPlayer
from util.game_convert import game_to_natural_language, format_playable_actions
from visualization.board_plot import plot_board
import json


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


def example_openrouter_game(
    max_turns=None, num_players=2, mini_map=False, vps=10, game_name=None, thinking=True
):
    """
    Example game with OpenRouter LLM player vs random players.
    """
    # Set up logging
    logger = setup_logging(game_name)

    # Create players
    print(thinking)
    players = [
        LLMPlayer(
            color=Color.RED,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="google/gemini-2.5-flash-preview-05-20",
            name=f"gemini2.5-flash-5-20:{'thinking' if thinking else ''}",
            # model="qwen/qwen3-30b-a3b",
            # name=f"qwen3-30b-a3b:{'thinking' if thinking else 'no-think'}",
            prompt="basic",
            thinking=thinking,
        ),
    ]

    colors = [Color.BLUE, Color.WHITE, Color.ORANGE]
    for i in range(num_players - 1):
        players.append(AlphaBetaPlayer(colors[i]))

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
    if max_turns:
        print(f"Max turns: {max_turns}")
    winning_color = game.play()
    print(f"Game finished! Winner: {winning_color}")

    return winning_color


def example_vllm_game(
    max_turns=None, num_players=2, mini_map=False, vps=10, game_name=None
):
    """
    Example game with vLLM player vs random players.
    """
    print("=== vLLM Local Player Example ===")

    # Set up logging
    logger = setup_logging(game_name)

    # Create players
    players = [
        LLMPlayer(
            color=Color.RED,
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen3-1.7B",  # Use default model from vLLM server
            name="Qwen-3-1.7B",
            max_tokens=8192,
        ),
    ]

    colors = [Color.BLUE, Color.WHITE, Color.ORANGE]
    for i in range(num_players - 1):
        players.append(RandomPlayer(colors[i]))

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
    if max_turns:
        print(f"Max turns: {max_turns}")
    print("Game state and LLM outputs will be logged to openrouter-output.log")
    winning_color = game.play()
    print(f"Game finished! Winner: {winning_color}")

    return winning_color


if __name__ == "__main__":
    import argparse
# 
    parser = argparse.ArgumentParser(description="LLM Catan Player Examples")
    parser.add_argument(
        "--mode",
        choices=["openrouter", "vllm", "benchmark"],
        default="openrouter",
        help="Which example to run",
    )
    parser.add_argument("--games", type=int, default=1, help="Number of games to run")
    parser.add_argument(
        "--vps", type=int, default=10, help="Number of victory points to win"
    )
    parser.add_argument(
        "--max-turns", type=int, default=None, help="Maximum number of turns per game"
    )
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--mini-map", action="store_true", help="Use mini map")
    parser.add_argument("--game-name", type=str, default=None, help="Name of the game")
    parser.add_argument("--no_think", action="store_true", help="Whether to use thinking")
    args = parser.parse_args()

    try:
        if args.mode == "openrouter":
            print(args.no_think)
            if args.games == 1:
                example_openrouter_game(
                    args.max_turns,
                    args.num_players,
                    args.mini_map,
                    args.vps,
                    args.game_name,
                    thinking=not args.no_think,
                )
        elif args.mode == "vllm":
            if args.games == 1:
                example_vllm_game(
                    args.max_turns,
                    args.num_players,
                    args.mini_map,
                    args.vps,
                    args.game_name,
                )

    except Exception as e:
        print(f"Error: {e}")