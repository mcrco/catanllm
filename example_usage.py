#!/usr/bin/env python3
"""
Example usage of the LLM Player for Catan.
"""

import os
import logging
from datetime import datetime
from catanatron import Game, Color, RandomPlayer

from llm_player import create_openrouter_player, create_vllm_player
from game_convert import game_to_natural_language


def setup_logging():
    """Set up logging to file for game states and LLM outputs."""
    # Create a logger
    logger = logging.getLogger('catan_game')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler('openrouter-output.log', mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger


def log_game_state(logger, game, current_player_color, turn_number):
    """Log the current game state."""
    logger.info(f"\n{'='*50}")
    logger.info(f"TURN {turn_number} - {current_player_color.name}'s turn")
    logger.info(f"{'='*50}")
    
    game_state_nl = game_to_natural_language(game, current_player_color)
    logger.info(f"\nGAME STATE:\n{game_state_nl}")


class LoggingGame(Game):
    """Custom Game class that logs LLM interactions."""
    
    def __init__(self, players, logger=None, max_turns=None):
        super().__init__(players)
        self.logger = logger
        self.turn_number = 0
        self.max_turns = max_turns
    
    def play_tick(self):
        if self.logger:
            self.turn_number += 1
            current_player = self.state.current_player()
            # Log game state before the turn
            log_game_state(self.logger, self, current_player.color, self.turn_number)
            # Check if it's an LLM player and log their response
            if hasattr(current_player, 'llm_wrapper'):
                self.logger.info(f"\nLLM PLAYER {current_player.name} is making a decision...")
                # Get playable actions
                playable_actions = self.state.playable_actions
                if playable_actions:
                    from game_convert import format_playable_actions
                    actions_nl = format_playable_actions(playable_actions)
                    self.logger.info(f"\nAVAILABLE ACTIONS:\n{actions_nl}")
        # Save the player who is about to act
        prev_player = self.state.current_player()
        # Call the original play_tick method
        super().play_tick()
        # Log the action taken and LLM output for the player who just acted
        if self.logger and hasattr(prev_player, 'llm_wrapper'):
            if self.state.actions:
                last_action = self.state.actions[-1]
                self.logger.info(f"\nACTION TAKEN: {last_action}")
            llm_response = getattr(prev_player, 'last_llm_response', None)
            if llm_response is not None:
                self.logger.info(f"\nLLM OUTPUT:\n{llm_response}")
    
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
                    self.logger.info(f"\nGAME ENDED: Maximum turns ({self.max_turns}) reached")
                break
            self.play_tick()
        
        result = self.winning_color()
        
        if self.logger:
            self.logger.info(f"\nGAME FINISHED at {datetime.now()}")
            if result:
                self.logger.info(f"Winner: {result}")
            else:
                self.logger.info("Game ended without a winner (max turns reached)")
        
        return result


def example_openrouter_game(max_turns=None):
    """
    Example game with OpenRouter LLM player vs random players.
    """
    print("=== OpenRouter LLM Player Example ===")
    
    # Set up logging
    logger = setup_logging()
    
    # Create players
    players = [
        create_openrouter_player(
            Color.RED, 
            api_key=None,  # Will use OPENROUTER_API_KEY environment variable
            model="qwen/qwen3-8b",
            name="Qwen3-8B"
        ),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    
    # Create and play game with logging
    game = LoggingGame(players, logger, max_turns)
    
    print("Starting game...")
    if max_turns:
        print(f"Max turns: {max_turns}")
    print("Game state and LLM outputs will be logged to openrouter-output.log")
    winning_color = game.play()
    print(f"Game finished! Winner: {winning_color}")
    
    return winning_color


def example_vllm_game(max_turns=None):
    """
    Example game with vLLM player vs random players.
    """
    print("=== vLLM Local Player Example ===")
    
    # Set up logging
    logger = setup_logging()
    
    # Create players
    players = [
        create_vllm_player(
            Color.RED,
            base_url="http://localhost:8000",
            model_name=None,  # Use default model from vLLM server
            name="Local-LLM"
        ),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    
    # Create and play game with logging
    game = LoggingGame(players, logger, max_turns)
    
    print("Starting game...")
    if max_turns:
        print(f"Max turns: {max_turns}")
    print("Game state and LLM outputs will be logged to openrouter-output.log")
    winning_color = game.play()
    print(f"Game finished! Winner: {winning_color}")
    
    return winning_color


def example_mixed_game(max_turns=None):
    """
    Example game with both OpenRouter and vLLM players.
    """
    print("=== Mixed LLM Players Example ===")
    
    # Set up logging
    logger = setup_logging()
    
    # Create players
    players = [
        create_openrouter_player(
            Color.RED,
            name="Claude-LLM"
        ),
        create_vllm_player(
            Color.BLUE,
            name="Local-LLM"
        ),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    
    # Create and play game with logging
    game = LoggingGame(players, logger, max_turns)
    
    print("Starting game...")
    if max_turns:
        print(f"Max turns: {max_turns}")
    print("Game state and LLM outputs will be logged to openrouter-output.log")
    winning_color = game.play()
    print(f"Game finished! Winner: {winning_color}")
    
    return winning_color


def run_multiple_games(num_games=10, use_openrouter=True, max_turns=None):
    """
    Run multiple games and collect statistics.
    """
    print(f"=== Running {num_games} games ===")
    if max_turns:
        print(f"Max turns per game: {max_turns}")
    
    results = {Color.RED: 0, Color.BLUE: 0, Color.WHITE: 0, Color.ORANGE: 0}
    
    for i in range(num_games):
        print(f"Game {i+1}/{num_games}")
        
        if use_openrouter:
            # Check if API key is available
            if not os.getenv("OPENROUTER_API_KEY"):
                print("Warning: OPENROUTER_API_KEY not set, using vLLM instead")
                use_openrouter = False
        
        try:
            if use_openrouter:
                winner = example_openrouter_game(max_turns)
            else:
                winner = example_vllm_game(max_turns)
            
            if winner:
                results[winner] += 1
            
        except Exception as e:
            print(f"Error in game {i+1}: {e}")
            continue
    
    print("\n=== Results ===")
    total_games = sum(results.values())
    for color, wins in results.items():
        percentage = (wins / total_games * 100) if total_games > 0 else 0
        print(f"{color.name}: {wins}/{total_games} ({percentage:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Catan Player Examples")
    parser.add_argument("--mode", choices=["openrouter", "vllm", "mixed", "benchmark"], 
                       default="openrouter", help="Which example to run")
    parser.add_argument("--games", type=int, default=1, help="Number of games to run")
    parser.add_argument("--vps", type=int, default=3, help="Number of victory points to win")
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum number of turns per game")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "openrouter":
            if args.games == 1:
                example_openrouter_game(args.max_turns)
            else:
                run_multiple_games(args.games, use_openrouter=True, max_turns=args.max_turns)
        elif args.mode == "vllm":
            if args.games == 1:
                example_vllm_game(args.max_turns)
            else:
                run_multiple_games(args.games, use_openrouter=False, max_turns=args.max_turns)
        elif args.mode == "mixed":
            example_mixed_game(args.max_turns)
        elif args.mode == "benchmark":
            run_multiple_games(args.games, use_openrouter=True, max_turns=args.max_turns)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Set OPENROUTER_API_KEY environment variable (for OpenRouter)")
        print("2. Started vLLM server on localhost:8000 (for vLLM)")
        print("3. Installed dependencies: pip install -e .") 