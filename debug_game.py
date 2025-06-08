#!/usr/bin/env python3
"""
Debug game simulation using DebugPlayer and a Catanatron bot.
Prints the natural language game state at each turn.
"""

from catanatron import Game, Color, RandomPlayer
from catanatron.models.map import MINI_MAP_TEMPLATE, CatanMap
from debug_player import create_debug_player
from game_convert import game_to_natural_language


def debug_game():
    """
    Simulate a game with a DebugPlayer and a RandomPlayer, printing the NL game state each turn.
    """
    print("=== Debug Game Example ===")

    players = [
        create_debug_player(Color.RED, name="Debugger"),
        RandomPlayer(Color.BLUE),
    ]

    game = Game(players, catan_map=CatanMap.from_template(MINI_MAP_TEMPLATE), vps_to_win=10)

    print("Starting debug game...")
    turn = 0
    while game.winning_color() is None:
        turn += 1
        print(f"\n--- Turn {turn} ---")
        # Print the natural language game state for the current player
        current_player = game.state.current_player()
        game_state_nl = game_to_natural_language(game, current_player.color)
        print(f"Current player: {current_player.color}")
        print("Game State (Natural Language):\n", game_state_nl)
        input("Press Enter to continue...")
        # Let the game proceed one step
        game.play_tick()

    print("\nGame finished! Winner:", game.winning_color())


if __name__ == "__main__":
    debug_game()

