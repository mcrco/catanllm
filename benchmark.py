#!/usr/bin/env python3
"""
Benchmark script for comparing LLM prompting strategies in Catan.
"""

import os
import json
import multiprocessing
from datetime import datetime
from functools import partial
from itertools import combinations
from typing import Callable
from dotenv import load_dotenv

from catanatron import Game, Color, RandomPlayer
from catanatron.models.player import Player
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.state_functions import get_actual_victory_points
from players.llm_player import LLMPlayer

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn


load_dotenv()
console = Console()

# --- Benchmark Configuration ---
PROMPTING_STRATEGIES = ["basic", "rules", "strategy", "search", "no_think"]
NUM_GAMES_PER_MATCH = 10
V_POINTS_TO_WIN = 10
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen3-235b-a22b"
# ---

def create_llm_player(color: Color, prompt: str) -> LLMPlayer:
    """Factory function for creating an LLMPlayer."""
    return LLMPlayer(
        color=color,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=MODEL,
        name=f"qwen3-235b-a22b-{prompt}",
        prompt=prompt
    )

def create_ab_player(color: Color) -> AlphaBetaPlayer:
    """Factory function for creating an AlphaBetaPlayer."""
    return AlphaBetaPlayer(color, name="AlphaBeta")

def create_random_player(color: Color) -> RandomPlayer:
    """Factory function for creating a RandomPlayer."""
    return RandomPlayer(color, name="Random")

def get_player_factory(player_type: str, prompt: str = None) -> Callable[[Color], Player]:
    """Get a factory function for a given player type."""
    if player_type == 'llm':
        if not prompt:
            raise ValueError("Prompt must be specified for LLM player.")
        return partial(create_llm_player, prompt=prompt)
    elif player_type == 'ab':
        return create_ab_player
    elif player_type == 'random':
        return create_random_player
    raise ValueError(f"Unknown player type: {player_type}")

def count_tokens(text: str) -> int:
    """Simple token counter by splitting on whitespace."""
    return len(text.split()) if text else 0

def _play_game(args):
    """
    Helper function to play a single game. Designed to be picklable for multiprocessing.
    """
    p1_factory, p2_factory, game_index = args
    
    p1 = p1_factory(Color.RED)
    p2 = p2_factory(Color.BLUE)
    
    # Swap starting player for fairness
    players = [p1, p2] if game_index % 2 == 0 else [p2, p1]

    game = Game(
        players,
        catan_map=CatanMap.from_template(BASE_MAP_TEMPLATE),
        vps_to_win=V_POINTS_TO_WIN
    )
    game.play()
    winner_color = game.winning_color()

    p1_score = get_actual_victory_points(game.state, p1.color)
    p2_score = get_actual_victory_points(game.state, p2.color)
    
    game_result = {
        'game_id': game_index,
        'winner': winner_color.name if winner_color else "DRAW",
        'scores': {p1.name: p1_score, p2.name: p2_score},
        'final_state': game.state.to_dict(),
    }
    
    p1_tokens = {'reasoning': 0, 'response': 0}
    p1_turns = 0
    if isinstance(p1, LLMPlayer):
        for log in p1.log:
            p1_tokens['reasoning'] += count_tokens(log.get('reasoning', ''))
            p1_tokens['response'] += count_tokens(log.get('response', ''))
        p1_turns = len(p1.log)

    p2_tokens = {'reasoning': 0, 'response': 0}
    p2_turns = 0
    if isinstance(p2, LLMPlayer):
        for log in p2.log:
            p2_tokens['reasoning'] += count_tokens(log.get('reasoning', ''))
            p2_tokens['response'] += count_tokens(log.get('response', ''))
        p2_turns = len(p2.log)

    return game_result, p1_tokens, p1_turns, p2_tokens, p2_turns, winner_color

def run_match(p1_factory: Callable, p2_factory: Callable, num_games: int):
    """Runs a match between two players for a number of games in parallel."""
    p1_wins = 0
    p2_wins = 0
    
    p1_total_tokens = {'reasoning': 0, 'response': 0}
    p2_total_tokens = {'reasoning': 0, 'response': 0}
    p1_llm_turns = 0
    p2_llm_turns = 0

    match_results = []
    
    p1_name = p1_factory(Color.RED).name
    p2_name = p2_factory(Color.BLUE).name

    tasks = [(p1_factory, p2_factory, i) for i in range(num_games)]

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "({task.completed}/{task.total})",
        console=console
    ) as progress:
        task_description = f"Playing {p1_name} vs {p2_name}"
        task = progress.add_task(task_description, total=num_games)
        
        with multiprocessing.Pool() as pool:
            results_iterator = pool.imap_unordered(_play_game, tasks)
            
            for game_result, p1_tok, p1_t, p2_tok, p2_t, winner_col in results_iterator:
                match_results.append(game_result)
                
                if winner_col == Color.RED:
                    p1_wins += 1
                elif winner_col == Color.BLUE:
                    p2_wins += 1
                
                p1_total_tokens['reasoning'] += p1_tok['reasoning']
                p1_total_tokens['response'] += p1_tok['response']
                p1_llm_turns += p1_t

                p2_total_tokens['reasoning'] += p2_tok['reasoning']
                p2_total_tokens['response'] += p2_tok['response']
                p2_llm_turns += p2_t

                progress.update(task, advance=1)

    match_results.sort(key=lambda x: x['game_id'])
    
    p1_avg_tokens = {}
    if p1_llm_turns > 0:
        p1_avg_tokens['reasoning'] = p1_total_tokens['reasoning'] / p1_llm_turns
        p1_avg_tokens['response'] = p1_total_tokens['response'] / p1_llm_turns

    p2_avg_tokens = {}
    if p2_llm_turns > 0:
        p2_avg_tokens['reasoning'] = p2_total_tokens['reasoning'] / p2_llm_turns
        p2_avg_tokens['response'] = p2_total_tokens['response'] / p2_llm_turns
        
    return {
        'p1_name': p1_name,
        'p2_name': p2_name,
        'p1_wins': p1_wins,
        'p2_wins': p2_wins,
        'ties': num_games - p1_wins - p2_wins,
        'p1_avg_tokens': p1_avg_tokens,
        'p2_avg_tokens': p2_avg_tokens,
        'games': match_results
    }

def main():
    """Main function to run the benchmark."""
    if not OPENROUTER_API_KEY:
        console.print("[bold red]Error: OPENROUTER_API_KEY environment variable not set.[/bold red]")
        return

    console.print(f"[bold]Starting Catan LLM Benchmark[/bold]")
    console.print(f"Model: [cyan]{MODEL}[/cyan], Games per match: [cyan]{NUM_GAMES_PER_MATCH}[/cyan]")
    
    all_results = []

    # 1. LLM vs LLM
    console.print("\n[bold green]Running LLM vs LLM matches...[/bold green]")
    llm_factories = {prompt: get_player_factory('llm', prompt) for prompt in PROMPTING_STRATEGIES}
    llm_pairs = list(combinations(PROMPTING_STRATEGIES, 2))
    for p1_prompt, p2_prompt in llm_pairs:
        p1_factory = llm_factories[p1_prompt]
        p2_factory = llm_factories[p2_prompt]
        result = run_match(p1_factory, p2_factory, NUM_GAMES_PER_MATCH)
        all_results.append(result)

    # 2. LLM vs Bots
    console.print("\n[bold green]Running LLM vs Bot matches...[/bold green]")
    bot_factories = {
        'AlphaBeta': get_player_factory('ab'),
        'Random': get_player_factory('random')
    }
    for prompt in PROMPTING_STRATEGIES:
        llm_factory = llm_factories[prompt]
        for bot_name, bot_factory in bot_factories.items():
            result = run_match(llm_factory, bot_factory, NUM_GAMES_PER_MATCH)
            all_results.append(result)
    
    # --- Reporting ---
    console.print("\n\n[bold underline]Benchmark Results[/bold underline]")

    # Winrate Table
    table = Table(title="Pairwise Win Rates")
    table.add_column("Player 1", justify="right", style="cyan", no_wrap=True)
    table.add_column("Player 2", justify="left", style="magenta", no_wrap=True)
    table.add_column("P1 Wins", justify="center")
    table.add_column("P2 Wins", justify="center")
    table.add_column("Ties", justify="center")
    table.add_column("P1 Win %", justify="right", style="green")

    for res in all_results:
        win_percentage = (res['p1_wins'] / NUM_GAMES_PER_MATCH) * 100
        table.add_row(
            res['p1_name'],
            res['p2_name'],
            str(res['p1_wins']),
            str(res['p2_wins']),
            str(res['ties']),
            f"{win_percentage:.1f}%"
        )
    console.print(table)
    
    # Token Usage Table
    token_table = Table(title="Average Token Usage per Turn for LLM Players")
    token_table.add_column("Player", justify="left", style="cyan")
    token_table.add_column("Avg Reasoning Tokens", justify="right")
    token_table.add_column("Avg Response Tokens", justify="right")
    
    token_stats = {}
    for res in all_results:
        if res['p1_avg_tokens']:
            name = res['p1_name']
            if name not in token_stats: token_stats[name] = {'reasoning': [], 'response': []}
            token_stats[name]['reasoning'].append(res['p1_avg_tokens']['reasoning'])
            token_stats[name]['response'].append(res['p1_avg_tokens']['response'])
        if res['p2_avg_tokens']:
            name = res['p2_name']
            if name not in token_stats: token_stats[name] = {'reasoning': [], 'response': []}
            token_stats[name]['reasoning'].append(res['p2_avg_tokens']['reasoning'])
            token_stats[name]['response'].append(res['p2_avg_tokens']['response'])

    for name, stats in sorted(token_stats.items()):
        avg_reasoning = sum(stats['reasoning']) / len(stats['reasoning'])
        avg_response = sum(stats['response']) / len(stats['response'])
        token_table.add_row(name, f"{avg_reasoning:.2f}", f"{avg_response:.2f}")
    
    console.print(token_table)

    # --- Save full results to JSON ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"benchmark-results-{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    console.print(f"\nFull results saved to [bold cyan]{filename}[/bold cyan]")


if __name__ == "__main__":
    main() 