#!/usr/bin/env python3
"""
Benchmark script for comparing LLM prompting strategies in Catan.
"""

import os
import json
import multiprocessing
from datetime import datetime
from functools import partial
from typing import Callable
from collections import defaultdict
from dotenv import load_dotenv
from itertools import combinations

from catanatron import Game, Color, RandomPlayer
from catanatron.models.player import Player
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.models.map import BASE_MAP_TEMPLATE, MINI_MAP_TEMPLATE, CatanMap
from catanatron.state_functions import get_actual_victory_points
from players.llm_player import LLMPlayer
from visualization.board_plot import plot_board

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn

from util.game_convert import game_to_natural_language


load_dotenv()
console = Console()

# --- Benchmark Configuration ---
PROMPTING_STRATEGIES = ["basic", "strategy", "search", "rules"]
NUM_GAMES_PER_MATCH = 5
V_POINTS_TO_WIN = 10
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.5-flash-preview-05-20"
# MODEL = "qwen/qwen3-30b-a3b"
MAX_TOKENS = 8192
USE_MINI_MAP = True
# ---

def sanitize_filename(name: str) -> str:
    """Remove invalid characters from a string to make it a valid filename."""
    return "".join(c for c in name if c.isalnum() or c in ('-', '_')).rstrip()

def create_llm_player(color: Color, prompt: str, thinking: bool, use_mini_map: bool) -> LLMPlayer:
    """Factory function for creating an LLMPlayer."""
    thinking_str = "think" if thinking else "no-think"
    map_str = "-mini" if use_mini_map else ""
    return LLMPlayer(
        color=color,
        name=f"{MODEL}-{prompt}-{thinking_str}{map_str}",
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=MODEL,
        prompt=prompt,
        thinking=thinking,
        max_tokens=MAX_TOKENS,
        use_mini_map=use_mini_map
    )

def create_ab_player(color: Color) -> AlphaBetaPlayer:
    """Factory function for creating an AlphaBetaPlayer."""
    player = AlphaBetaPlayer(color)
    player.name = "AlphaBeta"
    return player

def create_random_player(color: Color) -> RandomPlayer:
    """Factory function for creating a RandomPlayer."""
    player = RandomPlayer(color)
    player.name = "Random"
    return player

def get_player_factory(player_type: str, prompt: str = None, thinking: bool = True, use_mini_map: bool = False) -> Callable[[Color], Player]:
    """Get a factory function for a given player type."""
    if player_type == 'llm':
        if not prompt:
            raise ValueError("Prompt must be specified for LLM player.")
        return partial(create_llm_player, prompt=prompt, thinking=thinking, use_mini_map=use_mini_map)
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
    p1_factory, p2_factory, game_index, use_mini_map, benchmark_timestamp = args
    
    p1 = p1_factory(Color.RED)
    p2 = p2_factory(Color.BLUE)
    
    # Swap starting player for fairness
    players = [p1, p2] if game_index % 2 == 0 else [p2, p1]

    map_template = MINI_MAP_TEMPLATE if use_mini_map else BASE_MAP_TEMPLATE
    game = Game(
        players,
        catan_map=CatanMap.from_template(map_template),
        vps_to_win=V_POINTS_TO_WIN
    )
    game.play()
    winner_color = game.winning_color()

    # Save final board state as an image
    boards_dir = os.path.join("benchmarks", benchmark_timestamp, "boards")
    os.makedirs(boards_dir, exist_ok=True)
    sanitized_p1_name = sanitize_filename(p1.name)
    sanitized_p2_name = sanitize_filename(p2.name)
    board_filename = f"game_{game_index}_{sanitized_p1_name}_vs_{sanitized_p2_name}.png"
    board_filepath = os.path.join(boards_dir, board_filename)
    try:
        plot_board(game.state, board_filepath)
    except Exception as e:
        print(f"Warning: Could not plot board for game {game_index}. Error: {e}")

    p1_score = get_actual_victory_points(game.state, p1.color)
    p2_score = get_actual_victory_points(game.state, p2.color)
    
    game_result = {
        'game_id': game_index,
        'winner': winner_color.name if winner_color else "DRAW",
        'scores': {p1.name: p1_score, p2.name: p2_score},
        'final_state': game_to_natural_language(game, winner_color),
        'board_filepath': board_filepath
    }
    
    p1_tokens = {'reasoning': 0, 'response': 0}
    p1_turns = 0
    if isinstance(p1, LLMPlayer):
        for log in p1.log:
            p1_tokens['reasoning'] += count_tokens(log.get('reasoning', ''))
            p1_tokens['response'] += count_tokens(log.get('response', ''))
        p1_turns = len(p1.log)

    p1_fallback_info = (0, 0)
    if isinstance(p1, LLMPlayer):
        p1_fallback_info = (p1.fallback_count, p1.decision_count)

    p2_tokens = {'reasoning': 0, 'response': 0}
    p2_turns = 0
    if isinstance(p2, LLMPlayer):
        for log in p2.log:
            p2_tokens['reasoning'] += count_tokens(log.get('reasoning', ''))
            p2_tokens['response'] += count_tokens(log.get('response', ''))
        p2_turns = len(p2.log)

    p2_fallback_info = (0, 0)
    if isinstance(p2, LLMPlayer):
        p2_fallback_info = (p2.fallback_count, p2.decision_count)

    return game_result, p1_tokens, p1_turns, p2_tokens, p2_turns, winner_color, p1.name, p2.name, p1_fallback_info, p2_fallback_info

def main():
    """Main function to run the benchmark."""
    if not OPENROUTER_API_KEY:
        console.print("[bold red]Error: OPENROUTER_API_KEY environment variable not set.[/bold red]")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    console.print(f"[bold]Starting Catan LLM Benchmark[/bold]")
    console.print(f"Model: [cyan]{MODEL}[/cyan], Games per match: [cyan]{NUM_GAMES_PER_MATCH}[/cyan], Mini Map: [cyan]{USE_MINI_MAP}[/cyan]")
    
    # --- Define player variants ---
    llm_variants = [("basic", True)]
    for prompt in PROMPTING_STRATEGIES:
        llm_variants.append((prompt, False))

    llm_factories = {
        variant: get_player_factory("llm", prompt=variant[0], thinking=variant[1], use_mini_map=USE_MINI_MAP)
        for variant in llm_variants
    }
    bot_factories = {
        "AlphaBeta": get_player_factory("ab"),
        "Random": get_player_factory("random"),
    }

    # --- Create all match and game tasks ---
    matchups = []
    # 1. LLM vs LLM
    console.print("\n[bold green]Preparing LLM vs LLM matches...[/bold green]")
    llm_pairs = list(combinations(llm_variants, 2))
    for p1_variant, p2_variant in llm_pairs:
        p1_factory = llm_factories[p1_variant]
        p2_factory = llm_factories[p2_variant]
        matchups.append((p1_factory, p2_factory))

    # 2. LLM vs Bots
    console.print("[bold green]Preparing LLM vs Bot matches...[/bold green]")
    for variant in llm_variants:
        llm_factory = llm_factories[variant]
        for bot_name, bot_factory in bot_factories.items():
            matchups.append((llm_factory, bot_factory))

    game_tasks = []
    for p1_factory, p2_factory in matchups:
        for i in range(NUM_GAMES_PER_MATCH):
            game_tasks.append((p1_factory, p2_factory, i, USE_MINI_MAP, timestamp))

    # --- Run all games in parallel ---
    console.print("\n[bold green]Running all games in parallel...[/bold green]")
    all_game_results_raw = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "({task.completed}/{task.total})",
        console=console
    ) as progress:
        task_description = "Playing all games"
        task = progress.add_task(task_description, total=len(game_tasks))
        
        with multiprocessing.Pool() as pool:
            results_iterator = pool.imap_unordered(_play_game, game_tasks)
            
            for result in results_iterator:
                all_game_results_raw.append(result)
                progress.update(task, advance=1)

    # --- Process results ---
    console.print("\n[bold]Aggregating results...[/bold]")
    match_results_grouped = defaultdict(list)
    fallback_stats = defaultdict(lambda: {'fallbacks': 0, 'decisions': 0})
    for res in all_game_results_raw:
        game_res, p1_tok, p1_t, p2_tok, p2_t, winner_col, p1_name, p2_name, p1_fallback, p2_fallback = res
        match_results_grouped[(p1_name, p2_name)].append(
            (game_res, p1_tok, p1_t, p2_tok, p2_t, winner_col)
        )
        if p1_fallback[1] > 0:
            fallback_stats[p1_name]['fallbacks'] += p1_fallback[0]
            fallback_stats[p1_name]['decisions'] += p1_fallback[1]
        if p2_fallback[1] > 0:
            fallback_stats[p2_name]['fallbacks'] += p2_fallback[0]
            fallback_stats[p2_name]['decisions'] += p2_fallback[1]
    
    all_results = []
    for (p1_name, p2_name), game_results_list in match_results_grouped.items():
        p1_wins = 0
        p2_wins = 0
        p1_total_tokens = {'reasoning': 0, 'response': 0}
        p2_total_tokens = {'reasoning': 0, 'response': 0}
        p1_llm_turns = 0
        p2_llm_turns = 0
        p1_agg_score = 0
        p2_agg_score = 0
        match_games = []

        for game_result, p1_tok, p1_t, p2_tok, p2_t, winner_col in game_results_list:
            match_games.append(game_result)
            if winner_col == Color.RED:
                p1_wins += 1
            elif winner_col == Color.BLUE:
                p2_wins += 1
            
            p1_agg_score += game_result['scores'][p1_name]
            p2_agg_score += game_result['scores'][p2_name]
            
            p1_total_tokens['reasoning'] += p1_tok['reasoning']
            p1_total_tokens['response'] += p1_tok['response']
            p1_llm_turns += p1_t

            p2_total_tokens['reasoning'] += p2_tok['reasoning']
            p2_total_tokens['response'] += p2_tok['response']
            p2_llm_turns += p2_t
        
        match_games.sort(key=lambda x: x['game_id'])

        p1_avg_tokens = {}
        if p1_llm_turns > 0:
            p1_avg_tokens['reasoning'] = p1_total_tokens['reasoning'] / p1_llm_turns
            p1_avg_tokens['response'] = p1_total_tokens['response'] / p1_llm_turns

        p2_avg_tokens = {}
        if p2_llm_turns > 0:
            p2_avg_tokens['reasoning'] = p2_total_tokens['reasoning'] / p2_llm_turns
            p2_avg_tokens['response'] = p2_total_tokens['response'] / p2_llm_turns

        win_percentage = (p1_wins / NUM_GAMES_PER_MATCH) * 100
        all_results.append({
            'p1_name': p1_name,
            'p2_name': p2_name,
            'p1_wins': p1_wins,
            'p2_wins': p2_wins,
            'ties': len(game_results_list) - p1_wins - p2_wins,
            'p1_agg_score': p1_agg_score,
            'p2_agg_score': p2_agg_score,
            'p1_avg_tokens': p1_avg_tokens,
            'p2_avg_tokens': p2_avg_tokens,
            'games': match_games
        })
    
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
    
    # Score Table
    score_table = Table(title="Aggregate Victory Points")
    score_table.add_column("Player 1", justify="right", style="cyan", no_wrap=True)
    score_table.add_column("Player 2", justify="left", style="magenta", no_wrap=True)
    score_table.add_column("P1 Total VPs", justify="center")
    score_table.add_column("P2 Total VPs", justify="center")

    for res in all_results:
        score_table.add_row(
            res['p1_name'],
            res['p2_name'],
            str(res['p1_agg_score']),
            str(res['p2_agg_score']),
        )
    console.print(score_table)
    
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

    # Fallback Rate Table
    fallback_table = Table(title="LLM Player Fallback Decision Rates")
    fallback_table.add_column("Player", justify="left", style="cyan")
    fallback_table.add_column("Fallback Decisions", justify="right")
    fallback_table.add_column("Total Decisions", justify="right")
    fallback_table.add_column("Failure Rate", justify="right", style="red")
    
    for name, stats in sorted(fallback_stats.items()):
        fallbacks = stats['fallbacks']
        decisions = stats['decisions']
        failure_rate = (fallbacks / decisions) * 100 if decisions > 0 else 0
        fallback_table.add_row(name, str(fallbacks), str(decisions), f"{failure_rate:.2f}%")
    
    console.print(fallback_table)

    # --- Save full results to JSON ---
    benchmark_run_dir = os.path.join("benchmarks", timestamp)
    os.makedirs(benchmark_run_dir, exist_ok=True)
    filename = "benchmark-results.json"
    filepath = os.path.join(benchmark_run_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    console.print(f"\nFull results saved to [bold cyan]{filepath}[/bold cyan]")


if __name__ == "__main__":
    main() 