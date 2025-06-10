

################################################################################

from collections import defaultdict
import random

import choix
import matplotlib.pyplot as plt
import numpy as np
import functools
import seaborn as sns
from rating import *

sns.set_theme(style='whitegrid')
sns.set_context("paper", font_scale=1.5)

def make_figures(matches):

    fig, axs = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True, dpi=300)
    ax_nmatches = axs[0, 0]
    ax_score = axs[0, 1]
    ax_prob = axs[1, 0]
    ax_rating = axs[1, 1]
    fig.suptitle("Gemini 2.5 Flash Performance")

    ################################################################################

    n_matches = defaultdict(int)

    for match in matches:
        agents = [match["p1_name"], match["p2_name"]]
        n_matches[agents[0], agents[1]] += len(match["games"])
        n_matches[agents[1], agents[0]] += len(match["games"])

    matrix = np.zeros((n_players, n_players), dtype="int")
    for i, player1 in enumerate(players):
        for j, player2 in enumerate(players):
            if player1 == player2:
                continue

            matrix[i][j] = n_matches[player1, player2]

    sns.heatmap(matrix, ax=ax_nmatches, annot=True, xticklabels=short_names, yticklabels=short_names)
    ax_nmatches.tick_params(axis='x', rotation=30)
    ax_nmatches.tick_params(axis='y', rotation=0)
    ax_nmatches.invert_yaxis()
    #plt.xticks(rotation=30)

    ax_nmatches.set_title("Number of matches")

    ################################################################################


    wins = defaultdict(int)
    for match in matches:   
        agents = [match["p1_name"], match["p2_name"]]
        wins[agents[0], agents[1]] += sum(game["scores"][agents[0]] for game in match["games"])
        wins[agents[1], agents[0]] += sum(game["scores"][agents[1]] for game in match["games"])

    for (agent1, agent2), score in wins.items():
        wins[agent1, agent2] = score

    matrix = np.empty((len(players), len(players)))
    matrix.fill(np.nan)
    for i, player1 in enumerate(players):
        for j, player2 in enumerate(players):
            if player1 == player2:
                continue

            matrix[i, j] = wins[player1, player2]

    sns.heatmap(matrix, ax=ax_score, annot=True, xticklabels=short_names, yticklabels=short_names)

    ax_score.set_title("Total score")
    ax_score.set_ylabel("Total VPs this agent scored...")
    ax_score.set_xlabel("... against this agent")
    ax_score.tick_params(axis='x', rotation=30)
    ax_score.tick_params(axis='y', rotation=0)
    ax_score.invert_yaxis()

    ################################################################################

    bootstrapped_params = bootstrap_params(matches)
    ratings = bootstrapped_params.mean(0)

    sorted_indices = np.argsort(ratings)
    sorted_data = bootstrapped_params[:, sorted_indices]
    sorted_labels = [short_names[i] for i in sorted_indices]

    sns.boxplot(data=sorted_data, whis=(5, 95), fliersize=0, ax=ax_rating)
    ax_rating.set_ylabel("Rating")
    ax_rating.set_xticks(ticks=range(len(players)), labels=sorted_labels)
    ax_rating.tick_params(axis='x', rotation=30)

    ################################################################################

    matrix = np.zeros((n_players, n_players))
    for i in range(n_players):
        for j in range(n_players):
            matrix[i, j] = choix.probabilities([i, j], ratings)[0]

    sns.heatmap(matrix, ax=ax_prob, annot=True, xticklabels=short_names, yticklabels=short_names, fmt=".2f")
    ax_prob.set_title("Win probabilities")
    ax_prob.set_ylabel("Probability this agent...")
    ax_prob.set_xlabel("... beats this agent")
    ax_prob.tick_params(axis='x', rotation=30)
    ax_prob.tick_params(axis='y', rotation=0)
    ax_prob.invert_yaxis()


    ################################################################################

    plt.savefig(f"figures/gemini-2.5-flash.png")
    plt.close()

make_figures(get_matches())