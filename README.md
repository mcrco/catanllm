# Benchmark for LLMs Using Catan(atron)

SP 25 CS 159 Project.

This project provides a framework for benchmarking Large Language Model (LLM) players in the game of Catan, using the `catanatron` engine. It allows for comparing different LLM prompting strategies against each other and against traditional AI players.

## How to Run

### 1. Installation

The project uses `uv` for dependency management. To install the required packages, run:

```bash
uv sync
```

This project also depends on `catanatron`, which is installed as a git dependency from its [repository](https://github.com/bcollazo/catanatron.git).

### 2. Environment Setup

The LLM wrapper is written for the OpenAI SDK since I decided to use Openrouter. The API key should be stored like this:

1.  Create a `.env` file in the root of the project.
2.  Add your API key to the `.env` file:

    ```
    OPENROUTER_API_KEY="your_api_key_here"
    ```

### 3. Running the Benchmark

The main script for running the benchmark is `benchmark.py`. You can configure the benchmark by modifying the variables at the top of the file:

-   `PROMPTING_STRATEGIES`: A list of prompting strategies to test for the LLM player.
-   `NUM_GAMES_PER_MATCH`: The number of games to play for each matchup.
-   `V_POINTS_TO_WIN`: The number of victory points required to win a game.
-   `MODEL`: The LLM model to use for the benchmark.
-   `MAX_TOKENS`: The maximum number of tokens for the LLM's responses.
-   `USE_MINI_MAP`: A boolean to toggle between the standard and a smaller Catan map.

To run the benchmark, execute the following command:

```bash
python benchmark.py
```

The script will create a new directory in `benchmarks/` with a timestamp, e.g., `benchmarks/20250609-163423/`. This directory will contain:

-   `benchmark-results.json`: A JSON file with the detailed results of all games.
-   A `boards/` subdirectory with PNG images of the final board state for each game.

### 4. Generating Results and Figures

After the benchmark is complete, you can generate summary figures and ratings.

1.  **Update the results path:** Open `rating.py` and modify the `get_matches()` function to point to the `benchmark-results.json` file generated in the previous step.

    ```python
    def get_matches():
        # Update this path to your latest benchmark results
        return load_json("benchmarks/20250609-163423/benchmark-results.json")
    ```

2.  **Run the analysis script:** Execute the `generate_all_results.py` script:

    ```bash
    python generate_all_results.py
    ```

    This will generate a `gemini-2.5-flash.png` file in the `figures/` directory, containing plots that summarize the benchmark results. You may want to update the output filename in `generate_all_results.py` to reflect the model you used.

## Project Structure

-   `benchmark.py`: Main script to run the Catan game benchmarks.
-   `generate_all_results.py`: Script to generate result visualizations.
-   `rating.py`: Calculates player ratings using a pairwise comparison model.
-   `log_game.py`: Contains logic for logging game events (not used by default in benchmark).
-   `replay.py`: Can be used to replay games from log files.
-    `debug_game.py`: A small script to debug a single game between two players.
-   `players/`: Contains the `llm_player.py` implementation.
-   `prompts/`: Contains the different prompt strategies for the LLM player.
-   `util/`: Utility functions, e.g., for converting game state to natural language.
-   `visualization/`: Code for plotting the Catan board.
-   `benchmarks/`: Output directory for benchmark results.
-   `figures/`: Output directory for result plots.
-   `replays/`: Directory for game replays (if logging is enabled).
-   `logs/`: Directory for logs (if logging is enabled).