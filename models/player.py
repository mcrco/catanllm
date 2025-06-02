import os
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

class UnslothCatanModel:
    def __init__(self, model_name="unsloth/qwen3-8b-instruct", rules_file="catan_rules.txt", chat_template_name="qwen2.5"):
        """
        Initializes the Unsloth Catan Model.

        Args:
            model_name (str): The name of the Unsloth model to use.
            rules_file (str): Path to the file containing Catan rules.
            chat_template_name (str): The name of the chat template to use (e.g., "qwen2.5", "chatml").
        """
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

        # Apply the chat template to the tokenizer
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=chat_template_name,
        )

        self.model_name = model_name
        self.system_prompt_content = self._load_rules(rules_file)
        print(f"Model {self.model_name} initialized. Chat template '{chat_template_name}' applied. System prompt loaded from {rules_file}.")

    def _load_rules(self, rules_file):
        """Loads Catan rules from a file to be used as the system prompt."""
        # Construct the path relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_rules_path = os.path.join(base_dir, rules_file)
        try:
            with open(full_rules_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Rules file not found at {full_rules_path}")
            return "Default Catan rules: Be strategic and win." # Fallback

    def predict_action(self, game_state_text, player_id):
        """
        Predicts the next action for a player given the current game state.

        Args:
            game_state_text (str): A natural language description of the current game state.
            player_id (int): The ID of the player for whom to predict the action.

        Returns:
            str: The predicted action (e.g., "build road", "trade wood for sheep with player 2").
        """
        user_prompt = f"""Current Catan Game State:
{game_state_text}

You are Player {player_id}. What is your next action? Consider your resources, development cards, board position, and available actions."""

        messages = [
            {"role": "system", "content": self.system_prompt_content},
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=150,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_tokens = outputs[0][inputs.shape[1]:]
        action_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        if not action_text:
            action_text = f"Placeholder: Model returned empty action for Player {player_id}. Defaulting to roll dice."

        print(f"--- Raw model output for Player {player_id} (decoded new tokens) ---")
        print(action_text)
        print("--- End Raw Model Output ---")
        
        return action_text

if __name__ == '__main__':
    # Example Usage (assuming catan_rules.txt is in the same directory or path is adjusted)
    print("Initializing UnslothCatanModel...")
    # Adjust path if running this script directly and catan_rules.txt is in the parent 'models' dir
    # For this example, we assume catan_rules.txt is in the same directory as this script.
    # To make it robust, let's ensure it looks in the correct place.
    rules_path = os.path.join(os.path.dirname(__file__), "catan_rules.txt")
    if not os.path.exists(rules_path):
        # Create a dummy rules file for the example if it doesn't exist
        print(f"Creating dummy catan_rules.txt at {rules_path} for example.")
        with open(rules_path, "w", encoding='utf-8') as f:
            f.write("These are the basic rules of Catan. The goal is to reach 10 victory points.")

    model = UnslothCatanModel(rules_file="catan_rules.txt") # Will look for catan_rules.txt in the same dir

    game_state_example = (
        "Board: Standard setup. Robber on desert.\n"
        "Player 1 (Red): 3 victory points, Resources: 2 Wood, 1 Brick, 0 Sheep, 1 Wheat, 0 Ore. Dev Cards: 0.\n"
        "Player 2 (Blue): 2 victory points, Resources: 1 Wood, 2 Brick, 1 Sheep, 0 Wheat, 1 Ore. Dev Cards: 1 (Knight).\n"
        "It's Player 1's turn.\n"
        "Available actions for Player 1: Roll dice, Offer trade, Build road (needs 1 wood, 1 brick - has enough), Buy development card (needs 1 ore, 1 sheep, 1 wheat - needs sheep, ore)."
    )
    print("\nPredicting action for Player 1...")

    if not torch.cuda.is_available():
        print("CUDA not available, simulating for example. Model will run on CPU.")

    action = model.predict_action(game_state_example, player_id=1)
    print(f"\nPredicted action for Player 1: {action}")

    # Clean up dummy rules file if created
    if os.path.exists(rules_path) and "dummy catan_rules.txt" in open(rules_path, 'r', encoding='utf-8').read(100):
        print(f"Dummy rules file ({rules_path}) can be manually removed if no longer needed.") 