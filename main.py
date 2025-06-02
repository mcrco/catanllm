from models.player import UnslothCatanModel
from game_engine.engine import CatanatronWrapper
import os

def main():
    print("Starting Catan LLM Comparator Demo...")

    # --- Setup --- 
    # Ensure catan_rules.txt is accessible by UnslothCatanModel
    # The UnslothCatanModel expects catan_rules.txt in its own directory (models/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rules_file_path = os.path.join(script_dir, "models", "catan_rules.txt")

    # Create a dummy rules file if it doesn't exist, for demo purposes
    if not os.path.exists(rules_file_path):
        print(f"Dummy rules file not found at {rules_file_path}. Creating one for demo.")
        os.makedirs(os.path.dirname(rules_file_path), exist_ok=True)
        with open(rules_file_path, "w") as f:
            f.write("Basic Catan Rules: Score 10 VPs to win. Trade and build wisely.")
    else:
        print(f"Using existing rules file: {rules_file_path}")

    # Initialize the Catanatron wrapper (simulated game engine)
    print("\nInitializing Catanatron Wrapper...")
    catanatron = CatanatronWrapper()

    # Initialize the Unsloth Catan Model (simulated LLM player)
    # It will look for 'catan_rules.txt' inside the 'models' directory relative to its own location
    print("\nInitializing Unsloth Catan Model...")
    # The model class itself handles finding catan_rules.txt in its own directory
    llm_player = UnslothCatanModel(rules_file="catan_rules.txt") 

    # --- Game Loop (Simplified Example) --- 
    print("\n--- Starting Simplified Game Loop ---")
    for turn in range(1, 3): # Simulate a couple of turns
        print(f"\n--- Turn {turn} ---")

        # 1. Get current game state from Catanatron
        print("Catanatron: Fetching current game state...")
        structured_state = catanatron.get_current_game_state()
        
        # 2. Convert game state to natural language for the LLM
        print("Catanatron: Converting state to natural language...")
        natural_language_state = catanatron.convert_state_to_natural_language(structured_state)
        # print(f"\nLLM Input (Natural Language State):\n{natural_language_state}") # Optional: print the full state

        current_player_id = structured_state.get("current_player_id", "Unknown")
        if current_player_id == "Unknown":
            print("Error: Could not determine current player.")
            break

        # 3. LLM predicts action
        print(f"LLM (Player {current_player_id}): Predicting action...")
        predicted_action = llm_player.predict_action(natural_language_state, player_id=current_player_id)
        print(f"LLM (Player {current_player_id}): Predicted action: {predicted_action}")

        # 4. Send action to Catanatron to update game state (simulated)
        print(f"Catanatron: Player {current_player_id} performing action: {predicted_action}...")
        # In a real scenario, predicted_action would need to be parsed into a Catanatron-understandable format
        action_success = catanatron.perform_action(player_id=current_player_id, action_details={"type": "llm_generated_action", "description": predicted_action})
        
        if action_success:
            print("Catanatron: Action processed.")
            # Simulate next player's turn (very basic, Catanatron would handle this)
            if structured_state["current_player_id"] == 1:
                structured_state["current_player_id"] = 2
            else:
                structured_state["current_player_id"] = 1
            structured_state["turn_phase"] = "rolling" # Reset phase for next player
        else:
            print("Catanatron: Action failed or was invalid.")
            # Potentially re-prompt LLM or handle error

        if structured_state.get("winner") is not None:
            print(f"\nGame Over! Player {structured_state['winner']} wins!")
            break

    print("\n--- End of Demo ---")

if __name__ == "__main__":
    main() 