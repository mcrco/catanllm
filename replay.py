import streamlit as st
import json
import os
from PIL import Image

def get_game_replays():
    """Find all game replays in the replays directory."""
    replays_dir = "replays"
    if not os.path.exists(replays_dir):
        return []
    
    game_names = []
    for game_name in sorted(os.listdir(replays_dir)):
        game_path = os.path.join(replays_dir, game_name)
        if os.path.isdir(game_path):
            # check if there are any json files inside
            if any(f.endswith('.json') for f in os.listdir(game_path)):
                game_names.append(game_name)
    return game_names

def get_player_replays(game_name):
    """Get player replay files for a given game."""
    game_dir = f"replays/{game_name}"
    if not os.path.exists(game_dir):
        return []
    
    player_files = []
    for f in sorted(os.listdir(game_dir)):
        if f.endswith('.json'):
            player_files.append(os.path.splitext(f)[0])
    return player_files

st.set_page_config(layout="wide")
st.title("Catan Game Replay for LLM Players")

# --- Sidebar for selections ---
st.sidebar.title("Game Selection")
game_names = get_game_replays()
if not game_names:
    st.sidebar.warning("No game replays found in 'replays' directory.")
    st.stop()

selected_game = st.sidebar.selectbox("Select a game:", game_names)

if selected_game:
    player_names = get_player_replays(selected_game)
    if not player_names:
        st.sidebar.warning(f"No player replays found for game '{selected_game}'.")
        st.stop()
    
    # Filter for LLM players (assuming they have longer replays or a specific naming convention)
    llm_players = [p for p in player_names if "LLM" in p or "Qwen" in p or "qwen" in p or 'gemini' in p]
    if not llm_players:
        st.sidebar.warning(f"No LLM player replays found for game '{selected_game}'.")
        llm_players = player_names # Fallback to all players if no specific LLM player found
    
    selected_player = st.sidebar.selectbox("Select a player to view their turns:", llm_players)

    if selected_player:
        replay_file = f"replays/{selected_game}/{selected_player}.json"
        
        try:
            with open(replay_file, 'r') as f:
                player_replay = json.load(f)
        except FileNotFoundError:
            st.error(f"Replay file not found: {replay_file}")
            st.stop()
        except json.JSONDecodeError:
            st.error(f"Error decoding JSON from {replay_file}. The file might be corrupted or empty.")
            st.stop()
            
        if not player_replay:
            st.warning("Player replay is empty.")
            st.stop()

        # --- Main content ---
        
        # Turn navigation state
        if 'turn_index' not in st.session_state or st.session_state.get('game_player') != f"{selected_game}-{selected_player}":
            st.session_state.turn_index = 0
            st.session_state.game_player = f"{selected_game}-{selected_player}"

        total_turns = len(player_replay)
        
        # Turn slider
        st.session_state.turn_index = st.slider(
            "Select Turn", 0, total_turns - 1, st.session_state.turn_index
        )
        
        turn_data = player_replay[st.session_state.turn_index]

        st.header(f"Turn {st.session_state.turn_index + 1}/{total_turns} for {selected_player}")

        # --- Display Area ---
        col1, col2 = st.columns(2)

        with col1:
            # Board image
            st.subheader("Board State")
            if 'board_image_path' in turn_data and turn_data['board_image_path'] and os.path.exists(turn_data['board_image_path']):
                try:
                    image = Image.open(turn_data['board_image_path'])
                    st.image(image, caption="Board state before LLM's move", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load image: {e}")
            else:
                st.warning("Board image not found for this turn.")
        
        with col2:
            st.subheader("LLM Interaction Details")
            # Tabs for details
            tab_titles = ["Reasoning", "Response", "Available Actions", "Game State"]
            tabs = st.tabs(tab_titles)
            
            details = {
                "Reasoning": turn_data.get('reasoning', 'No reasoning recorded.'),
                "Response": turn_data.get('response', 'No response recorded.'),
                "Available Actions": turn_data.get('actions', 'No actions recorded.'),
                "Game State": turn_data.get('game_state', 'No game state recorded.')
            }

            for tab, title in zip(tabs, tab_titles):
                with tab:
                    st.text_area(title, details[title], height=600, key=f"text_{title}_{st.session_state.turn_index}")
else:
    st.info("Select a game from the sidebar to begin.") 