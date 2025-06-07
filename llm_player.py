"""
LLM Player for Catanatron using OpenRouter or vLLM.
"""

import re
import random
import logging
from typing import Union, List

from catanatron import Game, Player, Action, Color

from game_convert import game_to_natural_language, format_playable_actions
from openrouter_wrapper import OpenRouterWrapper
from vllm_wrapper import VLLMWrapper


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMPlayer(Player):
    """
    LLM-powered Catan player that uses either OpenRouter or vLLM for decision making.
    """
    
    def __init__(self, color: Color, llm_wrapper: Union[OpenRouterWrapper, VLLMWrapper], name: str = None):
        """
        Initialize LLM player.
        
        Args:
            color: Player color
            llm_wrapper: Either OpenRouterWrapper or VLLMWrapper instance
            name: Optional player name
        """
        super().__init__(color)
        self.llm_wrapper = llm_wrapper
        self.name = name or f"LLM-{color}"
        self.last_llm_response = None
        
        logger.info(f"Initialized {self.name} with {type(llm_wrapper).__name__}")
    
    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        """
        Make a decision using the LLM.
        
        Args:
            game: Current game state
            playable_actions: List of available actions
        
        Returns:
            Action: Chosen action from playable_actions
        """
        if not playable_actions:
            raise ValueError("No playable actions available")
        
        # If only one action available, choose it without LLM call
        if len(playable_actions) == 1:
            logger.info(f"{self.name}: Only one action available, choosing: {playable_actions[0]}")
            return playable_actions[0]
        
        try:
            # Convert game state to natural language
            game_state_nl = game_to_natural_language(game, self.color)
            actions_nl = format_playable_actions(playable_actions)
            
            logger.info(f"{self.name}: Making LLM decision with {len(playable_actions)} actions")
            
            # Get LLM response
            llm_response = self.llm_wrapper.generate_response(game_state_nl, actions_nl)
            self.last_llm_response = llm_response  # Store the LLM output
            print(f"{self.name}: LLM response: {llm_response}")
            
            # Parse the response to extract action index
            chosen_action = self._parse_llm_response(llm_response, playable_actions)
            
            logger.info(f"{self.name}: Chose action {playable_actions.index(chosen_action)}: {chosen_action}")
            return chosen_action
            
        except Exception as e:
            error_msg = f"LLM failed to make a decision: {type(e).__name__}: {e}"
            self.last_llm_response = error_msg
            logger.error(f"{self.name}: Error in LLM decision making: {e}")
            logger.info(f"{self.name}: Falling back to random action selection")
            return self._fallback_decision(playable_actions)
    
    def _parse_llm_response(self, response: str, playable_actions: List[Action]) -> Action:
        """
        Parse LLM response to extract the chosen action index.
        
        Args:
            response: Raw LLM response
            playable_actions: List of available actions
        
        Returns:
            Action: Parsed action from the response
        """
        logger.debug(f"{self.name}: LLM response: {response}")
        
        # Try to find a number in the response
        # Look for patterns like "2", "Action 2", "option 2", etc.
        patterns = [
            r'(?:^|\s)(\d+)(?:\s|$)',  # Standalone number
            r'(?:action|option|choice)\s*(\d+)',  # "action 2", "option 2"
            r'(\d+)(?:\s*:|\s*\.|$)',  # "2:" or "2."
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                try:
                    action_index = int(matches[0])
                    if 0 <= action_index < len(playable_actions):
                        return playable_actions[action_index]
                except (ValueError, IndexError):
                    continue
        
        # If no valid number found, try to find the first occurrence of any digit
        digits = re.findall(r'\d+', response)
        for digit_str in digits:
            try:
                action_index = int(digit_str)
                if 0 <= action_index < len(playable_actions):
                    return playable_actions[action_index]
            except (ValueError, IndexError):
                continue
        
        # If still no valid action found, log warning and use fallback
        logger.warning(f"{self.name}: Could not parse action from LLM response: {response[:100]}...")
        return self._fallback_decision(playable_actions)
    
    def _fallback_decision(self, playable_actions: List[Action]) -> Action:
        """
        Fallback decision when LLM fails or response can't be parsed.
        Uses simple heuristics instead of pure random.
        
        Args:
            playable_actions: List of available actions
        
        Returns:
            Action: Chosen action using fallback logic
        """
        # Simple heuristic: prioritize certain action types
        action_priorities = {
            'BUILD_SETTLEMENT': 10,
            'BUILD_CITY': 9,
            'BUILD_ROAD': 7,
            'BUY_DEVELOPMENT_CARD': 6,
            'PLAY_KNIGHT_CARD': 8,
            'PLAY_YEAR_OF_PLENTY': 5,
            'PLAY_ROAD_BUILDING': 6,
            'PLAY_MONOPOLY': 4,
            'MARITIME_TRADE': 3,
            'ROLL': 2,
            'END_TURN': 1,
        }
        
        # Score actions by priority
        scored_actions = []
        for action in playable_actions:
            action_type = action[0].name
            priority = action_priorities.get(action_type, 0)
            scored_actions.append((priority, action))
        
        # Sort by priority (highest first) and choose the best
        scored_actions.sort(key=lambda x: x[0], reverse=True)
        
        # Add some randomness among top actions
        top_priority = scored_actions[0][0]
        top_actions = [action for priority, action in scored_actions if priority == top_priority]
        
        chosen_action = random.choice(top_actions)
        logger.info(f"{self.name}: Fallback chose action: {chosen_action}")
        return chosen_action


# Convenience functions for creating LLM players
def create_openrouter_player(color: Color, api_key: str = None, model: str = "anthropic/claude-3.5-sonnet", name: str = None) -> LLMPlayer:
    """
    Create an LLM player using OpenRouter.
    
    Args:
        color: Player color
        api_key: OpenRouter API key
        model: Model to use
        name: Optional player name
    
    Returns:
        LLMPlayer: Configured LLM player
    """
    wrapper = OpenRouterWrapper(api_key=api_key, model=model)
    return LLMPlayer(color, wrapper, name)


def create_vllm_player(color: Color, base_url: str = "http://localhost:8000", model_name: str = None, name: str = None) -> LLMPlayer:
    """
    Create an LLM player using vLLM.
    
    Args:
        color: Player color
        base_url: vLLM server URL
        model_name: Model name (optional)
        name: Optional player name
    
    Returns:
        LLMPlayer: Configured LLM player
    """
    wrapper = VLLMWrapper(base_url=base_url, model_name=model_name)
    return LLMPlayer(color, wrapper, name)
