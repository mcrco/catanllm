"""
LLM Player for Catanatron using OpenRouter or vLLM.
"""

from typing import List
from dotenv import load_dotenv
from catanatron import Game, Player, Action, Color, ActionType

from util.game_convert import game_to_natural_language, format_playable_actions
from util.llm_wrapper import LLMWrapper


load_dotenv()


class LLMPlayer(Player):
    """
    LLM-powered Catan player that uses either OpenRouter or vLLM for decision making.
    """

    def __init__(
        self,
        color: Color,
        name: str = None,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        prompt: str = None,
        thinking: bool = True,
        max_tokens: int = 16384,
        use_mini_map: bool = False,
    ):
        """
        Initialize LLM player.

        Args:
            color: Player color
            name: Optional player name
            base_url: Base URL for the LLM API.
            api_key: API key for the provider.
            model: The model to use for generating responses.
            prompt: Optional custom prompt for the LLM.
            **kwargs: Additional keyword arguments for the LLMWrapper.
        """
        super().__init__(color)
        if 'gemini' in model.lower() and 'thinking' not in model.lower() and thinking:
            model = f"{model}:thinking"
        self.model = model
        self.llm_wrapper = LLMWrapper(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            thinking=thinking,
            max_tokens=max_tokens,
            use_mini_map=use_mini_map,
        )
        if 'qwen' in model.lower() and not thinking:
            model = f"{model}:no_think"
        self.name = name or f"LLM-{color}"
        self.prompt = prompt
        self.log = []
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.fallback_count = 0
        self.decision_count = 0
        self.use_mini_map = use_mini_map

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        """
        Make a decision using the LLM.

        Args:
            game: Current game state
            playable_actions: List of available actions

        Returns:
            Action: Chosen action from playable_actions
        """
        self.decision_count += 1
        if not playable_actions:
            raise ValueError("No playable actions available")

        if len(playable_actions) == 1:
            return playable_actions[0]

        try:
            # Convert game state to natural language
            game_state_nl = game_to_natural_language(game, self.color)
            if self.log:
                notes = [log["notes"] for log in self.log]
                notes_str = "\n".join(notes)
                game_state_nl += f"\n\nPrevious moves: {notes_str}"
            actions_nl = format_playable_actions(playable_actions)

            # Get LLM response
            response, reasoning = self.llm_wrapper.generate_response(
                game_state_nl, actions_nl
            )
            response_lines = response.split("\n")

            # Get notes
            if len(response_lines) >= 2:
                notes = response_lines[-2]
            else:
                notes = response_lines[0]

            # Get reasoning from response for non-reasoning models
            if reasoning is None and not self.thinking and len(response_lines) > 2:
                extracted_reasoning = "\n".join(response_lines[:-2])
                reasoning = (
                    f"No reasoning provided. Assuming no_think:\n{extracted_reasoning}"
                )
                response = "\n".join(response.split("\n")[-2:])

            log_entry = {
                "reasoning": reasoning,
                "notes": notes,
                "response": response,
                "actions": actions_nl,
                "game_state": game_state_nl,
            }
            if hasattr(self, "current_board_image_path"):
                log_entry["board_image_path"] = self.current_board_image_path
            self.log.append(log_entry)

            # Parse the response to extract action index
            chosen_action = self._parse_llm_response(response, playable_actions)
            return chosen_action

        except Exception as e:
            print(e)
            return self._fallback_decision(playable_actions)

    def _parse_llm_response(
        self, response: str, playable_actions: List[Action]
    ) -> Action:
        """
        Parse LLM response to extract the chosen action index.

        Args:
            response: Raw LLM response
            playable_actions: List of available actions

        Returns:
            Action: Parsed action from the response
        """
        action_index = int(response.split("\n")[-1])
        if 0 <= action_index < len(playable_actions):
            return playable_actions[action_index]

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
        self.fallback_count += 1
        if len(playable_actions) == 1:
            return playable_actions[0]

        # Choose the first action
        for action in playable_actions:
            if action.action_type == ActionType.END_TURN:
                return action
        return playable_actions[0]

