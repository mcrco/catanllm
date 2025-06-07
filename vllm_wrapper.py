"""
vLLM wrapper for local LLM inference.
"""

import requests


class VLLMWrapper:
    """
    Wrapper for vLLM server to make local LLM requests.
    """

    def __init__(
        self, base_url: str = "http://localhost:8000", model_name: str | None = None
    ):
        """
        Initialize vLLM wrapper.

        Args:
            base_url: Base URL of the vLLM server
            model_name: Name of the model (optional, vLLM server usually serves one model)
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.completions_url = f"{self.base_url}/v1/chat/completions"

    def generate_response(
        self, game_state: str, available_actions: str, system_prompt: str | None = None
    ) -> str:
        """
        Generate a response from the LLM given game state and available actions.

        Args:
            game_state: Natural language description of the game state
            available_actions: Natural language description of available actions
            system_prompt: Optional system prompt to guide the LLM

        Returns:
            str: LLM response
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        user_message = f"{game_state}\n\n{available_actions}\n\nBased on the current game state and available actions, choose the best action by responding with the action number (e.g., '2' for action 2). Provide a brief explanation of your reasoning."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        payload = {
            "messages": messages,
            "temperature": 0.1,  # Low temperature for more consistent decision-making
            "max_tokens": 500,
            "stream": False,
        }

        # Add model name if specified
        if self.model_name:
            payload["model"] = self.model_name

        try:
            response = requests.post(
                self.completions_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,  # Local inference might be slower
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"vLLM API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response format from vLLM: {e}")

    def generate_response_legacy(
        self, game_state: str, available_actions: str, system_prompt: str = None
    ) -> str:
        """
        Generate response using legacy completions endpoint (for older vLLM versions).

        Args:
            game_state: Natural language description of the game state
            available_actions: Natural language description of available actions
            system_prompt: Optional system prompt to guide the LLM

        Returns:
            str: LLM response
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        # Format as a single prompt for legacy endpoint
        prompt = f"{system_prompt}\n\nGame State:\n{game_state}\n\n{available_actions}\n\nBased on the current game state and available actions, choose the best action by responding with the action number (e.g., '2' for action 2). Provide a brief explanation of your reasoning.\n\nResponse:"

        payload = {
            "prompt": prompt,
            "temperature": 0.1,
            "max_tokens": 500,
            "stream": False,
        }

        # Add model name if specified
        if self.model_name:
            payload["model"] = self.model_name

        try:
            legacy_url = f"{self.base_url}/generate"
            response = requests.post(
                legacy_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            return result["text"][0].strip()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"vLLM legacy API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response format from vLLM legacy: {e}")

    def is_available(self) -> bool:
        """
        Check if the vLLM server is available.

        Returns:
            bool: True if server is available, False otherwise
        """
        try:
            health_url = f"{self.base_url}/health"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for Catan decision-making.

        Returns:
            str: Default system prompt
        """
        return """You are an expert Settlers of Catan player. Your goal is to win the game by reaching 10 victory points first.

Key strategies to consider:
1. Prioritize building settlements and cities for victory points and resource generation
2. Build roads to connect settlements and block opponents
3. Collect and manage resources efficiently
4. Use development cards strategically
5. Consider the robber's position and its impact on resource generation
6. Plan for different resource types and their scarcity
7. Think about longest road and largest army bonuses

When choosing actions:
- Consider both immediate benefits and long-term strategy
- Think about resource efficiency and opportunity cost
- Be aware of what other players might do on their turns
- Prioritize actions that bring you closer to victory

Always respond with just the action number followed by a brief explanation on a new line."""
