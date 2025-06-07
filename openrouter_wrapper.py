"""
OpenRouter wrapper for LLM requests using the openai SDK.
"""

import os
import dotenv
import openai

dotenv.load_dotenv()

with open("./sample_prompt.txt") as f:
    DEFAULT_PROMPT = f.read()


class OpenRouterWrapper:
    """
    Wrapper for OpenRouter API to make LLM requests using the openai SDK.
    """

    def __init__(
        self, api_key: str | None = None, model: str = "qwen/qwen3-8b"
    ):
        """
        Initialize OpenRouter wrapper.

        Args:
            api_key: OpenRouter API key. If None, will try to get from environment.
            model: Model to use for requests.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key must be provided or set in OPENROUTER_API_KEY environment variable"
            )

        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

        # Set up the OpenAI client for OpenRouter
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/marco-catanllm/catanllm",  # Optional, for OpenRouter analytics
                "X-Title": "Catan LLM Player",
            },
        )

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

        print("--- Sending request to OpenRouter ---")
        print(f"Model: {self.model}")
        print(f"Messages: {messages}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=32768,
            )
            print("--- Received response from OpenRouter ---")
            print(f"Raw response object: {response}")

            if response.choices:
                content = response.choices[0].message.content
                print(f"Extracted content: {repr(content)}")
                return content.strip()
            else:
                print("Warning: OpenRouter response has no choices.")
                return ""

        except Exception as e:
            print(f"--- OpenRouter API request failed ---")
            print(f"Error: {e}")
            raise RuntimeError(f"OpenRouter API request failed: {e}")

    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for Catan decision-making.

        Returns:
            str: Default system prompt
        """
        return DEFAULT_PROMPT 

if __name__ == "__main__":
    wrapper = OpenRouterWrapper()
    response = wrapper.generate_response("Test State", "Test Actions")
    print(response)