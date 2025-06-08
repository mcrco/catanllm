"""
OpenRouter wrapper for LLM requests using the openai SDK.
"""

import os
import dotenv
import openai

dotenv.load_dotenv()

with open("./sample_prompt.txt") as f:
    DEFAULT_PROMPT = f.read()


class LLMWrapper:
    """
    Wrapper for OpenRouter API to make LLM requests using the openai SDK.
    """

    def __init__(
        self, base_url: str, api_key: str | None = None, model: str = "Qwen/Qwen3-1.7B", max_tokens: int = 16384
    ):
        """
        Initialize OpenRouter wrapper.

        Args:
            api_key: OpenRouter API key. If None, will try to get from environment.
            model: Model to use for requests.
        """
        
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/marco-catanllm/catanllm",
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content.strip()

    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for Catan decision-making.

        Returns:
            str: Default system prompt
        """
        return DEFAULT_PROMPT 

if __name__ == "__main__":
    wrapper = LLMWrapper(base_url='http://localhost:8000')
    response = wrapper.generate_response("Test State", "Test Actions")
    print(response)