"""
OpenRouter wrapper for LLM requests using the openai SDK.
"""

import os
import dotenv
import openai

dotenv.load_dotenv()

class LLMWrapper:
    """
    Wrapper for OpenRouter API to make LLM requests using the openai SDK.
    """

    def __init__(
        self, base_url: str, api_key: str | None = None, model: str = "qwen/qwen3-8b", max_tokens: int | None = None, prompt: str = 'basic', thinking: bool = True, use_mini_map: bool = False
    ):
        """
        Initialize OpenRouter wrapper.

        Args:
            api_key: OpenRouter API key. If None, will try to get from environment.
            model: Model to use for requests.
        """
        prompt_dir = "prompts"
        if use_mini_map:
            prompt_dir = os.path.join(prompt_dir, "mini")

        with open(os.path.join(prompt_dir, "basic_prompt.txt")) as f:
            BASIC_PROMPT = f.read()
        with open(os.path.join(prompt_dir, "search_prompt.txt")) as f:
            SEARCH_PROMPT = f.read()
        with open(os.path.join(prompt_dir, "strategy_prompt.txt")) as f:
            STRATEGY_PROMPT = f.read()
        with open(os.path.join(prompt_dir, "rules_prompt.txt")) as f:
            RULES_PROMPT = f.read()
        
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.thinking = thinking
        if prompt == 'basic':
            self.prompt = BASIC_PROMPT
        elif prompt == 'search':
            self.prompt = SEARCH_PROMPT
        elif prompt == 'strategy':
            self.prompt = STRATEGY_PROMPT
        elif prompt == 'rules':
            self.prompt = RULES_PROMPT
        else:
            raise ValueError(f"Invalid prompt: {prompt}")
        if not thinking and 'qwen' in self.model.lower():
            self.prompt = self.prompt + "\n\n/no_think"
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
        user_message = f"{game_state}\n\n{available_actions}\n\nBased on the current game state and available actions, provide a brief explanation of your reasoning and choose the best action by responding with the action number (e.g., '2' for action 2)."

        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": user_message},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=self.max_tokens,
            extra_body={
                'reasoning': {
                    'max_tokens': int(self.max_tokens * 0.75),
                }
            }
        )

        if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning is not None:
            reasoning = response.choices[0].message.reasoning.strip()
        else:
            reasoning = None

        response_text = response.choices[0].message.content.strip()
        return response_text, reasoning


if __name__ == "__main__":
    wrapper = LLMWrapper(base_url='http://localhost:8000')
    response = wrapper.generate_response("Test State", "Test Actions")
    print(response)