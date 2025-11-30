import os
import re

import dotenv
from openai import OpenAI

from lmp_driver.prompts import SYSTEM_PROMPT

dotenv.load_dotenv()


class LLMAgent:
    """
    Real implementation using OpenAI API to generate driving policies.
    """

    def __init__(self, model_name="openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")

        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model_name = model_name

    def generate_policy(self, instruction, env_info):
        """
        env_info: dict containing weather, time, density
        """
        context_str = f"""
        User Instruction: "{instruction}"
        Current Environment:
        - Weather: {env_info.get('weather', 'Clear')}
        - Time: {env_info.get('time_of_day', 'Day')}
        - Traffic Density: {env_info.get('density', 'Normal')}
        """

        print(f"Context Sent to LLM:\n{context_str}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context_str}
            ],
            temperature=0.0
        )

        return self._clean_code(response.choices[0].message.content)

    def _clean_code(self, raw_text):
        """
        Extracts pure Python code from the LLM's response.
        Handles cases where the LLM wraps code in markdown ```python ... ```
        """
        # Regex to find code inside ```python ... ``` or just ``` ... ```
        code_match = re.search(r'```(?:python)?\n(.*?)```', raw_text, re.DOTALL)

        if code_match:
            return code_match.group(1).strip()

        # We strip non-code conversational lines if they start with #
        lines = [line for line in raw_text.split('\n') if
                 not line.strip().lower().startswith(('here', 'sure', 'i have'))]
        return "\n".join(lines).strip()
