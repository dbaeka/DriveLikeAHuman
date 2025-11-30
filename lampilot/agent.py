import os
import re
import dotenv
from openai import OpenAI

from lampilot.prompts import SYSTEM_PROMPT

dotenv.load_dotenv()

class LLMAgent:
    """
    Real implementation using OpenAI API to generate driving policies.
    """

    def __init__(self, model_name="openai/gpt-oss-20b"):
        # Ensure API key is set
        api_key = os.getenv("GROQ_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")

        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model_name = model_name

    def generate_policy(self, instruction):
        """
        Sends the user instruction to the LLM and retrieves the Python code.
        """
        print(f"🤖 Connecting to OpenAI ({self.model_name})...")
        print(f"   Instruction: '{instruction}'")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Instruction: {instruction}"}
                ],
                temperature=0.0  # Deterministic for code generation
            )

            raw_content = response.choices[0].message.content
            return self._clean_code(raw_content)

        except Exception as e:
            print(f"❌ OpenAI API Error: {e}")
            # Fallback safe policy if API fails
            return "def policy(api): api.slow_down()"

    def _clean_code(self, raw_text):
        """
        Extracts pure Python code from the LLM's response.
        Handles cases where the LLM wraps code in markdown ```python ... ```
        """
        # Regex to find code inside ```python ... ``` or just ``` ... ```
        code_match = re.search(r'```(?:python)?\n(.*?)```', raw_text, re.DOTALL)

        if code_match:
            return code_match.group(1).strip()

        # If no markdown tags, assume the whole text is code (risky but necessary fallback)
        # We strip non-code conversational lines if they start with #
        lines = [line for line in raw_text.split('\n') if
                 not line.strip().lower().startswith(('here', 'sure', 'i have'))]
        return "\n".join(lines).strip()
