import base64
import json
import sqlite3

from openai import OpenAI
from rich import print

from scenario.scenario import Scenario


class OutputParser:
    def __init__(self, sce: Scenario, client: OpenAI, model_name: str = "llama3-70b-8192", is_ollama: bool = False) -> None:
        self.sce = sce
        self.client = client
        self.model_name = model_name
        self.is_ollama = is_ollama

    def agentRun(self, final_results: dict) -> dict:
        print('[green]Output parser is running...[/green]')

        combined_input = f"{final_results.get('answer', '')} {final_results.get('thoughts', '')}"

        system_prompt = """
        You are a JSON formatting assistant.
        Extract the decision from the text and output valid JSON.

        Output Schema:
        {
            "action_id": int, // 0: change_lane_left, 1: keep_speed, 2: change_lane_right, 3: accelerate, 4: decelerate
            "action_name": str, // Must match action_id (e.g. "change_lane_left")
            "explanation": str // Summary in 40 words
        }
        """

        user_prompt = f"Here is the model output:\n{combined_input}\n\nProvide the JSON response."

        try:
            api_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0
            }

            # Ollama may not support response_format parameter
            if not self.is_ollama:
                api_params["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**api_params)

            content = response.choices[0].message.content

            # Extract JSON from markdown code blocks if present (common with Ollama)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            self.parseredOutput = json.loads(content)
            self.dataCommit()
            print("[cyan]Parser finished.[/cyan]")
            return self.parseredOutput

        except Exception as e:
            print(f"[red]Parser Error: {e}[/red]")
            return {
                "action_id": 1,
                "action_name": "keep_speed",
                "explanation": "Error in parsing, defaulting to keep speed"
            }

    def dataCommit(self):
        conn = sqlite3.connect(self.sce.database)
        cur = conn.cursor()

        if hasattr(self, 'parseredOutput'):
            json_str = json.dumps(self.parseredOutput)
            base64Output = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')

            cur.execute(
                "UPDATE decisionINFO SET outputParser = ? WHERE frame = ?",
                (base64Output, self.sce.frame)
            )
            conn.commit()
        conn.close()
