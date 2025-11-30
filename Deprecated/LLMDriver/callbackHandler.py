from typing import List

from rich import print


class CustomHandler:
    def __init__(self):
        self.memory: List[str] = []

    def log_thought(self, content: str):
        if content:
            self.memory.append(content)

    def log_tool_call(self, tool_name: str, tool_args: str):
        log_entry = f"Action: {tool_name}\nAction Input: {tool_args}"
        self.memory.append(log_entry)
        print(f"[blue]{log_entry}[/blue]")

    def log_observation(self, output: str):
        log_entry = f"Observation: {output}"
        if self.memory:
            self.memory[-1] += f"\n{log_entry}\n"
        else:
            self.memory.append(log_entry)
