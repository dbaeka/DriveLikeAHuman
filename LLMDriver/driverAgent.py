import json
import sqlite3

from openai import OpenAI
from rich import print

from LLMDriver.agent_propmts import (
    SYSTEM_MESSAGE_PREFIX,
    SYSTEM_MESSAGE_SUFFIX,
    TRAFFIC_RULES,
    DECISION_CAUTIONS
)
from LLMDriver.callbackHandler import CustomHandler
from scenario.scenario import Scenario


class DriverAgent:
    def __init__(
            self,
            client: OpenAI,
            toolModels: list,
            sce: Scenario,
            model_name: str = "openai/gpt-oss-20b",
            verbose: bool = False,
            is_ollama: bool = False,
            weather: str = "sunny",
            instruction: str = "Drive safely"
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.sce = sce
        self.verbose = verbose
        self.is_ollama = is_ollama
        self.ch = CustomHandler()
        self.weather = weather
        self.instruction = instruction

        self.tool_registry = {}
        self.tools_schema = []

        for ins in toolModels:
            func = getattr(ins, 'inference')

            # Sanitize name
            original_name = getattr(func, 'name', 'UnknownTool')
            clean_name = original_name.replace(" ", "_")
            description = getattr(func, 'description', '')

            self.tool_registry[clean_name] = func

            self.tools_schema.append({
                "type": "function",
                "function": {
                    "name": clean_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The input argument for the tool"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })

    def agentRun(self, last_step_decision: dict):
        print(f'Decision at frame {self.sce.frame} is running ...')
        print('[green]Driver agent is running (No-LangChain)...[/green]')
        self.ch.memory = []

        if last_step_decision and "action_name" in last_step_decision:
            last_step_action = last_step_decision["action_name"]
            last_step_explanation = last_step_decision["explanation"]
        else:
            last_step_action = "Not available"
            last_step_explanation = "Not available"

        # Removed FORMAT_INSTRUCTIONS to prevent ReAct format conflict
        system_prompt = f"""
        {SYSTEM_MESSAGE_PREFIX}

        RULES:
        {TRAFFIC_RULES}

        CAUTIONS:
        {DECISION_CAUTIONS}

        {SYSTEM_MESSAGE_SUFFIX}
        """

        user_prompt = f"""
        You, the 'ego' car, are now driving a car on a highway. You have already drive for {self.sce.frame} seconds.
        The decision you made LAST time step was `{last_step_action}`. Your explanation was `{last_step_explanation}`. 
        Here is the current scenario: \n ```json\n{self.sce.export2json()}\n```\n. 
        Please make decision for the `ego` car. Analyze the state, use tools if needed, and output your decision.

        Output format when finished:
        Final Answer: 
            "decision":{{"ego car's decision..."}},
            "expalanations":{{"your explaination..."}}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        max_steps = 10

        for _ in range(max_steps):
            try:
                # Ollama may need slightly different parameters
                api_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.0
                }

                # Only add tools if not using Ollama or if Ollama supports it
                if not self.is_ollama or self.tools_schema:
                    api_params["tools"] = self.tools_schema
                    api_params["tool_choice"] = "auto"

                response = self.client.chat.completions.create(**api_params)

                msg = response.choices[0].message
                content = msg.content
                tool_calls = msg.tool_calls

                if content:
                    self.ch.log_thought(content)
                    if "Final Answer:" in content:
                        break

                if tool_calls:
                    messages.append(msg)

                    for tool_call in tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)

                        self.ch.log_tool_call(func_name, str(func_args))

                        if func_name in self.tool_registry:
                            tool_func = self.tool_registry[func_name]
                            try:
                                # Safe extraction of 'query' or default to first arg
                                arg_val = func_args.get("query")
                                if arg_val is None and len(func_args) > 0:
                                    # Fallback if model used a different key name like 'input' or 'action_input'
                                    arg_val = list(func_args.values())[0]

                                tool_result = tool_func(str(arg_val))
                            except Exception as e:
                                tool_result = f"Error executing tool: {e}"
                        else:
                            tool_result = f"Error: Tool {func_name} not found."

                        self.ch.log_observation(str(tool_result))

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": str(tool_result)
                        })
                else:
                    break

            except Exception as e:
                print(f"[red]Error in Agent Loop: {e}[/red]")
                break

        print('[cyan]Final decision reached.[/cyan]')
        if self.ch.memory:
            print(self.ch.memory[-1])
            self.dataCommit()

    def exportThoughts(self):
        output = {}
        if not self.ch.memory:
            return {"thoughts": "", "answer": ""}

        last_entry = self.ch.memory[-1]
        if 'Final Answer:' in last_entry:
            try:
                output['thoughts'], output['answer'] = last_entry.split('Final Answer:', 1)
            except ValueError:
                output['thoughts'] = last_entry
                output['answer'] = ""
        else:
            output['thoughts'] = "\n".join(self.ch.memory)
            output['answer'] = ""
        return output

    def dataCommit(self):
        scenario_data = self.sce.export2json()
        thinkAndThoughts = '\n'.join(self.ch.memory[:-1]) if len(self.ch.memory) > 1 else ""
        finalAnswer = self.ch.memory[-1] if self.ch.memory else ""

        conn = sqlite3.connect(self.sce.database)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO decisionINFO
               VALUES (?, ?, ?, ?, ?)""",
            (self.sce.frame, scenario_data, thinkAndThoughts, finalAnswer, '')
        )
        conn.commit()
        conn.close()
