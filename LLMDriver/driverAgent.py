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
        print(f'\n{"#" * 100}')
        print(f'# FRAME {self.sce.frame} - DECISION MAKING START')
        print(f'{"#" * 100}')
        print(f'[green]Driver agent is running...[/green]')

        # Print all vehicles on screen
        print(f"\n[DEBUG] === ALL VEHICLES ON SCREEN ===")
        ego = self.sce.vehicles.get('ego')
        if ego:
            print(f"[DEBUG] EGO: lane={ego.lane_id}, pos={ego.lanePosition:.2f}m, speed={ego.speed:.2f}m/s")

        # Group vehicles by lane for better visualization
        lanes = {'lane_0': [], 'lane_1': [], 'lane_2': [], 'lane_3': []}
        for vid, veh in self.sce.vehicles.items():
            if vid != 'ego':
                distance_to_ego = veh.lanePosition - ego.lanePosition if ego else 0

                # BUG FIX: Handle undefined lanes gracefully to prevent crash
                if veh.lane_id in lanes:
                    lanes[veh.lane_id].append((vid, veh.lanePosition, veh.speed, distance_to_ego))
                else:
                    print(
                        f"[DEBUG] WARNING: Vehicle {vid} is in unknown lane '{veh.lane_id}' - Skipping visualization.")

                # Fix the display logic
                if distance_to_ego < 0:
                    pos_text = f"AHEAD by {abs(distance_to_ego):.2f}m"
                else:
                    pos_text = f"BEHIND by {distance_to_ego:.2f}m"
                print(
                    f"[DEBUG] {vid}: lane={veh.lane_id}, pos={veh.lanePosition:.2f}m, speed={veh.speed:.2f}m/s, {pos_text}")

        # Print lane summaries
        print(f"\n[DEBUG] === LANE SUMMARIES ===")
        for lane_id, vehicles in lanes.items():
            if vehicles:
                print(f"[DEBUG] {lane_id}: {len(vehicles)} vehicles")
                vehicles.sort(key=lambda x: x[1])  # Sort by position
                for v in vehicles:
                    relative_pos = "AHEAD" if v[3] < 0 else "BEHIND"
                    print(f"[DEBUG]   - {v[0]}: {relative_pos} by {abs(v[3]):.2f}m, speed={v[2]:.2f}m/s")
            else:
                print(f"[DEBUG] {lane_id}: EMPTY")

        self.ch.memory = []

        if last_step_decision and "action_name" in last_step_decision:
            last_step_action = last_step_decision["action_name"]
            last_step_explanation = last_step_decision["explanation"]
        else:
            last_step_action = "Not available"
            last_step_explanation = "Not available"

        print(f"\n[DEBUG] === LAST STEP INFO ===")
        print(f"[DEBUG] Last action: {last_step_action}")
        print(f"[DEBUG] Last explanation: {last_step_explanation[:100]}..." if len(
            last_step_explanation) > 100 else f"[DEBUG] Last explanation: {last_step_explanation}")

        system_prompt = f"""
        {SYSTEM_MESSAGE_PREFIX}

        RULES:
        {TRAFFIC_RULES}

        CAUTIONS:
        {DECISION_CAUTIONS}

        {SYSTEM_MESSAGE_SUFFIX}
        """

        print(f"\n[DEBUG] === MISSION CONTEXT ===")
        print(f"[DEBUG] Weather: {self.weather}")
        print(f"[DEBUG] Mission instruction: {self.instruction}")

        # SMART UPDATE: Dynamic Prompt Injection based on Benchmarks
        weather_context = ""
        safety_adjustment = ""
        if self.weather == "foggy":
            weather_context = "Heavy Fog. Visibility is low. Sensor noise is high."
            safety_adjustment = "INCREASE safety margins by 50%. Be extra cautious."
        elif self.weather == "rainy":
            weather_context = "Rain. Road friction is low. Braking distance is increased."
            safety_adjustment = "INCREASE safety margins by 30%. Avoid sudden maneuvers."
        elif self.weather == "sunny":
            weather_context = "Sunny. Normal road conditions."
            safety_adjustment = "Standard safety margins apply."
        elif self.weather == "windy":
            weather_context = "Windy. Vehicle stability may be affected."
            safety_adjustment = "INCREASE safety margins by 20%. Avoid high speeds."
        elif self.weather == "snowy":
            weather_context = "Snowy. Road friction is very low."
            safety_adjustment = "INCREASE safety margins by 50%. Be extremely cautious."

        user_prompt = f"""
        You, the 'ego' car, are now driving a car on a highway. You have already drive for {self.sce.frame} seconds.
        The decision you made LAST time step was `{last_step_action}`. Your explanation was `{last_step_explanation}`. 
        Here is the current scenario: \n ```json\n{self.sce.export2json()}\n```\n. 

        ========================================
        ENVIRONMENTAL CONTEXT:
        Weather Condition: {self.weather.upper()}
        Condition Details: {weather_context}
        Safety Adjustment Required: {safety_adjustment}
        ========================================

        DRIVING STYLE PREFERENCE (Secondary to Safety):
        "{self.instruction}"

        IMPORTANT: The driving style preference is a SUGGESTION for how to drive WHEN IT IS SAFE to do so. 
        It does NOT override safety rules. If the style conflicts with safety, ALWAYS choose safety.
        For example:
        - "Drive like Vin Diesel" means be confident and efficient with lane changes, BUT only when verified safe
        - "Drive like a grandma" means be extra cautious and slow, prioritizing comfort over speed
        - In ALL cases, collision avoidance is the top priority
        ========================================

        Please make decision for the `ego` car using the following MANDATORY steps:

        Step 1: Analyze the current state and weather conditions
        Step 2: Use Get_Available_Actions to know what actions are possible
        Step 3: For EACH action you are considering:
                a) Use Get_Lane_Involved_Car to find ALL vehicles in the lane affected by that action
                b) For EACH vehicle found, use the appropriate safety tool (Is_Acceleration_Conflict_With_Car, etc.)
                c) Only if ALL safety checks pass, the action is safe
        Step 4: Choose an action that is SAFE first, and matches the driving style second
        Step 5: Output your decision with clear explanation

        CRITICAL REMINDER: You CANNOT assume there are no vehicles. You MUST use Get_Lane_Involved_Car first. 
        If you skip this step and cause a collision, you have failed your primary objective.

        Output format:
        Final Answer: 
            "decision":{{"ego car's decision..."}},
            "explanations":{{"your explanation..."}}
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

                try:
                    response = self.client.chat.completions.create(**api_params)
                except Exception as api_error:
                    error_msg = str(api_error)

                    if "tool_use_failed" in error_msg or "validation failed" in error_msg:
                        print(
                            f"[yellow]Tool call validation failed. Model may not support tool calling properly.[/yellow]")
                        print(f"[yellow]Error: {error_msg}[/yellow]")
                        print(
                            f"[red]RECOMMENDATION: Switch to a model with better tool support (e.g., gpt-4o, llama-3.1-70b-versatile)[/red]")

                        fallback_message = messages[-1]["content"] if messages else user_prompt
                        fallback_message += "\n\nIMPORTANT: Your model does not support tool calling. Please provide your decision directly in the format:\nFinal Answer:\n\"decision\": {{action}},\n\"explanations\": {{explanation}}"

                        messages[-1] = {"role": "user", "content": fallback_message}

                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=0.0
                        )
                    else:
                        raise

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
                        raw_func_name = tool_call.function.name
                        func_name = raw_func_name.split('<|')[0]
                        func_name = func_name.split('|')[0]
                        func_name = func_name.strip()

                        if raw_func_name != func_name:
                            print(f"[yellow]Sanitized tool name: '{raw_func_name}' -> '{func_name}'[/yellow]")

                        func_args = json.loads(tool_call.function.arguments)

                        self.ch.log_tool_call(func_name, str(func_args))

                        print(f"\n[DEBUG] >>> TOOL CALL: {func_name}")
                        print(f"[DEBUG] >>> Arguments: {func_args}")

                        if func_name in self.tool_registry:
                            tool_func = self.tool_registry[func_name]
                            try:
                                arg_val = func_args.get("query")
                                if arg_val is None and len(func_args) > 0:
                                    arg_val = list(func_args.values())[0]

                                tool_result = tool_func(str(arg_val))
                            except Exception as e:
                                tool_result = f"Error executing tool: {e}"
                        else:
                            possible_matches = [k for k in self.tool_registry.keys() if
                                                k.lower() in func_name.lower() or func_name.lower() in k.lower()]
                            if possible_matches:
                                matched_name = possible_matches[0]
                                print(f"[yellow]Fuzzy matched '{func_name}' to '{matched_name}'[/yellow]")
                                tool_func = self.tool_registry[matched_name]
                                func_name = matched_name
                                try:
                                    arg_val = func_args.get("query")
                                    if arg_val is None and len(func_args) > 0:
                                        arg_val = list(func_args.values())[0]
                                    tool_result = tool_func(str(arg_val))
                                except Exception as e:
                                    tool_result = f"Error executing tool: {e}"
                            else:
                                tool_result = f"Error: Tool '{func_name}' not found. Available tools: {list(self.tool_registry.keys())}"

                        print(f"[DEBUG] >>> TOOL RESULT: {tool_result[:200]}..." if len(
                            str(tool_result)) > 200 else f"[DEBUG] >>> TOOL RESULT: {tool_result}")
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

        print('\n[cyan]Final decision reached.[/cyan]')
        if self.ch.memory:
            print(f"\n[DEBUG] === FINAL DECISION ===")
            final_output = self.ch.memory[-1]
            print(final_output)
            print(f"[DEBUG] Raw final answer length: {len(str(final_output))}")
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