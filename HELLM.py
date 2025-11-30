import json
import os
import highway_env
import gymnasium as gym
import numpy as np
import yaml
from dotenv import load_dotenv
from gymnasium.wrappers import RecordVideo
from openai import OpenAI

from LLMDriver.customTools import (
    getAvailableActions, getAvailableLanes, getLaneInvolvedCar,
    isChangeLaneConflictWithCar, isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar, isDecelerationSafe, isActionSafe,
)
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.outputAgent import OutputParser
from scenario.scenario import Scenario

load_dotenv()

CONTROL_FILE = 'mission_control.json'


def get_dynamic_config(current_weather, current_instruction):
    """
    Smart Config Loader:
    1. Handles invalid JSON (User is typing)
    2. Checks for CONFIRM_UPDATE flag (User is drafting)
    """
    if not os.path.exists(CONTROL_FILE):
        # Initialize with CONFIRM_UPDATE = True so it works out of the box
        with open(CONTROL_FILE, 'w') as f:
            json.dump({
                "weather": current_weather,
                "instruction": current_instruction,
                "CONFIRM_UPDATE": True
            }, f, indent=4)
        return current_weather, current_instruction, False

    try:
        with open(CONTROL_FILE, 'r') as f:
            data = json.load(f)

        # MARKER CHECK:
        # If the user sets this to False, we ignore changes (Draft Mode)
        if not data.get("CONFIRM_UPDATE", False):
            return current_weather, current_instruction, False

        new_weather = data.get("weather", current_weather)
        new_instruction = data.get("instruction", current_instruction)

        # Only trigger update if values actually changed
        has_changed = (new_weather != current_weather) or (new_instruction != current_instruction)

        return new_weather, new_instruction, has_changed

    except json.JSONDecodeError:
        # User is likely typing (e.g., missing comma/bracket)
        # We silently ignore this and keep running with old config
        return current_weather, current_instruction, False

    except Exception as e:
        print(f"[red]Control File Error: {e}[/red]")
        return current_weather, current_instruction, False


def create_llm_client(config):
    """
    Factory function to create the appropriate LLM client based on configuration.

    Args:
        config: Dictionary with LLM provider configuration

    Returns:
        tuple: (client, model_name)
    """
    provider = config.get('LLM_PROVIDER', 'groq').lower()

    if provider == 'groq':
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv('GROQ_KEY', '')
        )
        model_name = config.get('GROQ_MODEL', 'openai/gpt-oss-20b')
        print(f"[bold cyan]Using Groq with model: {model_name}[/bold cyan]")

    elif provider == 'openai':
        client = OpenAI(
            api_key=os.getenv('OPENAI_KEY', '')
        )
        model_name = config.get('OPENAI_MODEL', 'gpt-4o')
        print(f"[bold cyan]Using OpenAI with model: {model_name}[/bold cyan]")

    elif provider == 'ollama':
        client = OpenAI(
            base_url=config.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1'),
            api_key='ollama'  # Ollama doesn't require a real API key
        )
        model_name = config.get('OLLAMA_MODEL', 'llama3.1:8b')
        print(f"[bold cyan]Using Ollama with model: {model_name}[/bold cyan]")

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Choose from: groq, openai, ollama")

    return client, model_name


def apply_weather_physics(env, weather):
    obs_noise = 0.0
    if weather == "foggy":
        obs_noise = 2.5
    elif weather == "rainy":
        obs_noise = 1.0

    try:
        env.unwrapped.observation_type.observation_noise = obs_noise
    except AttributeError:
        pass


with open('config.yaml') as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)
client, MODEL_NAME = create_llm_client(CONFIG)

current_weather = os.getenv('WEATHER_CONDITION', 'sunny')
current_instruction = os.getenv('MISSION_INSTRUCTION', 'Drive safely')

vehicleCount = CONFIG.get('SIMULATION', {}).get('VEHICLE_COUNT', 15)

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": vehicleCount,
        "see_behind": True,
        "observation_noise": 0.0,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": np.linspace(0, 32, 9),
    },
    "duration": CONFIG.get('SIMULATION', {}).get('DURATION', 100),
    "vehicles_density": CONFIG.get('SIMULATION', {}).get('VEHICLES_DENSITY', 2),
    "render_agent": True,
    "show_trajectories": True,
}

env = gym.make('highway-v0', render_mode="rgb_array")
env.unwrapped.configure(config)
env = RecordVideo(env, './results-video', name_prefix="highwayv0", disable_logger=True)
obs, info = env.reset()

if not os.path.exists('results-db/'):
    os.mkdir('results-db')
sce = Scenario(vehicleCount, f"results-db/highwayv0.db")

toolModels = [
    getAvailableActions(env), getAvailableLanes(sce), getLaneInvolvedCar(sce),
    isChangeLaneConflictWithCar(sce), isAccelerationConflictWithCar(sce),
    isKeepSpeedConflictWithCar(sce), isDecelerationSafe(sce), isActionSafe(),
]

DA = DriverAgent(
    client=client,
    toolModels=toolModels,
    sce=sce,
    model_name=MODEL_NAME,
    verbose=True,
    is_ollama=(CONFIG.get('LLM_PROVIDER') == 'ollama'),
    weather=current_weather,
    instruction=current_instruction
)
outputParser = OutputParser(sce=sce, client=client, model_name=MODEL_NAME)

# Main Loop
output = None
done = truncated = False
frame = 0

print(f"[bold green]Simulation Running.[/bold green]")
print(f"Edit [bold]mission_control.json[/bold] and set [bold]\"CONFIRM_UPDATE\": true[/bold] to apply changes.")

try:
    while not (done or truncated):
        # CHECK FOR UPDATES
        new_weather, new_instruction, changed = get_dynamic_config(current_weather, current_instruction)

        if changed:
            print(f"\n[bold yellow]⚠ HOT RELOAD DETECTED (Frame {frame})[/bold yellow]")
            print(f"Instruction: {new_instruction}")
            print(f"Weather: {new_weather}")

            DA.weather = new_weather
            DA.instruction = new_instruction
            apply_weather_physics(env, new_weather)

            current_weather = new_weather
            current_instruction = new_instruction

        sce.upateVehicles(obs, frame)
        DA.agentRun(output)
        da_output = DA.exportThoughts()
        output = outputParser.agentRun(da_output)

        action_id = output.get("action_id", 1)
        if not isinstance(action_id, int):
            try:
                action_id = int(action_id)
            except:
                action_id = 1

        obs, reward, done, truncated, info = env.step(action_id)
        env.render()

        print(f"Frame {frame}: {output.get('action_name', 'Unknown')}")
        frame += 1

finally:
    env.close()
    print("[bold red]Simulation Ended[/bold red]")
