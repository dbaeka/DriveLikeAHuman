import os

import gymnasium as gym
import numpy as np
import highway_env
import yaml
from gymnasium.wrappers import RecordVideo
from openai import OpenAI

# Custom Tools
from LLMDriver.customTools import (
    getAvailableActions,
    getAvailableLanes,
    getLaneInvolvedCar,
    isChangeLaneConflictWithCar,
    isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar,
    isDecelerationSafe,
    isActionSafe,
)
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.outputAgent import OutputParser
from scenario.scenario import Scenario

# 1. Configuration
with open('config.yaml') as f:
    OPENAI_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

# 2. Initialize Client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=OPENAI_CONFIG['GROQ_KEY']
)
MODEL_NAME = "openai/gpt-oss-20b"

# 3. Environment Config
vehicleCount = 15
config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False,
        "vehicles_count": vehicleCount,
        "see_behind": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": np.linspace(0, 32, 9),
    },
    "duration": 40,
    "vehicles_density": 2,
    "show_trajectories": True,
    "render_agent": True,
}

# 4. Setup Environment
env = gym.make('highway-v0', render_mode="rgb_array")

# --- FIX: Use .unwrapped to access custom configure method ---
env.unwrapped.configure(config)
# -------------------------------------------------------------

# Setup Video Recording
if not os.path.exists('./results-video'):
    os.makedirs('./results-video')

env = RecordVideo(
    env,
    './results-video',
    name_prefix="highwayv0",
    disable_logger=True
)

obs, info = env.reset()

# 5. Scenario and Agent Setup
if not os.path.exists('results-db/'):
    os.mkdir('results-db')
database = f"results-db/highwayv0.db"
sce = Scenario(vehicleCount, database)

# Initialize Tools
toolModels = [
    getAvailableActions(env),
    getAvailableLanes(sce),
    getLaneInvolvedCar(sce),
    isChangeLaneConflictWithCar(sce),
    isAccelerationConflictWithCar(sce),
    isKeepSpeedConflictWithCar(sce),
    isDecelerationSafe(sce),
    isActionSafe(),
]

# Initialize Agents
DA = DriverAgent(
    client=client,
    toolModels=toolModels,
    sce=sce,
    model_name=MODEL_NAME,
    verbose=True
)

outputParser = OutputParser(
    sce=sce,
    client=client,
    model_name=MODEL_NAME
)

# 6. Main Loop
output = None
done = truncated = False
frame = 0

print(f"[bold green]Starting Simulation with Model: {MODEL_NAME}[/bold green]")

try:
    while not (done or truncated):
        # Update Scenario
        sce.upateVehicles(obs, frame)

        # Run Driver Agent
        DA.agentRun(output)
        da_output = DA.exportThoughts()

        # Run Output Parser
        output = outputParser.agentRun(da_output)

        # Safe Action Handling
        action_id = output.get("action_id", 1)  # Default to 1 (Idle/Keep Speed) if missing

        # Ensure action is a standard python int for Gym
        if not isinstance(action_id, int):
            try:
                action_id = int(action_id)
            except:
                action_id = 1

        # Step Environment
        obs, reward, done, truncated, info = env.step(action_id)

        # Render
        env.render()

        print(f"Frame {frame}: Action {action_id} | {output.get('action_name', 'Unknown')}")
        frame += 1

finally:
    env.close()
    print("[bold red]Simulation Ended[/bold red]")
