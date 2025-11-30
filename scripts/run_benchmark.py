import datetime
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lampilot.envs.adapters import make_lampilot_env
from lampilot.primitives import LaMPilotPrimitives
from lampilot.agent import LLMAgent


def log_decision_cycle(command, context, lmp_code, filename="talk2drive_log.json"):
    """
    Logs the Command -> Context -> LMP cycle for analysis.
    """
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "command": command,
        "context": context,  # Weather, Time, Density
        "lmp": lmp_code  # The generated Python policy
    }

    with open(filename, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def run_single_scenario(scenario_data):
    scenario_id = scenario_data['id']
    instruction = scenario_data['instruction']

    env_params = scenario_data.get('environment', {
        "weather": "Clear",
        "time_of_day": "Day",
        "density": 1.0
    })

    print(f"\n>>> RUNNING SCENARIO {scenario_id}")
    print(f"    Instruction: {instruction}")
    print(f"    Context: {env_params}")

    env = make_lampilot_env(
        scenario_data['scenario'],
        density=env_params['density'],
        time_of_day=env_params['time_of_day']
    )

    env.reset()
    try:
        ego_vehicle = env.unwrapped.vehicle
        if hasattr(ego_vehicle, 'set_weather_friction'):
            ego_vehicle.set_weather_friction(env_params['weather'])
            print(f"    🌧️ Physics updated: Friction set for {env_params['weather']}")

        if env_params['time_of_day'] == 'Night':
            print("    🌑 Time updated: Visibility reduced (Night Mode)")

    except AttributeError:
        print("    PhysicsVehicle not found. Using standard physics.")

    primitives = LaMPilotPrimitives(env)
    try:
        agent = LLMAgent(model_name="openai/gpt-oss-20b")
    except ValueError as e:
        print(f"    ❌ Setup Error: {e}")
        return False

    print("    Generating Policy...")
    policy_code = agent.generate_policy(instruction, env_params)

    log_decision_cycle(instruction, env_params, policy_code)

    exec_scope = {}
    try:
        exec(policy_code, {}, exec_scope)
        if 'policy' not in exec_scope:
            raise ValueError("LLM response did not contain 'def policy(api):'")
        policy_function = exec_scope['policy']
    except Exception as e:
        print(f"    ❌ Code Compilation Failed: {e}")
        env.close()
        return False

    obs, info = env.reset()
    done = False
    truncated = False
    step_count = 0

    print("    🚗 Starting Drive...")
    while not (done or truncated):
        primitives.update(obs)

        try:
            policy_function(primitives)
        except Exception as e:
            print(f"    ⚠️ Runtime Error: {e}")
            break

        obs, reward, done, truncated, info = env.step(primitives.action)
        env.render()

        step_count += 1
        # Stop after 20 seconds (15 FPS * 20 = 300 steps)
        if step_count > 300:
            break

    env.close()
    print(f"    ✅ Scenario {scenario_id} Finished.")
    return True


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "dataset", "LaMPilot-Bench.json")

    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, 'r') as f:
        scenarios = json.load(f)

    for scenario in scenarios:
        run_single_scenario(scenario)
        time.sleep(1)  # Brief pause between runs


if __name__ == "__main__":
    main()
