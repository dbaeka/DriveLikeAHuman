import json
import os
import sys
import time

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lampilot.envs.adapters import make_lampilot_env
from lampilot.primitives import LaMPilotPrimitives
from lampilot.agent import LLMAgent


def run_single_scenario(scenario_data):
    instruction = scenario_data['instruction']
    env_id = scenario_data['scenario']

    print(f"\n>>> SCENARIO {scenario_data['id']}")

    # 1. Init
    env = make_lampilot_env(env_id, render_mode="human")
    primitives = LaMPilotPrimitives(env)

    try:
        agent = LLMAgent(model_name="openai/gpt-oss-20b")
    except ValueError as e:
        print(e)
        return False

    # 2. Generate Code (Real API Call)
    print("   Requesting policy from LLM...")
    policy_code = agent.generate_policy(instruction)

    print(f"\n--- LLM Generated Code ---\n{policy_code}\n--------------------------")

    # 3. Compile Code
    exec_scope = {}
    try:
        # This is where we catch syntax errors from the LLM
        exec(policy_code, {}, exec_scope)
        if 'policy' not in exec_scope:
            raise ValueError("LLM did not define 'policy(api)' function.")
        policy_function = exec_scope['policy']
    except Exception as e:
        print(f"❌ Failed to compile LLM code: {e}")
        env.close()
        return False

    # 4. Simulation
    obs, info = env.reset()
    done = False
    truncated = False
    step_count = 0

    while not (done or truncated):
        primitives.update(obs)

        try:
            policy_function(primitives)
        except Exception as e:
            print(f"Runtime Error inside policy: {e}")
            break

        obs, reward, done, truncated, info = env.step(primitives.action)
        env.render()

        step_count += 1
        if step_count > 300: break

    env.close()
    return True


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "dataset", "LaMPilot-Bench.json")

    if not os.path.exists(dataset_path):
        print("Dataset not found. Please create dataset/LaMPilot-Bench.json first.")
        return

    with open(dataset_path, 'r') as f:
        scenarios = json.load(f)

    for scenario in scenarios:
        run_single_scenario(scenario)
        time.sleep(1)


if __name__ == "__main__":
    main()
