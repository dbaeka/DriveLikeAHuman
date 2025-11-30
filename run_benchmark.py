import datetime
import json
import os
import sys
import time

import numpy as np
from gymnasium.wrappers import RecordVideo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lampilot.envs.adapters import make_lampilot_env
from lampilot.primitives import LaMPilotPrimitives
from lampilot.agent import LLMAgent


def log_decision_cycle(command, context, lmp_code, filename="talk2drive_log.json"):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "command": command,
        "context": context,  # Weather, Time, Density
        "lmp": lmp_code  # The generated Python policy
    }
    with open(filename, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def save_evaluation_results(results, filename="results/benchmark_report.json"):
    """Saves the final metrics to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    total = len(results)
    crashes = sum(1 for r in results if r['crashed'])
    successes = total - crashes
    if total > 0:
        raw_avg = np.mean([r['avg_speed'] for r in results])
        avg_speed = float(raw_avg)
    else:
        avg_speed = 0.0

    total_dist = float(sum(r.get('distance', 0) for r in results))

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_scenarios": total,
        "success_rate": f"{(successes / total) * 100:.1f}%" if total > 0 else "0%",
        "collision_rate": f"{(crashes / total) * 100:.1f}%" if total > 0 else "0%",
        "average_speed_mps": round(avg_speed, 2),
        "distance_covered_m": round(total_dist, 2),
        "details": results
    }

    with open(filename, "w") as f:
        json.dump(summary, f, indent=4)

    return summary


def run_single_scenario(scenario_data, video_folder):
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

    env.unwrapped.render_mode = "rgb_array"

    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=f"scenario_{scenario_id}",
        disable_logger=True
    )

    primitives = LaMPilotPrimitives(env)
    try:
        agent = LLMAgent(model_name="openai/gpt-oss-20b")
    except ValueError as e:
        print(f"    ❌ Setup Error: {e}")
        return None

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
        return {"id": scenario_id, "crashed": True, "error": "Compilation Failed", "avg_speed": 0}

    print(f"    🎥 Recording to {video_folder}/scenario_{scenario_id}-episode-0.mp4")
    obs, info = env.reset()

    # APPLY PHYSICS NOW (After the car is spawned)
    try:
        ego_vehicle = env.unwrapped.vehicle
        if hasattr(ego_vehicle, 'set_weather_friction'):
            ego_vehicle.set_weather_friction(env_params['weather'])
            print(f"    🌧️ Physics Applied: {env_params['weather']}")
    except AttributeError:
        pass

    done = False
    truncated = False
    step_count = 0
    speeds = []
    crashed = False

    while not (done or truncated):
        primitives.update(obs)

        try:
            policy_function(primitives)
        except Exception as e:
            print(f"    ⚠️ Runtime Error: {e}")
            break

        obs, reward, done, truncated, info = env.step(primitives.action)

        # Track Metrics
        speed = primitives.get_ego_speed() * 30  # Approx m/s
        speeds.append(speed)

        if info.get('crashed', False):
            crashed = True
            print("    💥 CRASH DETECTED!")

        step_count += 1
        # Stop after 20 seconds (15 FPS * 20 = 300 steps)
        if step_count > 300:
            break

    env.close()

    avg_speed = np.mean(speeds) if speeds else 0
    distance = avg_speed * (step_count / 15.0)  # approx

    result = {
        "id": scenario_id,
        "instruction": instruction,
        "crashed": crashed,
        "success": not crashed,  # Success defined as "Did not crash"
        "avg_speed": round(float(avg_speed), 2),
        "distance": round(float(distance), 2),
        "steps": step_count,
        "weather": env_params['weather']
    }

    status_icon = "❌" if crashed else "✅"
    print(f"    {status_icon} Result: Crashed={crashed}, Speed={avg_speed:.1f} m/s")

    return result


def main():
    dataset_path = os.path.join("dataset", "LaMPilot-Bench.json")
    video_folder = os.path.join("results", "videos")

    if not os.path.exists(dataset_path):
        print("Dataset not found.")
        return

    os.makedirs(video_folder, exist_ok=True)

    with open(dataset_path, 'r') as f:
        scenarios = json.load(f)

    results = []

    for scenario in scenarios:
        res = run_single_scenario(scenario, video_folder)
        if res:
            results.append(res)
        time.sleep(1)

    summary = save_evaluation_results(results)

    print("\n" + "=" * 40)
    print("      BENCHMARK FINAL REPORT      ")
    print("=" * 40)
    print(f"Total Scenarios: {summary['total_scenarios']}")
    print(f"Success Rate:    {summary['success_rate']}")
    print(f"Collision Rate:  {summary['collision_rate']}")
    print(f"Avg Speed:       {summary['average_speed_mps']} m/s")
    print(f"Distance Covered:  {summary['distance_covered_m']} m")
    print("=" * 40)
    print(f"Detailed report saved to: results/benchmark_report.json")


if __name__ == "__main__":
    main()
