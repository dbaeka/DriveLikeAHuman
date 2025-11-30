import datetime
import json
import os
import sys
import time

import numpy as np
from gymnasium.wrappers import RecordVideo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmp_driver.envs.adapters import make_lmp_driver_env
from lmp_driver.primitives import LLMDriverPrimitives
from lmp_driver.agent import LLMAgent

SAVE_INTERVAL = 5  # Save results every 5 scenarios
REPORT_FILE = "results/benchmark_report.json"


def log_decision_cycle(command, context, lmp_code, filename="talk2drive_log.json"):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "command": command,
        "context": context,  # Weather, Time, Density
        "lmp": lmp_code  # The generated Python policy
    }
    with open(filename, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def save_evaluation_results(results, filename=REPORT_FILE):
    """Saves the final metrics to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    total = len(results)
    crashes = sum(1 for r in results if r['crashed'])
    successes = total - crashes

    high_risk_runs = [r for r in results if "High" in r.get('expected_risk', '')]
    high_risk_total = len(high_risk_runs)
    high_risk_survived = sum(1 for r in high_risk_runs if not r['crashed'])

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
        "high_risk_scenarios": high_risk_total,
        "high_risk_survival_rate": f"{(high_risk_survived / high_risk_total) * 100:.1f}%" if high_risk_total > 0 else "N/A",
        "collision_rate": f"{(crashes / total) * 100:.1f}%" if total > 0 else "0%",
        "average_speed_mps": round(avg_speed, 2),
        "distance_covered_m": round(total_dist, 2),
        "details": results
    }

    with open(filename, "w") as f:
        json.dump(summary, f, indent=4)
        print(f"    💾 Checkpoint saved to {filename}")

    return summary


def run_single_scenario(scenario_data, video_folder):
    scenario_id = scenario_data['id']
    instruction = scenario_data['instruction']

    expected_risk = scenario_data.get('expected_risk', 'Unknown')

    env_params = scenario_data.get('environment', {
        "weather": "Clear",
        "time_of_day": "Day",
        "density": 1.0
    })

    print(f"\n>>> RUNNING SCENARIO {scenario_id} [{expected_risk} Risk]")
    print(f"    Instruction: {instruction}")
    print(f"    Context: {env_params}")

    env = make_lmp_driver_env(
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

    primitives = LLMDriverPrimitives(env)
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
        return {"id": scenario_id, "crashed": True, "error": "Compilation Failed", "avg_speed": 0, "distance": 0,
                "steps": 0, "weather": env_params['weather']}

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
        "weather": env_params['weather'],
        "expected_risk": expected_risk,
        "crashed": crashed,
        "success": not crashed,
        "steps": step_count,
        "avg_speed": round(float(avg_speed), 2),
        "distance": round(float(distance), 2)
    }

    status_icon = "❌" if crashed else "✅"
    risk_icon = "⚠️" if "High" in expected_risk else "safe"
    print(f"    {status_icon} Result: Crashed={crashed} | Risk Level: {risk_icon} {expected_risk}")

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
    completed_ids = set()

    # Check if a report already exists
    if os.path.exists(REPORT_FILE):
        try:
            with open(REPORT_FILE, 'r') as f:
                data = json.load(f)
                # Load previous results into memory
                if 'details' in data:
                    results = data['details']
                    completed_ids = {r['id'] for r in results}
                    print(f"🔄 Resuming... Found {len(results)} completed scenarios.")
        except json.JSONDecodeError:
            print("⚠️ Warning: Existing report file was corrupted. Starting fresh.")

    # Filter out scenarios that are already done
    scenarios_to_run = [s for s in scenarios if s['id'] not in completed_ids]

    if not scenarios_to_run:
        print("✅ All scenarios are already completed!")
        return

    print(f"🚀 Starting benchmark for {len(scenarios_to_run)} remaining scenarios...")

    for i, scenario in enumerate(scenarios_to_run):
        res = run_single_scenario(scenario, video_folder)
        if res:
            results.append(res)

        # SAVE EVERY 'SAVE_INTERVAL' SCENARIOS
        if (i + 1) % SAVE_INTERVAL == 0:
            save_evaluation_results(results, REPORT_FILE)

        time.sleep(1)

    summary = save_evaluation_results(results, REPORT_FILE)

    print("\n" + "=" * 40)
    print("      BENCHMARK FINAL REPORT      ")
    print("=" * 40)
    print(f"Total Scenarios: {summary['total_scenarios']}")
    print(f"Success Rate:    {summary['success_rate']}")
    print(f"Collision Rate:  {summary['collision_rate']}")
    print(f"Avg Speed:       {summary['average_speed_mps']} m/s")
    print(f"Distance Covered:  {summary['distance_covered_m']} m")
    print("=" * 40)
    print(f"Detailed report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()