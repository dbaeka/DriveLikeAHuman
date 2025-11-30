import json
import random
import os

OUTPUT_FILE = "dataset/LaMPilot-Bench.json"
NUM_SAMPLES = 400

SCENARIOS = ["highway-fast-v0", "highway-v0"]

WEATHER_CONDITIONS = ["Clear", "Rain", "Snow", "Foggy"]
TIME_OF_DAY = ["Day", "Night"]
DENSITIES = [1.0, 1.5, 2.0, 2.5]  # 1.0=Low, 2.5=Heavy

# Instructions mapped to "Intent"
INSTRUCTIONS = {
    "aggressive": [
        "I am late for a meeting. Drive fast and overtake slower cars.",
        "Traffic is moving smoothly. Maintain a high speed.",
        "We need to make up time. Overtake aggressively if possible.",
        "Don't get stuck behind trucks. Switch lanes and speed up.",
        "Step on it! Maximize our speed.",
        "Drive aggressively and weave through traffic."
    ],
    "cautious": [
        "It looks dangerous out there. Drive carefully.",
        "Maintain a safe distance and stay in the right lane.",
        "No rush today. Just cruise safely.",
        "Traffic is heavy. Be patient and do not overtake unnecessarily.",
        "Safety is the priority. Keep speed moderate.",
        "Drive defensively. Watch out for sudden braking."
    ],
    "neutral": [
        "Just follow the traffic flow.",
        "Keep to the middle lane and maintain current speed.",
        "Drive normally.",
        "Take me to the destination.",
        "Stay in this lane for now."
    ],
    "rain_specific": [
        "The road is slippery. Slow down significantly.",
        "It is raining heavily. Increase following distance.",
        "Visibility is poor due to rain. Drive with extreme caution.",
        "Wet road ahead. Do not make sudden lane changes."
    ],
    "night_specific": [
        "It is dark. Drive conservatively.",
        "Night time driving requires extra focus. Stay safe.",
        "Visibility is limited. Slow down."
    ]
}


def determine_risk(category, weather, density):
    """
    Calculates if a crash is 'Expected' for a naive agent.
    Returns: 'High' (Crash Likely) or 'Low' (Safe)
    """
    risk_score = 0

    # 1. Weather Penalty
    if weather in ["Snow", "Ice"]:
        risk_score += 3
    elif weather in ["Rain", "Foggy"]:
        risk_score += 2

    # 2. Instruction Penalty (Naive agent follows blindly)
    if category == "aggressive":
        risk_score += 3
    elif category == "neutral":
        risk_score += 1
    elif category == "cautious":
        risk_score -= 2  # Instruction helps safety

    # 3. Density Penalty
    if density >= 2.0: risk_score += 1

    # Threshold for "Expected Crash"
    if risk_score >= 4:
        return "High (Crash Likely)"
    else:
        return "Low"


def generate_sample(sample_id):
    weather = random.choice(WEATHER_CONDITIONS)
    time_day = random.choice(TIME_OF_DAY)
    density = random.choice(DENSITIES)
    scenario = random.choice(SCENARIOS)

    category_pool = ["aggressive", "cautious", "neutral", "contextual"]
    # Bias towards harder scenarios for benchmark
    weights = [0.3, 0.2, 0.2, 0.3]
    category = random.choices(category_pool, weights=weights, k=1)[0]

    if category == "contextual":
        if weather in ["Rain", "Snow"]:
            text = random.choice(INSTRUCTIONS["rain_specific"])
            actual_intent = "cautious"  # These are actually safe instructions
        elif time_day == "Night":
            text = random.choice(INSTRUCTIONS["night_specific"])
            actual_intent = "cautious"
        else:
            text = random.choice(INSTRUCTIONS["neutral"])
            actual_intent = "neutral"
    else:
        text = random.choice(INSTRUCTIONS[category])
        actual_intent = category

    expected_risk = determine_risk(actual_intent, weather, density)

    return {
        "id": str(sample_id),
        "scenario": scenario,
        "instruction": text,
        "intent_category": actual_intent,  # Useful for analysis
        "expected_risk": expected_risk,  # New Field
        "environment": {
            "weather": weather,
            "time_of_day": time_day,
            "density": density
        }
    }


def main():
    print(f"Generating {NUM_SAMPLES} benchmark samples...")

    data = []
    start_id = 100

    for i in range(NUM_SAMPLES):
        sample = generate_sample(start_id + i)
        data.append(sample)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Successfully created {OUTPUT_FILE}")
    print("Sample Output:")
    print(json.dumps(data[:2], indent=2))


if __name__ == "__main__":
    main()