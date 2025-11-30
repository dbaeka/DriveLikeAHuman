import gymnasium as gym
import highway_env


def make_lampilot_env(env_id="highway-fast-v0", render_mode="rgb_array"):
    """
    Creates a highway-env instance configured for LaMPilot.
    We need 'Kinematics' observation to easily parse vehicle positions.
    """
    env = gym.make(env_id, render_mode=render_mode)

    # Configure the environment to return raw vehicle coordinates
    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "normalize": True
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 40  # Seconds
    })

    return env