import gymnasium as gym
from lampilot.vehicle import PhysicsVehicle


def make_lampilot_env(env_id, density=1.0):
    """
    Creates the environment with custom PhysicsVehicle.
    """
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "duration": 40,
        "vehicles_count": int(20 * density),
        "controlled_vehicles": 1,
        "initial_vehicle_count": 10
    }

    env = gym.make(env_id, render_mode="human", config=config)

    env.unwrapped.vehicle_class = PhysicsVehicle

    return env