import gymnasium as gym
from lmp_driver.vehicle import PhysicsVehicle


def make_lmp_driver_env(env_id, density=1.0, time_of_day="Day"):
    """
    Creates the environment.
    - density: Multiplier for traffic count.
    - time_of_day: If 'Night', reduces sensor range (visible vehicles).
    """

    # Determine Visibility based on Time
    # Day = See 15 cars. Night = See only 5 closest cars.
    visible_vehicles = 15
    if time_of_day.lower() == "night":
        visible_vehicles = 5

    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": visible_vehicles,
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