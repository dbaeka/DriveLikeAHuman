import numpy as np
from highway_env.vehicle.controller import ControlledVehicle


class PhysicsVehicle(ControlledVehicle):
    """
    A Vehicle that reacts to 'Friction' and 'Weather'.
    If the friction is low, the vehicle cannot accelerate or turn as sharply.
    """

    def __init__(self, road, position, heading=0, speed=0, target_lane_index=None, target_speed=None, route=None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.friction = 1.0  # Default: Dry Road (1.0)

    def set_weather_friction(self, weather):
        """
        Adjusts grip based on weather conditions.
        """
        weather = weather.lower()
        if "rain" in weather:
            self.friction = 0.6  # 40% loss of grip
        elif "snow" in weather or "ice" in weather:
            self.friction = 0.3  # 70% loss of grip (Dangerous!)
        else:
            self.friction = 1.0

    def act(self, action=None):
        """
        Override the control loop to apply physical limits.
        """
        super().act(action)

        # If friction is low, we cannot accelerate/brake effectively
        if self.action is not None:
            # Scale the effectiveness of the throttle/brake by friction
            # In highway-env, action['acceleration'] is usually clamped to ranges.
            # We forcibly reduce that range.

            current_accel = self.action.get('acceleration', 0)
            max_grip_accel = 5.0 * self.friction  # Assuming 5 m/s^2 is max dry accel

            # Clip acceleration to what the tires can actually handle
            self.action['acceleration'] = np.clip(
                current_accel,
                -max_grip_accel,  # Braking limit
                max_grip_accel  # Acceleration limit
            )

            # Reduce steering responsiveness
            if 'steering' in self.action:
                self.action['steering'] *= self.friction
