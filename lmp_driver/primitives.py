class LLMDriverPrimitives:
    def __init__(self, env):
        self.env = env
        self.obs = None

        self.ACTIONS = {
            "LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4
        }

        self.action = 1
        self._action_priority = 0  # Used to prevent overwriting important moves

    def update(self, obs):
        """Called every step. Resets priority."""
        self.obs = obs
        self.action = self.ACTIONS["IDLE"]
        self._action_priority = 0
        # Priority Levels:
        # 0 = Default (Idle)
        # 1 = Speed Change (Accel/Decel)
        # 2 = Lane Change (High Priority)
        # 3 = Emergency Safety Override (Max Priority)

    def _get_neighbors(self):
        if self.obs is None: return []
        neighbors = self.obs[1:]
        return neighbors[neighbors[:, 0] == 1]

    # --- PERCEPTION ---
    def get_ego_speed(self):
        """Returns normalized speed."""
        return self.obs[0, 3]

    def get_distance_to_lead(self):
        neighbors = self._get_neighbors()
        min_dist = 1.0
        ego_y = self.obs[0, 2]

        for car in neighbors:
            if abs(car[2] - ego_y) < 0.1:  # Same lane
                if car[1] > 0:  # In front
                    if car[1] < min_dist:
                        min_dist = car[1]
        return min_dist

    def get_relative_speed_to_lead(self):
        """
        Returns how much faster we are than the car ahead.
        Positive = We are faster (closing in).
        Negative = They are faster (pulling away).
        """
        neighbors = self._get_neighbors()
        ego_y = self.obs[0, 2]
        ego_vx = self.obs[0, 3]  # Our speed

        for car in neighbors:
            if abs(car[2] - ego_y) < 0.1:  # Same lane
                if car[1] > 0:  # In front
                    # Relative speed = Our Speed - Their Speed
                    # car[3] is their absolute speed in observation
                    return ego_vx - car[3]

        return 0.0  # No car ahead

    def is_lane_free(self, direction):
        # ... (Keep your existing implementation) ...
        neighbors = self._get_neighbors()
        lat_offset = 0.25 if direction == "right" else -0.25
        ego_y = self.obs[0, 2]
        target_y = ego_y + lat_offset

        # 1. Boundary Check (Don't drive off road)
        # Assuming 4 lanes (0, 0.25, 0.5, 0.75). Adjust if needed.
        if target_y < -0.05 or target_y > 0.8:
            return False

        for car in neighbors:
            car_y = car[2]
            car_x = car[1]
            if abs(car_y - target_y) < 0.1:
                # Collision Zone: Don't cut off someone close (-0.3 to +0.3)
                if -0.3 < car_x < 0.3:
                    return False
        return True

    # --- ACTIONS ---

    def change_lane_left(self):
        if self._action_priority >= 2: return
        if self.is_lane_free("left"):
            self.action = self.ACTIONS["LANE_LEFT"]
            self._action_priority = 2
        else:
            print("⚠️ Safety: Left Lane Blocked.")

    def change_lane_right(self):
        if self._action_priority >= 2: return
        if self.is_lane_free("right"):
            self.action = self.ACTIONS["LANE_RIGHT"]
            self._action_priority = 2
        else:
            print("⚠️ Safety: Right Lane Blocked.")

    def speed_up(self):
        if self._action_priority >= 2: return

        dist = self.get_distance_to_lead()
        rel_speed = self.get_relative_speed_to_lead()

        # DYNAMIC SAFETY MARGIN
        # Base safe distance
        safe_dist = 0.15

        # If we are closing in fast, we need MORE space
        if rel_speed > 0:
            # Add padding based on closing speed
            safe_dist += rel_speed * 0.5

            # Retrieve current friction from the vehicle (if available)
        # We access the underlying vehicle class we injected
        current_friction = 1.0
        try:
            current_friction = self.env.unwrapped.vehicle.friction
        except:
            pass

        # If slippery, DOUBLE the required safety distance
        if current_friction < 0.9:
            safe_dist *= 1.5

        if dist < safe_dist:
            print(f"⚠️ Safety: Too close ({dist:.2f} < {safe_dist:.2f}). Braking.")
            self.slow_down()
            return

        self.action = self.ACTIONS["FASTER"]
        self._action_priority = 1

    def slow_down(self):
        if self._action_priority >= 2: return
        self.action = self.ACTIONS["SLOWER"]
        self._action_priority = 1

    def keep_speed(self):
        if self._action_priority > 0: return
        self.action = self.ACTIONS["IDLE"]
