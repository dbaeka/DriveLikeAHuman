class LaMPilotPrimitives:
    """
    The 'Smart' Body of the agent. 
    Includes internal safety checks and priority logic to fix 'dumb' LLM mistakes.
    """

    def __init__(self, env):
        self.env = env
        self.obs = None

        # Action Map
        # 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
        self.ACTIONS = {
            "LANE_LEFT": 0,
            "IDLE": 1,
            "LANE_RIGHT": 2,
            "FASTER": 3,
            "SLOWER": 4
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

    # ==========================================
    # PERCEPTION API (Read-Only)
    # ==========================================

    def get_ego_speed(self):
        """Returns normalized speed."""
        return self.obs[0, 3]

    def is_lane_free(self, direction):
        """
        Robust check for lane availability.
        """
        neighbors = self._get_neighbors()
        lat_offset = 0.25 if direction == "right" else -0.25
        ego_y = self.obs[0, 2]
        target_y = ego_y + lat_offset

        # 1. Boundary Check (Don't drive off road)
        # Assuming 4 lanes (0, 0.25, 0.5, 0.75). Adjust if needed.
        if target_y < -0.05 or target_y > 0.8:
            return False

        # 2. Car Check
        for car in neighbors:
            car_y = car[2]
            car_x = car[1]

            # If car is in target lane (roughly)
            if abs(car_y - target_y) < 0.1:
                # Collision Zone: Don't cut off someone close (-0.3 to +0.3)
                if -0.3 < car_x < 0.3:
                    return False
        return True

    def get_distance_to_lead(self):
        """Returns distance to lead car. Returns 1.0 if clear."""
        neighbors = self._get_neighbors()
        min_dist = 1.0
        ego_y = self.obs[0, 2]

        for car in neighbors:
            if abs(car[2] - ego_y) < 0.1:  # Same lane
                if car[1] > 0:  # In front
                    if car[1] < min_dist:
                        min_dist = car[1]
        return min_dist

    # ==========================================
    # ACTION API (Write - With Safety Logic)
    # ==========================================

    def change_lane_left(self):
        # GUARD 1: Priority Check
        # If we already decided to change lanes right, don't confuse the car.
        if self._action_priority >= 2: return

        # GUARD 2: Physical Reality Check
        if self.is_lane_free("left"):
            self.action = self.ACTIONS["LANE_LEFT"]
            self._action_priority = 2  # Lock this action
            print("Action: Changing Left")
        else:
            print("⚠️ Safety Guard: Ignored unsafe Left Turn command.")

    def change_lane_right(self):
        if self._action_priority >= 2: return

        if self.is_lane_free("right"):
            self.action = self.ACTIONS["LANE_RIGHT"]
            self._action_priority = 2
            print("Action: Changing Right")
        else:
            print("⚠️ Safety Guard: Ignored unsafe Right Turn command.")

    def speed_up(self):
        # GUARD 1: Don't overwrite a turn!
        # If priority is 2 (Turning), ignore speed changes.
        if self._action_priority >= 2:
            return

        # GUARD 2: Rear-End Collision Prevention
        dist = self.get_distance_to_lead()
        if dist < 0.15:  # Too close!
            print("⚠️ Safety Guard: Blocking acceleration (Too close). Auto-Braking.")
            self.slow_down()  # Override to brake
            return

        self.action = self.ACTIONS["FASTER"]
        self._action_priority = 1

    def slow_down(self):
        # Slowing down is usually safe, but check priority
        # We allow braking to override a turn if it's an emergency? 
        # For now, let's say turning is higher priority unless we add emergency logic.
        if self._action_priority >= 2: return

        self.action = self.ACTIONS["SLOWER"]
        self._action_priority = 1

    def keep_speed(self):
        if self._action_priority > 0: return
        self.action = self.ACTIONS["IDLE"]
