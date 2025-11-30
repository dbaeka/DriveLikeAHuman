import warnings
import highway_env
import gymnasium as gym

# Suppress gym warnings for cleaner output
warnings.filterwarnings("ignore")


# ==========================================
# PART 1: The Functional Primitives (API)
# ==========================================
class LaMPilotPrimitives:
    """
    This class defines the API that the LLM is allowed to use.
    It translates high-level commands (check_left, change_lane)
    into low-level highway-env actions and observations.
    """

    def __init__(self, env):
        self.env = env
        self.obs = None
        # highway-env action map: 0:L, 1:Idle, 2:R, 3:Fast, 4:Slow
        self.next_action = 1
        self.lane_width_proxy = 0.1  # In normalized observation space

    def update(self, obs):
        """Called every timestep to update the car's perception"""
        self.obs = obs
        self.next_action = 1  # Reset to Idle

    def _get_neighbors(self):
        """
        Parses highway-env kinematic observation.
        Row 0 is Ego. Rows 1+ are neighbors.
        Columns: [presence, x, y, vx, vy, cos_h, sin_h]
        """
        if self.obs is None: return []

        # Filter for present vehicles only (column 0 == 1)
        neighbors = self.obs[1:]
        active_neighbors = neighbors[neighbors[:, 0] == 1]
        return active_neighbors

    # --- Perception APIs (The LLM calls these) ---

    def get_ego_speed(self):
        """Returns normalized speed of ego vehicle"""
        # Column 3 is vx
        return self.obs[0, 3]

    def is_lane_available(self, direction):
        """
        Checks if a lane exists and is free of immediate collision.
        direction: 'left' or 'right'
        """
        neighbors = self._get_neighbors()

        # Define search zones (normalized coordinates)
        # y=0 is ego lane. y positive is Left in some configs, Right in others.
        # In highway-v0: y is lateral position.
        # We need to check if there is a car in the target lane window.

        target_y_min = 0.1 if direction == "right" else -0.4
        target_y_max = 0.4 if direction == "right" else -0.1

        for car in neighbors:
            rel_x = car[1]
            rel_y = car[2]

            # If car is in the target lateral zone
            if target_y_min < rel_y < target_y_max:
                # And close longitudinally (danger zone)
                if -0.15 < rel_x < 0.15:
                    return False
        return True

    def get_distance_to_lead_vehicle(self):
        """Returns distance to car directly in front"""
        neighbors = self._get_neighbors()
        min_dist = 1.0  # Max visual range

        for car in neighbors:
            rel_x = car[1]
            rel_y = car[2]

            # If in my lane (small y deviation) and in front (positive x)
            if abs(rel_y) < 0.1 and rel_x > 0:
                if rel_x < min_dist:
                    min_dist = rel_x
        return min_dist

    # --- Action APIs (The LLM calls these) ---

    def lane_change_left(self):
        # Safety Check: simplified
        if self.is_lane_available("left"):
            self.next_action = 0  # LEFT
            return True
        return False

    def lane_change_right(self):
        if self.is_lane_available("right"):
            self.next_action = 2  # RIGHT
            return True
        return False

    def accelerate(self):
        self.next_action = 3  # FASTER

    def decelerate(self):
        self.next_action = 4  # SLOWER

    def keep_speed(self):
        self.next_action = 1  # IDLE


# ==========================================
# PART 2: The LLM Policy Generator (The "Brain")
# ==========================================
class LLMController:
    """
    In a real scenario, this sends a prompt to GPT-4.
    Here, we simulate the LLM returning Python code based on the instruction.
    """

    def __init__(self):
        pass

    def generate_policy(self, instruction):
        print(f"🤖 LLM received instruction: '{instruction}'")
        print("🤖 Generating Python code...")

        # ---------------------------------------------------------
        # MOCK LLM RESPONSES
        # This simulates what GPT-4 would output for these prompts.
        # ---------------------------------------------------------

        if "fast" in instruction or "aggressive" in instruction:
            # Policy: Drive fast, weave through traffic
            return """
def policy(api):
    lead_dist = api.get_distance_to_lead_vehicle()

    # If open road, floor it
    if lead_dist > 0.3:
        api.accelerate()
    # If stuck behind someone
    else:
        # Try passing left
        if api.is_lane_available("left"):
            api.lane_change_left()
        # Try passing right
        elif api.is_lane_available("right"):
            api.lane_change_right()
        else:
            api.decelerate() # Brake if stuck
"""

        elif "safe" in instruction or "careful" in instruction:
            # Policy: Keep distance, don't change lanes often
            return """
def policy(api):
    lead_dist = api.get_distance_to_lead_vehicle()
    speed = api.get_ego_speed()

    # Maintain safe distance (approx 0.2 normalized)
    if lead_dist < 0.2:
        api.decelerate()
    elif speed < 0.25: # Don't go too fast
        api.accelerate()
    else:
        api.keep_speed()
"""

        else:
            # Default: Just cruise
            return """
def policy(api):
    api.keep_speed()
"""


# ==========================================
# PART 3: The Execution Engine
# ==========================================
def run_benchmark(instruction):
    # 1. Setup Environment
    # We use highway-fast-v0 for a dynamic highway scenario
    env = gym.make('highway-fast-v0', render_mode='human')
    # Configure environment to give us Kinematics (coordinates)
    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "normalize": True
        }
    })

    obs, _ = env.reset()

    # 2. Initialize LaMPilot components
    primitives = LaMPilotPrimitives(env)
    llm = LLMController()

    # 3. Get the code from the "LLM"
    code_str = llm.generate_policy(instruction)
    print(f"\n--- Generated Code ---\n{code_str}\n----------------------")

    # 4. Compile the policy
    # We create a local scope to execute the string as a function
    local_scope = {}
    exec(code_str, {}, local_scope)
    policy_func = local_scope['policy']

    # 5. Simulation Loop
    done = False
    truncated = False
    step = 0

    while not (done or truncated):
        # A. Update Perception
        primitives.update(obs)

        # B. Run the LLM-generated Policy
        try:
            policy_func(primitives)
        except Exception as e:
            print(f"Runtime Error in Policy: {e}")

        # C. Execute Action in Simulator
        action = primitives.next_action
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        step += 1

        if step > 200: break  # Limit run duration

    print("Simulation finished.")
    env.close()


# ==========================================
# PART 4: Run It
# ==========================================
if __name__ == "__main__":
    # SCENARIO 1: Aggressive Driving
    print("\n>>> TEST 1: Aggressive Instruction")
    run_benchmark("I am late! Drive fast and weave through traffic.")

    # SCENARIO 2: Safe Driving (Uncomment to run)
    # print("\n>>> TEST 2: Safe Instruction")
    # run_benchmark("Drive carefully and maintain a safe distance.")
