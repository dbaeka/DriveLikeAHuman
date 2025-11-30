from scenario.scenario import Scenario
from typing import Any
import math


def prompts(name, description):
    def decorator(func):
        func.name = name.replace(" ", "_")
        func.description = description
        return func

    return decorator


ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTIONS_DESCRIPTION = {
    0: 'change lane to the left of the current lane',
    1: 'remain in the current lane with current speed',
    2: 'change lane to the right of the current lane',
    3: 'accelerate the vehicle',
    4: 'decelerate the vehicle (can slow to a complete stop)'
}


class getAvailableActions:
    def __init__(self, env: Any) -> None:
        self.env = env

    @prompts(name='Get_Available_Actions',
             description="""Useful before you make decisions, this tool let you know what are your available actions in this situation. The input to this tool should be 'ego'.""")
    def inference(self, input: str) -> str:
        print(f"\n{'=' * 80}")
        print(f"[DEBUG] TOOL CALL: Get_Available_Actions")
        print(f"{'=' * 80}")

        if hasattr(self.env, 'unwrapped'):
            base_env = self.env.unwrapped
        else:
            base_env = self.env

        if hasattr(base_env, 'get_available_actions'):
            availableActions = base_env.get_available_actions()
        else:
            return "Error: Could not retrieve available actions from environment."

        print(f"[DEBUG] Available actions: {availableActions}")
        print(f"[DEBUG] Action mapping: {[ACTIONS_ALL.get(a) for a in availableActions]}")

        # Try to find current lane to detect stuck states
        vehicle = getattr(base_env, 'vehicle', None)
        current_lane_index = None

        if vehicle:
            if hasattr(vehicle, 'lane_index'):
                current_lane_index = vehicle.lane_index
            elif isinstance(vehicle, dict) and 'lane_index' in vehicle:
                current_lane_index = vehicle['lane_index']

        stuck_state = False
        if current_lane_index and isinstance(current_lane_index, tuple):
            lane_idx = current_lane_index[2]
            if lane_idx == 0 and 0 in availableActions:
                stuck_state = True

        outputPrefix = 'You can ONLY use one of the following actions: \n'
        for action in availableActions:
            desc = ACTIONS_DESCRIPTION.get(action, '')
            if action == 0:
                desc += " [CHECK: Is there a lane to the LEFT?]"
            elif action == 2:
                desc += " [CHECK: Is there a lane to the RIGHT?]"

            outputPrefix += ACTIONS_ALL.get(action, 'UNKNOWN') + '--' + desc + '; \n'

        if stuck_state:
            outputPrefix += '\n⚠️  SYSTEM ALERT: STUCK MANEUVER DETECTED. You may be stuck performing a lane change.\n'
            outputPrefix += '   - If you are stopped (speed ~0), you MUST Accelerate (FASTER) to complete the turn.\n'
            outputPrefix += '   - Safety checks may need to be overridden if you are turning AWAY from the obstacle.\n'

        outputPrefix += '\n**MANDATORY SAFETY VERIFICATION FOR ALL ACTIONS:**\n'
        outputPrefix += '\n⚠️  WARNING: You CANNOT assume lanes are empty. You MUST use Get_Lane_Involved_Car first!\n'
        outputPrefix += '⚠️  CRITICAL: Check the CLOSEST vehicle ahead. Distance < 0.2m is DANGER.\n'
        outputPrefix += '⚠️  DIRECTION CHECK: Use `Get_Available_Lanes` to confirm if Left/Right lanes actually exist before choosing LANE_LEFT or LANE_RIGHT.\n\n'

        return outputPrefix


class isActionSafe:
    def __init__(self) -> None:
        pass

    @prompts(name='Decision_making_Instructions',
             description="""This tool gives you a brief intruduction about how to ensure that the action you make is safe. The input to this tool should be a string, which is ONLY the action name.""")
    def inference(self, action: str) -> str:
        return f"""To check action safety you should follow three steps:
        Step 1: Identify the lanes affected by this action. Acceleration, deceleration and idle affect the current lane, while left and right lane changes affect the corresponding lane.
        Step 2:(Optional) Get the vehicles in this lane that you may affect, ONLY when you don't know.
        Step 3: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
        Follow the instructions and remember to use the proper tools mentioned in the tool list once a time.
        """


class getAvailableLanes:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce

    @prompts(name='Get_Available_Lanes',
             description="""useful when you want to know the available lanes of the vehicles. like: I want to know the available lanes of the vehicle `ego`. The input to this tool should be a string, representing the id of the vehicle.""")
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles:
            return f"Vehicle {vid} not found."

        veh = self.sce.vehicles[vid]
        currentLaneID = veh.lane_id

        if currentLaneID not in self.sce.lanes:
            return f"Vehicle {vid} is on an unknown lane {currentLaneID}."

        laneIdx = self.sce.lanes[currentLaneID].laneIdx

        leftLane = f'lane_{laneIdx - 1}' if laneIdx > 0 else "None"
        rightLane = f'lane_{laneIdx + 1}' if laneIdx < 3 else "None"

        info = f"The available lanes for `{vid}` (currently in `{currentLaneID}`, index {laneIdx}) are:\n"
        if leftLane != "None":
            info += f"- `{leftLane}` is to the LEFT (index {laneIdx - 1})\n"
        else:
            info += f"- NO lane to the LEFT (Action LANE_LEFT is INVALID)\n"

        info += f"- `{currentLaneID}` is the CURRENT lane\n"

        if rightLane != "None":
            info += f"- `{rightLane}` is to the RIGHT (index {laneIdx + 1})\n"
        else:
            info += f"- NO lane to the RIGHT (Action LANE_RIGHT is INVALID)\n"

        return info


class getLaneInvolvedCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce

    @prompts(name='Get_Lane_Involved_Car',
             description="""useful whent want to know the cars may affect your action in the certain lane. Make sure you have use tool `Get Available Lanes` first. The input is a string, representing the id of the specific lane you want to drive on, DONNOT input multiple lane_id once.""")
    def inference(self, laneID: str) -> str:
        print(f"\n{'=' * 80}")
        print(f"[DEBUG] TOOL CALL: Get_Lane_Involved_Car")
        print(f"[DEBUG] Target lane: {laneID}")
        print(f"{'=' * 80}")

        if laneID not in {'lane_0', 'lane_1', 'lane_2', 'lane_3'}:
            return "Not a valid lane id! Make sure you have use tool `Get Available Lanes` first."
        ego = self.sce.vehicles['ego']

        print(f"[DEBUG] Ego position: {ego.lanePosition:.2f}m, speed: {ego.speed:.2f}m/s, lane: {ego.lane_id}")

        laneVehicles = []
        for vk, vv in self.sce.vehicles.items():
            if vk != 'ego':
                if vv.lane_id == laneID:
                    laneVehicles.append((vv.id, vv.lanePosition))

        laneVehicles.sort(key=lambda x: x[1])

        vehicles_ahead = [v for v in laneVehicles if v[1] < ego.lanePosition]
        vehicles_behind = [v for v in laneVehicles if v[1] > ego.lanePosition]

        # Closest Ahead: The one with LARGEST pos in the Ahead list (closest to ego)
        closest_ahead = vehicles_ahead[-1] if vehicles_ahead else None

        # Closest Behind: The one with SMALLEST pos in the Behind list (closest to ego)
        closest_behind = vehicles_behind[0] if vehicles_behind else None

        if not closest_ahead:
            if not closest_behind:
                return f'There is no car currently detected in {laneID}. However, you MUST still verify the scenario data carefully. For lane changes, consider: 1) Vehicles might be approaching fast, 2) Your speed relative to lane traffic, 3) Whether to adjust speed during the lane change.'
            else:
                rearingCar = closest_behind[0]
                rearing_speed = round(self.sce.vehicles[rearingCar].speed, 1)
                rearing_distance = round(closest_behind[1] - ego.lanePosition, 2)
                return f"{rearingCar} is driving at {rearing_speed}m/s on {laneID}, {rearing_distance}m BEHIND ego car. Ego speed: {round(ego.speed, 1)}m/s. Lane change might be safe if they are far enough."
        else:
            leadingCar = closest_ahead[0]
            distance_ahead = round(ego.lanePosition - closest_ahead[1], 2)
            leading_car_vel = round(self.sce.vehicles[leadingCar].speed, 1)

            if not closest_behind:
                return f"{leadingCar} is driving at {leading_car_vel}m/s on {laneID}, {distance_ahead}m AHEAD of ego car. Ego speed: {round(ego.speed, 1)}m/s. Speed difference: {round(ego.speed - leading_car_vel, 1)}m/s. You MUST check safety with this vehicle."
            else:
                rearingCar = closest_behind[0]
                distance_behind = round(closest_behind[1] - ego.lanePosition, 2)
                rearing_car_vel = round(self.sce.vehicles[rearingCar].speed, 1)
                return f"{leadingCar} (AHEAD at {leading_car_vel}m/s, {distance_ahead}m away) and {rearingCar} (BEHIND at {rearing_car_vel}m/s, {distance_behind}m away) are on {laneID}. Ego speed: {round(ego.speed, 1)}m/s. You MUST check safety with BOTH vehicles."


class isChangeLaneConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 1.0  # Aggressive lane change

    @prompts(name='Is_Change_Lane_Conflict_With_Car',
             description="""useful when you want to know whether change lane to a specific lane is confict with a specific car. Input: "laneID, vid".""")
    def inference(self, inputs: str) -> str:
        print(f"\n{'=' * 80}")
        print(f"[DEBUG] TOOL CALL: Is_Change_Lane_Conflict_With_Car")
        print(f"[DEBUG] Input: {inputs}")
        print(f"{'=' * 80}")

        try:
            laneID, vid = inputs.replace(' ', '').split(',')
        except ValueError:
            return "Input Error: Please provide laneID and vid separated by comma."

        if vid not in self.sce.vehicles:
            return "Your input is not a valid vehicle id!"
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']

        print(f"[DEBUG] Ego: pos={ego.lanePosition:.2f}m, speed={ego.speed:.2f}m/s")
        print(f"[DEBUG] {vid}: pos={veh.lanePosition:.2f}m, speed={veh.speed:.2f}m/s")

        if veh.lanePosition < ego.lanePosition:
            relativeSpeed = ego.speed - veh.speed
            distance = ego.lanePosition - veh.lanePosition

            min_safe_distance = 0.2  # Very aggressive
            time_based_distance = max(0, self.TIME_HEAD_WAY * relativeSpeed) if relativeSpeed > 0 else 0
            required_distance = max(min_safe_distance, time_based_distance)

            print(f"[DEBUG] Vehicle AHEAD. Dist: {distance:.2f}m, Req: {required_distance:.2f}m")

            if distance > required_distance:
                return f"Change lane to `{laneID}` is SAFE with `{vid}` (ahead). Distance: {distance:.1f}m. Recommendation: Watch speed."
            else:
                return f"Change lane to `{laneID}` CONFLICTS with `{vid}` (ahead). Distance {distance:.1f}m too small (req {required_distance:.1f}m). UNSAFE."

        else:
            relativeSpeed = veh.speed - ego.speed
            distance = veh.lanePosition - ego.lanePosition

            min_safe_distance = 0.2
            time_based_distance = max(0, self.TIME_HEAD_WAY * relativeSpeed) if relativeSpeed > 0 else 0
            required_distance = max(min_safe_distance, time_based_distance)

            print(f"[DEBUG] Vehicle BEHIND. Dist: {distance:.2f}m, Req: {required_distance:.2f}m")

            if distance > required_distance:
                return f"Change lane to `{laneID}` is SAFE with `{vid}` (behind). Distance: {distance:.1f}m."
            else:
                return f"Change lane to `{laneID}` CONFLICTS with `{vid}` (behind). Distance {distance:.1f}m too small (req {required_distance:.1f}m). UNSAFE."


class isEmptyLaneChangeSafe:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce

    @prompts(name='Is_Empty_Lane_Change_Safe',
             description="""IMPORTANT: Use this tool when Get_Lane_Involved_Car reports NO vehicles in the target lane. Verifies safety.""")
    def inference(self, laneID: str) -> str:
        print(f"\n{'=' * 80}")
        print(f"[DEBUG] TOOL CALL: Is_Empty_Lane_Change_Safe")
        print(f"[DEBUG] Target lane: {laneID}")
        print(f"{'=' * 80}")

        if laneID not in {'lane_0', 'lane_1', 'lane_2', 'lane_3'}:
            return "Not a valid lane id!"

        ego = self.sce.vehicles['ego']

        current_lane_vehicles = []
        for vid, veh in self.sce.vehicles.items():
            if vid != 'ego' and veh.lane_id == ego.lane_id:
                distance = abs(ego.lanePosition - veh.lanePosition)
                if distance < 0.5:
                    current_lane_vehicles.append((vid, distance))

        if current_lane_vehicles:
            closest = min(current_lane_vehicles, key=lambda x: x[1])
            return f"⚠️ CAUTION: {closest[0]} is very close ({closest[1]:.2f}m) in CURRENT lane. Ensure you have room to maneuver. However, if target lane is empty, proceed with caution."

        vehicles_in_lane = []
        for vid, veh in self.sce.vehicles.items():
            if vid != 'ego' and veh.lane_id == laneID:
                vehicles_in_lane.append(vid)

        if vehicles_in_lane:
            return f"⚠️ WARNING: Lane {laneID} is NOT EMPTY! Found: {vehicles_in_lane}. Use Get_Lane_Involved_Car."

        return f"Lane {laneID} is verified EMPTY and SAFE. Proceed."


class isAccelerationConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 1.5  # Reduced from 5.0 for more aggressive following
        self.acceleration = 4.0

    @prompts(name='Is_Acceleration_Conflict_With_Car',
             description="""useful when you want to know whether acceleration is safe with a specific car.""")
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles:
            return "Invalid vehicle ID."
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']

        if veh.lane_id != ego.lane_id:
            return f'{vid} is not in current lane.'

        if veh.lanePosition < ego.lanePosition:
            distance = ego.lanePosition - veh.lanePosition
            relativeSpeed = ego.speed + self.acceleration - veh.speed

            min_safe_distance = 0.1  # Allow creeping
            time_based_distance = max(0, self.TIME_HEAD_WAY * relativeSpeed) if relativeSpeed > 0 else 0
            required_distance = max(min_safe_distance, time_based_distance)

            # STUCK CHECK: If speed is very low and distance is small but not zero, allow nudge
            if ego.speed < 0.5 and distance > 0.0:
                return f"⚠️ CONDITIONAL CAUTION: Vehicle is close ({distance:.2f}m) but ego is stopped. Acceleration MAY be used to nudge/steer for a lane change if the TARGET lane is clear. PROCEED WITH CAUTION."

            if distance > required_distance:
                return f"Acceleration is SAFE with `{vid}`. Distance: {distance:.1f}m."
            else:
                return f"Acceleration CONFLICTS with `{vid}`. Distance {distance:.1f}m < req {required_distance:.1f}m."
        else:
            return f"Acceleration is SAFE with {vid} (behind)."


class isKeepSpeedConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 1.5  # Reduced from 5.0

    @prompts(name='Is_Keep_Speed_Conflict_With_Car',
             description="""useful when you want to know whether keep speed is safe with a specific car.""")
    def inference(self, vid: str) -> str:
        print(f"\n{'=' * 80}")
        print(f"[DEBUG] TOOL CALL: Is_Keep_Speed_Conflict_With_Car")
        print(f"[DEBUG] Checking vehicle: {vid}")
        print(f"{'=' * 80}")

        if vid not in self.sce.vehicles:
            return "Invalid vehicle ID."
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']

        if veh.lane_id != ego.lane_id:
            return f'{vid} is not in current lane.'

        if veh.lanePosition < ego.lanePosition:
            # Vehicle is AHEAD
            distance = ego.lanePosition - veh.lanePosition
            relativeSpeed = ego.speed - veh.speed

            min_safe_distance = 0.1  # Allow closer following
            time_based_distance = max(0, self.TIME_HEAD_WAY * relativeSpeed) if relativeSpeed > 0 else 0
            required_distance = max(min_safe_distance, time_based_distance)

            print(
                f"[DEBUG] Vehicle AHEAD. Dist: {distance:.2f}m, Req: {required_distance:.2f}m, RelSpeed: {relativeSpeed:.2f}m/s")

            if distance > required_distance:
                return f"Keep Speed is SAFE with `{vid}` (ahead). Distance: {distance:.1f}m. Gap sufficient."
            else:
                return f"Keep Speed CONFLICTS with `{vid}` (ahead). Distance {distance:.1f}m < req {required_distance:.1f}m. You MUST DECELERATE or CHANGE LANE."
        else:
            distance = veh.lanePosition - ego.lanePosition
            if distance < 0.5:
                return f"Keep Speed SAFE with `{vid}` (behind), but they are very close ({distance:.1f}m)."
            return f"Keep Speed SAFE with `{vid}` (behind)."


class isDecelerationSafe:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 1.0
        self.deceleration = 3.0

    @prompts(name='Is_Deceleration_Safe',
             description="""useful when you want to know whether deceleration is safe. Check vehicles BEHIND. Safe to use for stopping.""")
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles:
            return "Invalid vehicle ID."
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']

        if veh.lane_id != ego.lane_id:
            return f'{vid} is not in current lane.'

        if veh.lanePosition > ego.lanePosition:
            distance = veh.lanePosition - ego.lanePosition
            relativeSpeed = veh.speed - (ego.speed - self.deceleration)

            min_safe_distance = 0.1
            time_based_distance = max(0, self.TIME_HEAD_WAY * relativeSpeed) if relativeSpeed > 0 else 0
            required_distance = max(min_safe_distance, time_based_distance)

            if distance > required_distance:
                return f"Deceleration is SAFE with `{vid}` (behind)."
            else:
                return f"Deceleration CONFLICTS with `{vid}` (behind). Distance {distance:.1f}m < req {required_distance:.1f}m."
        else:
            return f"Deceleration is SAFE with {vid} (ahead) - Use this to STOP if needed."