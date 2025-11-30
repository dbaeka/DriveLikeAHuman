from scenario.scenario import Scenario
from typing import Any


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
    0: 'change lane to the left of the current lane,',
    1: 'remain in the current lane with current speed',
    2: 'change lane to the right of the current lane',
    3: 'accelerate the vehicle',
    4: 'decelerate the vehicle'
}


class getAvailableActions:
    def __init__(self, env: Any) -> None:
        self.env = env

    @prompts(name='Get_Available_Actions',
             description="""Useful before you make decisions, this tool let you know what are your available actions in this situation. The input to this tool should be 'ego'.""")
    def inference(self, input: str) -> str:
        if hasattr(self.env, 'unwrapped'):
            base_env = self.env.unwrapped
        else:
            base_env = self.env

        # specific check for highway-env method
        if hasattr(base_env, 'get_available_actions'):
            availableActions = base_env.get_available_actions()
        else:
            # Fallback for safety
            return "Error: Could not retrieve available actions from environment."

        outputPrefix = 'You can ONLY use one of the following actions: \n'
        for action in availableActions:
            outputPrefix += ACTIONS_ALL.get(action, 'UNKNOWN') + \
                            '--' + ACTIONS_DESCRIPTION.get(action, '') + '; \n'

        # Safety-focused action guidelines (no bias toward any specific action)
        outputPrefix += '\n**MANDATORY SAFETY VERIFICATION FOR ALL ACTIONS:**\n'
        outputPrefix += '\n⚠️  WARNING: You CANNOT assume lanes are empty. You MUST use Get_Lane_Involved_Car first!\n\n'

        if 0 in availableActions or 2 in availableActions:
            outputPrefix += '- Lane changes (left/right):\n'
            outputPrefix += '  1. Call Get_Lane_Involved_Car with the target lane\n'
            outputPrefix += '  2. For EACH vehicle returned, call Is_Change_Lane_Conflict_With_Car\n'
            outputPrefix += '  3. Only proceed if ALL checks return "safe"\n\n'
        if 3 in availableActions:
            outputPrefix += '- Acceleration:\n'
            outputPrefix += '  1. Call Get_Lane_Involved_Car with your current lane\n'
            outputPrefix += '  2. For EACH vehicle AHEAD of you, call Is_Acceleration_Conflict_With_Car\n'
            outputPrefix += '  3. Only proceed if ALL checks return "safe"\n\n'
        if 1 in availableActions:
            outputPrefix += '- Idle (maintain speed):\n'
            outputPrefix += '  1. Call Get_Lane_Involved_Car with your current lane\n'
            outputPrefix += '  2. For EACH vehicle AHEAD of you, call Is_Keep_Speed_Conflict_With_Car\n'
            outputPrefix += '  3. Only proceed if ALL checks return "safe"\n\n'
        if 4 in availableActions:
            outputPrefix += '- Deceleration:\n'
            outputPrefix += '  1. Call Get_Lane_Involved_Car with your current lane\n'
            outputPrefix += '  2. For EACH vehicle BEHIND you, call Is_Deceleration_Safe\n'
            outputPrefix += '  3. IMPORTANT: Vehicles behind you may not be able to stop in time if you decelerate suddenly\n'
            outputPrefix += '  4. If unsafe with vehicles behind, consider lane change instead\n\n'

        outputPrefix += """
🚨 CRITICAL SAFETY PROCEDURE (YOU MUST FOLLOW THIS):
Step 1: Identify which lane your action affects
        - Acceleration, deceleration, idle → affect CURRENT lane
        - Lane changes → affect TARGET lane
Step 2: Call Get_Lane_Involved_Car for that lane
        - This returns ALL vehicles in that lane
        - If it returns vehicles, you MUST check them
        - If it says "no cars", you still need to verify with scenario data
Step 3: For EACH vehicle returned in Step 2, call the appropriate safety tool
        - Is_Change_Lane_Conflict_With_Car for lane changes
        - Is_Acceleration_Conflict_With_Car for acceleration
        - Is_Keep_Speed_Conflict_With_Car for maintaining speed
        - Is_Deceleration_Safe for deceleration
Step 4: ONLY proceed if ALL safety checks pass

❌ FORBIDDEN: Saying "there are no vehicles" without calling Get_Lane_Involved_Car
❌ FORBIDDEN: Skipping safety checks for any vehicle returned by Get_Lane_Involved_Car
✅ REQUIRED: Check safety with EVERY vehicle in the affected lane

Remember to use the proper tools mentioned in the tool list ONCE at a time.
"""
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

        # Safety check for lane existence
        if currentLaneID not in self.sce.lanes:
            return f"Vehicle {vid} is on an unknown lane {currentLaneID}."

        laneIdx = self.sce.lanes[currentLaneID].laneIdx
        if laneIdx == 3:
            leftLane = 'lane_2'
            return f"""The availabel lane of `{vid}` is `{leftLane}` and `{currentLaneID}`. `{leftLane}` is to the left of the current lane. `{currentLaneID}` is the current lane."""
        elif laneIdx == 0:
            rightLane = 'lane_1'
            return f"""The availabel lane of `{vid}` is `{currentLaneID}` and `{rightLane}`. `{currentLaneID}` is the current lane. `{rightLane}` is to the right of the current lane."""
        else:
            leftLane = 'lane_' + str(laneIdx - 1)
            rightLane = 'lane_' + str(laneIdx + 1)
            return f"""The availabel lane of `{vid}` is `{currentLaneID}`, `{rightLane}` and {leftLane}. `{currentLaneID}` is the current lane. `{rightLane}` is to the right of the current lane. `{leftLane}` is to the left of the current lane."""


class getLaneInvolvedCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce

    @prompts(name='Get_Lane_Involved_Car',
             description="""useful whent want to know the cars may affect your action in the certain lane. Make sure you have use tool `Get Available Lanes` first. The input is a string, representing the id of the specific lane you want to drive on, DONNOT input multiple lane_id once.""")
    def inference(self, laneID: str) -> str:
        if laneID not in {'lane_0', 'lane_1', 'lane_2', 'lane_3'}:
            return "Not a valid lane id! Make sure you have use tool `Get Available Lanes` first."
        ego = self.sce.vehicles['ego']
        laneVehicles = []
        for vk, vv in self.sce.vehicles.items():
            if vk != 'ego':
                if vv.lane_id == laneID:
                    laneVehicles.append((vv.id, vv.lanePosition))
        laneVehicles.sort(key=lambda x: x[1])
        leadingCarIdx = -1
        for i in range(len(laneVehicles)):
            vp = laneVehicles[i]
            if vp[1] >= ego.lanePosition:
                leadingCarIdx = i
                break
        if leadingCarIdx == -1:
            try:
                rearingCar = laneVehicles[-1][0]
            except IndexError:
                return f'There is no car driving on {laneID}. This lane appears clear, but you should still verify safety before taking action.'
            return f"{rearingCar} is driving on {laneID}, and it's driving behind ego car. You need to make sure that your actions do not conflict with each of the vehicles mentioned."
        elif leadingCarIdx == 0:
            leadingCar = laneVehicles[0][0]
            distance = round(laneVehicles[0][1] - ego.lanePosition, 2)
            leading_car_vel = round(self.sce.vehicles[leadingCar].speed, 1)
            return f"{leadingCar} is driving at {leading_car_vel}m/s on {laneID}, and it's driving in front of ego car for {distance} meters. You need to make sure that your actions do not conflict with each of the vehicles mentioned."
        else:
            leadingCar = laneVehicles[leadingCarIdx][0]
            rearingCar = laneVehicles[leadingCarIdx - 1][0]
            distance = round(laneVehicles[leadingCarIdx][1] - ego.lanePosition, 2)
            leading_car_vel = round(self.sce.vehicles[leadingCar].speed, 1)
            return f"{leadingCar} and {rearingCar} is driving on {laneID}, and {leadingCar} is driving at {leading_car_vel}m/s in front of ego car for {distance} meters, while {rearingCar} is driving behind ego car. You need to make sure that your actions do not conflict with each of the vehicles mentioned."


class isChangeLaneConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0

    @prompts(name='Is_Change_Lane_Conflict_With_Car',
             description="""useful when you want to know whether change lane to a specific lane is confict with a specific car, ONLY when your decision is change_lane_left or change_lane_right. The input to this tool should be a string of a comma separated string of two, representing the id of the lane you want to change to and the id of the car you want to check.""")
    def inference(self, inputs: str) -> str:
        try:
            laneID, vid = inputs.replace(' ', '').split(',')
        except ValueError:
            return "Input Error: Please provide laneID and vid separated by comma."

        if vid not in self.sce.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']
        if veh.lanePosition >= ego.lanePosition:
            relativeSpeed = ego.speed - veh.speed
            if veh.lanePosition - ego.lanePosition - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"change lane to `{laneID}` is safe with `{vid}`."
            else:
                return f"change lane to `{laneID}` may be conflict with `{vid}`, which is unacceptable."
        else:
            relativeSpeed = veh.speed - ego.speed
            if ego.lanePosition - veh.lanePosition - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"change lane to `{laneID}` is safe with `{vid}`."
            else:
                return f"change lane to `{laneID}` may be conflict with `{vid}`, which is unacceptable."


class isAccelerationConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0
        self.acceleration = 4.0

    @prompts(name='Is_Acceleration_Conflict_With_Car',
             description="""useful when you want to know whether acceleration is safe with a specific car, ONLY when your decision is accelerate. The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']

        # Only check vehicles in the same lane
        if veh.lane_id != ego.lane_id:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'

        # Check if vehicle is ahead of ego
        if veh.lanePosition >= ego.lanePosition:
            # Vehicle is ahead - need to verify safety
            distance = veh.lanePosition - ego.lanePosition - self.VEHICLE_LENGTH * 2
            relativeSpeed = ego.speed + self.acceleration - veh.speed
            required_distance = self.TIME_HEAD_WAY * relativeSpeed

            if distance > required_distance:
                return f"Acceleration is safe with `{vid}`. Distance: {distance:.1f}m, Required: {required_distance:.1f}m, Ego speed: {ego.speed:.1f}m/s, {vid} speed: {veh.speed:.1f}m/s."
            else:
                return f"Acceleration may conflict with `{vid}` which is UNACCEPTABLE. Distance: {distance:.1f}m is less than required {required_distance:.1f}m. Consider maintaining speed or decelerating."
        else:
            # Vehicle is behind ego - generally safe to accelerate
            return f"Acceleration is safe with {vid} (vehicle is behind ego)."


class isKeepSpeedConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0

    @prompts(name='Is_Keep_Speed_Conflict_With_Car',
             description="""useful when you want to know whether keep speed is safe with a specific car, ONLY when your decision is keep_speed. The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']
        if veh.lane_id != ego.lane_id:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        if veh.lane_id == ego.lane_id:
            if veh.lanePosition >= ego.lanePosition:
                relativeSpeed = ego.speed - veh.speed
                distance = veh.lanePosition - ego.lanePosition - self.VEHICLE_LENGTH * 2
                if distance > self.TIME_HEAD_WAY * relativeSpeed:
                    return f"keep lane with current speed is safe with {vid}"
                else:
                    return f"keep lane with current speed may be conflict with {vid}, you need consider decelerate"
            else:
                return f"keep lane with current speed is safe with {vid}"
        else:
            return f"keep lane with current speed is safe with {vid}"


class isDecelerationSafe:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0
        self.deceleration = 3.0

    @prompts(name='Is_Deceleration_Safe',
             description="""useful when you want to know whether deceleration is safe, ONLY when your decision is decelerate. The input to this tool should be a string, representing the id of the car you want to check. IMPORTANT: Deceleration primarily affects vehicles BEHIND you, so check vehicles behind ego.""")
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the deceleration of ego car, which is meaningless, input a valid vehicle id please!"
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']

        # Only check vehicles in the same lane
        if veh.lane_id != ego.lane_id:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'

        # Check if vehicle is BEHIND ego (this is what matters for deceleration!)
        if veh.lanePosition < ego.lanePosition:
            # Vehicle is behind - need to check if it can stop in time
            distance = ego.lanePosition - veh.lanePosition - self.VEHICLE_LENGTH
            # Relative approach speed if ego decelerates
            relativeSpeed = veh.speed - (ego.speed - self.deceleration)
            required_distance = self.TIME_HEAD_WAY * relativeSpeed

            if distance > required_distance:
                return f"Deceleration is safe with `{vid}` (behind ego). Distance: {distance:.1f}m, Required: {required_distance:.1f}m, Ego speed: {ego.speed:.1f}m/s, {vid} speed: {veh.speed:.1f}m/s."
            else:
                return f"Deceleration may cause conflict with `{vid}` (behind ego) which is UNACCEPTABLE. Distance: {distance:.1f}m is less than required {required_distance:.1f}m. Vehicle behind may not be able to stop in time. Consider lane change instead."
        else:
            # Vehicle is ahead - deceleration won't affect it negatively, always safe
            return f"Deceleration is safe with {vid} (vehicle is ahead of ego, not affected by our deceleration)."
