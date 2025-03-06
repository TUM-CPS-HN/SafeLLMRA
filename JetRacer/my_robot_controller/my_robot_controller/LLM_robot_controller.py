#!/home/jetson/ros_venv/bin/python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
from custom_msgs.srv import ChatPrompt, ChatPromptResponse
from std_msgs.msg import Float64MultiArray
import math
from collections import deque
import numpy as np
import asyncio
import websockets
import json
import threading
import time
import re


class RobotControllerNode:
    def __init__(self):
        # Initialize node
        rospy.init_node("safe_robot_controller", anonymous=True)

        # Publishers
        self.publisher = rospy.Publisher(
            "/parsed_plan", Float64MultiArray, queue_size=10
        )
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laser_scan_callback)
        self.cmd_vel_sub = rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)
        self.imu_sub = rospy.Subscriber("/imu/data", Imu, self.imu_callback)

        # Service client for LLM
        rospy.wait_for_service("chat_gpt_ask")
        self.llm_service = rospy.ServiceProxy("chat_gpt_ask", ChatPrompt)

        # Timer for control loop
        rospy.Timer(rospy.Duration(0.01), self.control_loop)

        # Variables for robot state
        self.latest_odom = None
        self.current_cmd_vel = Twist()
        self.previous_cmd_vel = Twist()

        # Position and orientation
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.yaw = 0.0

        # Goal information
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.distance_to_goal = 0.0
        self.angle_to_goal = 0.0

        # LiDAR data
        self.lidar_scan = []
        self.lidar_angles = []
        self.lidar_angle_increment = 0.0

        # IMU data
        self.linear_acceleration = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.angular_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Status information
        self.start_time = rospy.Time.now()
        self.battery_level = 100.0  # Simulated battery

        # Motion capture variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_qx = 0.0
        self.robot_qy = 0.0
        self.robot_qz = 0.0
        self.robot_qw = 0.0
        self.valid_mocap_received = False

        # Start websocket client for motion capture in a separate thread
        self.mocap_thread = threading.Thread(target=self.start_mocap_client)
        self.mocap_thread.daemon = True
        self.mocap_thread.start()

        # LLM control variables
        self.last_plan_time = rospy.Time.now()
        self.plan_timeout = rospy.Duration(1.0)  # Request new plan every second
        self.goal_reached = False

        # Add these new variables for multi-step plan execution
        self.current_plan = []
        self.plan_step = 0
        self.last_step_time = rospy.Time.now()
        self.step_duration = rospy.Duration(0.3)  # 300ms per step

        # Adjust plan timeout to request new plan less frequently
        self.plan_timeout = rospy.Duration(1.0)  # 1 second

        # Add a higher frequency timer for plan execution
        rospy.Timer(rospy.Duration(0.05), self.execute_plan_step)

        # rospy.loginfo("JetRacer LLM Controller initialized")

    async def connect_to_mocap(self, server_ip="192.168.64.147"):
        uri = f"ws://{server_ip}:8765"
        # rospy.loginfo(f"Connecting to motion capture system at {uri}")

        while not rospy.is_shutdown():
            try:
                async with websockets.connect(uri) as websocket:
                    # rospy.loginfo("Connected to motion capture system")
                    while not rospy.is_shutdown():
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)

                            if "Player" in data["objects"]:
                                self.robot_x = data["objects"]["Player"]["x"]
                                self.robot_y = data["objects"]["Player"]["y"]
                                self.robot_qx = data["objects"]["Player"]["qx"]
                                self.robot_qy = data["objects"]["Player"]["qy"]
                                self.robot_qz = data["objects"]["Player"]["qz"]
                                self.robot_qw = data["objects"]["Player"]["qw"]

                                # Update yaw from quaternion
                                siny_cosp = 2 * (
                                    self.robot_qw * self.robot_qz
                                    + self.robot_qx * self.robot_qy
                                )
                                cosy_cosp = 1 - 2 * (
                                    self.robot_qy * self.robot_qy
                                    + self.robot_qz * self.robot_qz
                                )
                                self.yaw = math.atan2(siny_cosp, cosy_cosp)
                                # print(f"Yaw Motion Capture: {(self.yaw - 1.87006) * 180 / math.pi + 31}")

                                self.yaw = self.yaw - 1.87006
                                self.yaw = self.yaw * 180 / math.pi + 31
                                # convert to radians
                                self.yaw = self.yaw * math.pi / 180
                                self.yaw += math.pi / 2
                                LOS = math.atan2(
                                    self.robot_y - self.goal_y,
                                    self.robot_x - self.goal_x,
                                )
                                reaching_radius = np.sqrt(
                                    (self.robot_x - self.goal_x) ** 2
                                    + (self.robot_y - self.goal_y) ** 2
                                )
                                print(
                                    f"LOS: {LOS}, Yaw: {self.yaw}, Reaching Radius: {reaching_radius}"
                                )

                                # # Use mocap for position
                                self.x_pos = self.robot_x
                                self.y_pos = self.robot_y

                            if "Goal" in data["objects"]:
                                self.goal_x = data["objects"]["Goal"]["x"]
                                self.goal_y = data["objects"]["Goal"]["y"]

                                # Update distance and angle to goal
                            self.update_goal_metrics()

                            self.valid_mocap_received = True

                        except websockets.exceptions.ConnectionClosed:
                            # rospy.logwarn(
                            #     "Motion capture connection lost, attempting to reconnect..."
                            # )
                            break
            except Exception as e:
                rospy.logerr(f"Motion capture connection error: {e}")
            await asyncio.sleep(0.5)

    def start_mocap_client(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.connect_to_mocap())

    def update_goal_metrics(self):
        """Update distance and angle to goal"""
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y

        # Calculate distance to goal
        self.distance_to_goal = math.sqrt(dx**2 + dy**2)

        # Calculate angle to goal (relative to robot's heading)
        goal_angle_global = math.atan2(dy, dx)
        self.angle_to_goal = goal_angle_global
        if self.distance_to_goal < 200:  # 20cm threshold
            if not self.goal_reached:
                rospy.loginfo("Goal reached!")
                self.goal_reached = True
        else:
            self.goal_reached = False

    def normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def odom_callback(self, msg):
        """Process odometry data"""
        self.latest_odom = msg

        # Extract velocity from odometry message
        self.current_velocity = {
            "linear": msg.twist.twist.linear.x,
            "angular": msg.twist.twist.angular.z,
        }

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        # self.yaw = math.atan2(siny_cosp, cosy_cosp)

        # print(f"Odom Yaw: {odom_yaw * 180 / math.pi}")

    def laser_scan_callback(self, data):
        """Process laser scan data"""
        # Store the raw ranges and angle information
        # Extract ranges from laser scan
        ranges = np.array(data.ranges)
        angles = np.arange(
            data.angle_min,
            data.angle_min + len(ranges) * data.angle_increment,
            data.angle_increment,
        )

        # Downsample to 5 degree increments (~0.087 radians)
        downsample_factor = int(np.radians(5) / data.angle_increment)
        angles = angles[::downsample_factor]
        ranges = ranges[::downsample_factor]

        # Filter angles between 135° and 225° (in radians: 3π/4 to 5π/4)
        mask = (angles >= 3 * np.pi / 4) | (angles <= -3 * np.pi / 4)

        self.lidar_scan = ranges[mask]

        self.lidar_scan = self.lidar_scan.tolist()

        # Process readings - replace inf/nan values
        for i in range(len(self.lidar_scan)):
            if np.isnan(self.lidar_scan[i]):
                self.lidar_scan[i] = -1  # Indicate no reading
            elif np.isinf(self.lidar_scan[i]):
                self.lidar_scan[i] = -1  # Indicate no reading
            else:
                self.lidar_scan[i] *= 10

    def imu_callback(self, msg):
        """Process IMU data"""
        self.linear_acceleration = {
            "x": msg.linear_acceleration.x,
            "y": msg.linear_acceleration.y,
            "z": msg.linear_acceleration.z,
        }

        self.angular_velocity = {
            "x": msg.angular_velocity.x,
            "y": msg.angular_velocity.y,
            "z": msg.angular_velocity.z,
        }

    def cmd_vel_callback(self, msg):
        """Store the current commanded velocities"""
        self.previous_cmd_vel = self.current_cmd_vel
        self.current_cmd_vel = msg

    def execute_plan_step(self, event):
        """Execute steps from the current plan at appropriate intervals"""
        # Skip if no plan or goal reached
        if not self.current_plan or self.goal_reached:
            return

        current_time = rospy.Time.now()
        time_since_last_step = current_time - self.last_step_time

        # Check if it's time for the next step
        if time_since_last_step >= self.step_duration:
            self.plan_step += 1

            # Check if we still have steps in the plan
            if self.plan_step < len(self.current_plan):
                step = self.current_plan[self.plan_step]
                linear_vel = float(step.get("linear_velocity", 0.0))
                angular_vel = float(step.get("angular_velocity", 0.0))

                # Apply velocity limits
                linear_vel = max(-0.5, min(0.6, linear_vel))
                angular_vel = max(-0.6, min(0.6, angular_vel))

                # # Check for obstacle emergency override
                # if self.emergency_obstacle_detected():
                #     rospy.logwarn("Emergency stop - obstacle detected!")
                #     linear_vel = 0.0
                #     angular_vel = 0.15  # Turn to avoid obstacle

                # Log the command being applied
                # rospy.loginfo(
                #     f"Executing plan step {self.plan_step + 1} - Linear: {linear_vel}, Angular: {angular_vel}"
                # )

                # Create and publish command
                cmd = Twist()
                cmd.linear.x = linear_vel
                cmd.angular.z = angular_vel
                # self.cmd_vel_pub.publish(cmd)

                self.last_step_time = current_time
            else:
                # We've executed all steps in the plan
                # We'll wait for the control_loop to request a new plan
                # rospy.loginfo("Plan execution completed")
                self.current_plan = []

    def emergency_obstacle_detected(self):
        """Check if there's an emergency obstacle that requires stopping"""
        if not self.lidar_scan:
            return False

        # Check for obstacles directly in front within emergency threshold
        for i, range_val in enumerate(self.lidar_scan):
            if range_val <= 0:  # Skip invalid readings
                continue

            angle = self.lidar_angles[i]
            # Check if reading is in front sector and very close
            if -0.5 <= angle <= 0.5 and range_val < 0.3:  # 30cm threshold
                return True

        return False

    def control_loop(self, event):
        """Main control loop that requests LLM control at regular intervals"""
        # Update goal metrics
        self.update_goal_metrics()

        # Skip if no valid mocap data yet
        if not self.valid_mocap_received:
            # rospy.logwarn_throttle(5, "Waiting for valid mocap data...")
            return

        # Check if we need to request a new plan
        current_time = rospy.Time.now()
        time_since_last_plan = current_time - self.last_plan_time

        # Only request a new plan if:
        # 1. Enough time has passed since the last plan
        # 2. We've finished executing the current plan
        # 3. We haven't reached the goal yet
        # if (
        #     not self.current_plan or self.plan_step >= len(self.current_plan)
        # ) and not self.goal_reached:
        self.request_llm_control()
        self.last_plan_time = current_time

    def request_llm_control(self):
        """Request and process control commands from LLM"""
        # Prepare sensor data for LLM
        sensor_data = self.prepare_sensor_data()

        # Convert to JSON for LLM
        sensor_data_json = json.dumps(sensor_data, indent=2)

        # Create the prompt with the system context and task
        prompt = self.create_llm_prompt(sensor_data_json)

        try:
            # Request control from LLM
            response = self.llm_service(prompt)
            llm_response = response.response
            # rospy.loginfo("Received response from LLM")

            # Parse and apply the response
            # self.parse_and_apply_llm_response(llm_response)
            plan = self.parse_chat_response(llm_response)

            print(f"Plan: {plan[0]}")
            cmd = Twist()
            cmd.linear.x = plan[0][0]
            cmd.angular.z = plan[0][1]

            msg = Float64MultiArray()
            msg.data = plan.flatten().tolist()  # Convert to 1D list
            # self.cmd_vel_pub.publish(cmd)
            self.publisher.publish(msg)

            # self.cmd_vel_pub.publish(cmd)
            # send the plan to the robot
            

        except rospy.ServiceException as e:
            rospy.logerr(f"LLM service call failed: {e}")

    def prepare_sensor_data(self):
        """Prepare sensor data with simplified directional information"""
        # Calculate elapsed time
        elapsed_time = (rospy.Time.now() - self.start_time).to_sec()

        # Calculate relative position
        # Add relative position data to make it clearer for the LLM
        dx = (self.goal_x - self.x_pos) / 10
        dy = (self.goal_y - self.y_pos) / 10

        # Prepare the data structure
        data = {
            "your_position": {
                "x": str(self.x_pos / 10) + "cm",
                "y": str(self.y_pos / 10) + "cm",
            },
            "orientation": {"yaw": str(self.yaw) + " radians"},
            # "velocity": {
            #     "linear": self.current_cmd_vel.linear.x,
            #     "angular": self.current_cmd_vel.angular.z,
            # },
            # "linear_acceleration": self.linear_acceleration,
            # "angular_velocity": self.angular_velocity,
            "lidar_scan": self.lidar_scan,
            "goal_position": {"x": self.goal_x / 10, "y": self.goal_y / 10},
            "relative_position": {  # Added for clarity
                "x": dx,
                "y": dy,
            },
            "angle_to_goal": math.atan2(
                self.robot_y - self.goal_y, self.robot_x - self.goal_x
            ),
            "distance_to_goal": str(self.distance_to_goal / 10) + " cm",
            # "angle_to_goal": self.angle_to_goal,
            "time_elapsed": elapsed_time,
            "previous_command": {
                "linear": self.previous_cmd_vel.linear.x,
                "angular": self.previous_cmd_vel.angular.z,
            },
        }

        print(f"Data: {data}")
        return data

    def get_obstacle_distances(self, lidar_data, lidar_angles):
        """Process LiDAR data into simplified obstacle distances by direction"""
        # Initialize distances
        distances = {
            "front": 999.9,
            "front-left": 999.9,
            "front-right": 999.9,
            "left": 999.9,
            "right": 999.9,
            "back-left": 999.9,
            "back-right": 999.9,
            "back": 999.9,
        }

        # Map readings to directions and find minimum distance in each
        for i, range_val in enumerate(lidar_data):
            if range_val <= 0:  # Skip invalid readings
                continue

            angle = lidar_angles[i]

            # Map angle to direction
            direction = self.get_direction_label(angle)

            # Update minimum distance for this direction
            distances[direction] = min(distances[direction], range_val)

        # Set a reasonable maximum for "no obstacle detected"
        for direction in distances:
            if distances[direction] > 10.0:
                distances[direction] = 10.0

        return distances

    def get_direction_label(self, angle):
        """Convert angle (in radians) to a direction label"""
        # Normalize angle to -π to π range
        angle = self.normalize_angle(angle)

        # Convert to direction label
        if -0.4 <= angle <= 0.4:
            return "back"
        elif 0.4 < angle <= 1.2:
            return "back-left"
        elif -1.2 <= angle < -0.4:
            return "back-right"
        elif 1.2 < angle <= 2.0:
            return "right"
        elif -2.0 <= angle < -1.2:
            return "left"
        elif 2.0 < angle <= 2.75:
            return "front-right"
        elif -2.75 <= angle < -2.0:
            return "front-left"
        else:
            return "front"

    def create_llm_prompt(self, sensor_data_json):
        """Create a simplified prompt with clearer directional guidance"""
        # prompt = (
        #     "You are an AI controller for a JetRacer robot operating in a ROS environment. "
        #     "Your task is to generate control commands that will navigate the robot to a specified goal position "
        #     "as efficiently as possible while avoiding obstacles.\n\n"
        #     "## Robot Capabilities\n"
        #     "- The JetRacer is a differential drive robot that can move forward/backward and rotate\n"
        #     "- Maximum linear velocity: 0.5 m/s\n"
        #     "- Minimum linear velocity: -0.5 m/s\n"
        #     "- Maximum angular velocity: 0.6 rad/s\n\n"
        #     "- Minimum angular velocity: -0.6 rad/s\n"
        #     "## Directional Guidance\n"
        #     "- The DIRECTION field indicates where the goal is in relation to the robot\n"
        #     "- YAW angles indicates your current orientation\n"
        #     "  * 'front': angle is between -1.73 radians and -2.08 radians\n"
        #     "  * 'front-left': angle between -2.08 and -2.44\n"
        #     "  * 'front-right': angle between -1.39 and -1.74\n"
        #     "  * 'left': angle between 2.65 radians and 2.3 radians\n"
        #     "  * 'right': angle between -0.26 radians and -0.61 radians\n"
        #     "  * 'back-left': angle between -0.17 and -1.39\n"
        #     "  * 'back-right': angle between 0.17 and 1.39\n"
        #     "  * 'back': angle between 1.37 radians and 1.02 radians\n\n"
        #     "- Angle to goal indicates the angle between the robot and the goal."
        #     "It is given in radians by this formula: math.atan2(goal_position_y - your_position_y, goal_position_x - your_position_x)\n\n"
        #     "- Linear velocity: POSITIVE = forward, NEGATIVE = backward\n"
        #     "- Angular velocity: POSITIVE = turn left, NEGATIVE = turn right\n\n"
        #     "## Current Sensor Data\n"
        #     f"{sensor_data_json}\n\n"
        #     "## Required Output Format\n"
        #     "Respond with a valid JSON object containing a 3-step plan with the following structure:\n"
        #     "```json\n"
        #     "{\n"
        #     '  "plan": [\n'
        #     "    {\n"
        #     '      "linear_velocity": <value for step 1>,\n'
        #     '      "angular_velocity": <value for step 1>\n'
        #     "    },\n"
        #     "    {\n"
        #     '      "linear_velocity": <value for step 2>,\n'
        #     '      "angular_velocity": <value for step 2>\n'
        #     "    },\n"
        #     "    {\n"
        #     '      "linear_velocity": <value for step 3>,\n'
        #     '      "angular_velocity": <value for step 3>\n'
        #     "    },\n"
        #     # "    {\n"
        #     # '      "linear_velocity": <value for step 4>,\n'
        #     # '      "angular_velocity": <value for step 4>\n'
        #     # "    },\n"
        #     # "    {\n"
        #     # '      "linear_velocity": <value for step 5>,\n'
        #     # '      "angular_velocity": <value for step 5>\n'
        #     # "    },\n"
        #     # "    {\n"
        #     # '      "linear_velocity": <value for step 6>,\n'
        #     # '      "angular_velocity": <value for step 6>\n'
        #     # "    },\n"
        #     # "    {\n"
        #     # '      "linear_velocity": <value for step 7>,\n'
        #     # '      "angular_velocity": <value for step 7>\n'
        #     # "    }\n"
        #     "  ],\n"
        #     '  "reasoning": "Your explanation of the overall plan"\n,'
        #     '  "direction": "Your choosen direction to move in"\n'
        #     "}\n"
        #     "```\n\n"
        #     "## Planning Guidelines\n"
        #     "1. IMPORTANT: Balance angular adjustments with forward progress. Do not get stuck in repeated angle adjustments.\n"
        #     "2. When the angle is approximately aligned (within ±0.3 radians), prioritize forward movement.\n"
        #     "3. Use a combination of rotation and forward movement when possible.\n"
        #     "4. Each step will be executed for approximately 0.3 seconds.\n"
        #     "5. For distant goals, focus on maintaining a consistent direction rather than perfect alignment.\n"
        #     "6. Linear velocity range: -0.5 to 0.5 m/s\n"
        #     "7. Angular velocity range: -0.6 to 0.6 rad/s\n\n"
        #     "IMPORTANT: Your goal is to reach the target position in the shortest time possible while avoiding obstacles. "
        #     "Return only the JSON object with no additional text."
        # )

        reaching_radius = np.sqrt(
            (self.robot_x - self.goal_x) ** 2 + (self.robot_y - self.goal_y) ** 2
        )
        LOS = math.atan2(self.robot_y - self.goal_y, self.robot_x - self.goal_x)

        prompt = (
            f"You are the motion controller of a 2D differential drive robot. "
            f"Generate control inputs (linear and angular velocities) to move the robot to the target position (x={self.goal_x}, y={self.goal_y}) "
            # "while avoiding obstacles detected by a Lidar sensor. "
            "The robot's state is (x, y, theta) with reaching radius R. The LOS angle to the target is given as LOS. "
            f"Current state: x={self.robot_x}, y={self.robot_y}, theta={self.yaw}, R={reaching_radius}, LOS={LOS}. "
            "The robot is controlled by: "
            "- Linear velocity (LVel) in [0.1, 0.3] m/s. "
            "- Angular velocity (AVel) in [-0.3, 0.3] rad/s which positive value turn CCW."
            # "Lidar has 18 beams (-90° to +90°), values in meters (3.5 means no obstacle). "
            # f"Current Lidar readings: {self.lidar_scan}. "
            "Move the robot to the target position by minimizing R while maintaining the yaw angle along the line of sight (LOS)."
            # "Rmemeber the colation of obstacles an do not move the robot to the obstales which you already have seen."
            "Output exactly in this format: LVel:[v1,v2,v3,v4,v5] AVel:[w1,w2,w3,w4,w5] "
            "where v1-v5 are linear velocities and w1-w5 are angular velocities. "
            "Do NOT include any extra words, only return the required format"
        )

        return prompt

    def parse_chat_response(self, response):
        """
        Parse the ChatGPT response to extract linear and angular velocities as 2D arrays.
        The response is assumed to be a string like: "LVel:[0.1, 0.1, 0.1] AVel:[0.0, 0.0, 0.0]".
        """
        try:
            # Log the raw response for debugging
            rospy.loginfo(f"Raw response: {response}")

            # Use regex to match both LVel and AVel arrays
            match_lvel = re.search(r"LVel:\s*\[([^\]]+)\]", response)
            match_avel = re.search(r"AVel:\s*\[([^\]]+)\]", response)

            if match_lvel and match_avel:
                # Log the matched portions (optional for debugging)
                ##self.get_logger().info(f"Matched AVel: {match_avel.group(1)}")

                # Extract the matched strings containing numbers for both LVel and AVel
                lvel_str = match_lvel.group(1)
                avel_str = match_avel.group(1)

                # Convert the strings to lists of floats
                lvel_values = [float(x.strip()) for x in lvel_str.split(",")]
                avel_values = [float(x.strip()) for x in avel_str.split(",")]

                # Ensure both arrays have the same length (3 elements each)
                if len(lvel_values) == 5 and len(avel_values) == 5:
                    # Combine both lists into a 2D array: [[LVel_x, LVel_y, LVel_z], [AVel_x, AVel_y, AVel_z]]
                    velocities = np.array([lvel_values, avel_values]).T
                    return velocities
                else:
                    raise ValueError(
                        "Both LVel and AVel should contain exactly 10 elements."
                    )
            else:
                raise ValueError(
                    "Invalid response format: Could not find 'LVel' or 'AVel' arrays."
                )

        except Exception as e:
            # Log the error (optional for debugging)
            rospy.logerr(f"Error parsing ChatGPT response: {e}")

            # Return a safe default velocities array (2x3 zero array) if parsing fails
            return np.zeros((5, 2))

    def parse_and_apply_llm_response(self, response_text):
        """Parse the LLM response and apply the multi-step control commands"""
        try:
            print(f"LLM Response: {response_text}")
            # Try to extract JSON from the response
            # First, look for JSON block in markdown
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )

            if json_match:
                json_str = json_match.group(1)
            else:
                # If no markdown block, try to find a JSON object directly
                json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text  # Assume the entire response is JSON

            # Parse the JSON
            command = json.loads(json_str)

            # Extract the plan
            plan = command.get("plan", [])
            reasoning = command.get("reasoning", "No reasoning provided")

            # Log the reasoning
            rospy.loginfo(f"LLM Plan Reasoning: {reasoning}")

            # Publish the plan for visualization/debugging
            plan_msg = Float64MultiArray()
            plan_data = []

            for step in plan:
                linear_vel = float(step.get("linear_velocity", 0.0))
                angular_vel = float(step.get("angular_velocity", 0.0))

                # Apply velocity limits
                linear_vel = max(-0.5, min(0.6, linear_vel))
                angular_vel = max(-0.6, min(0.6, angular_vel))

                plan_data.extend([linear_vel, angular_vel])

            plan_msg.data = plan_data
            self.publisher.publish(plan_msg)

            # Apply the first step immediately
            if plan and not self.goal_reached:
                first_step = plan[0]
                linear_vel = float(first_step.get("linear_velocity", 0.0))
                angular_vel = float(first_step.get("angular_velocity", 0.0))

                # Apply velocity limits
                linear_vel = max(-0.5, min(0.5, linear_vel))
                angular_vel = max(-1.2, min(1.2, angular_vel))

                # Log the command being applied
                rospy.loginfo(
                    f"Applying command - Linear: {linear_vel}, Angular: {angular_vel}"
                )

                # Create and publish command
                cmd = Twist()
                cmd.linear.x = linear_vel
                cmd.angular.z = angular_vel
                # self.cmd_vel_pub.publish(cmd)

                # Store the plan for future steps
                self.current_plan = plan
                self.plan_step = 0
                self.last_step_time = rospy.Time.now()

            else:
                # Stop if no plan or goal reached
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                # self.cmd_vel_pub.publish(cmd)

        except Exception as e:
            # rospy.logerr(f"Error parsing LLM response: {e}")
            # rospy.logerr(f"Raw response: {response_text}")

            # Stop the robot on parsing error
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            # self.cmd_vel_pub.publish(cmd)


def main():
    print("Starting Robot Controller Node")
    node = RobotControllerNode()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
