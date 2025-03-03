#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry  # Import Odometry message
from geometry_msgs.msg import Twist
from custom_msgs.srv import ChatPrompt
import openai
import re  # Import regular expression module for better parsing
from sensor_msgs.msg import LaserScan
import math
from collections import deque
import numpy as np  # For handling U_full, X_0T, and X_1T arrays
from std_msgs.msg import Float64MultiArray

from pyzonotope.Zonotope import Zonotope
from pyzonotope.SafetyLayer import SafetyLayer
import random 

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("safe_robot_controller")
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_timer(0.1, self.trigger_odom_callback)
        self.publisher = self.create_publisher(Float64MultiArray, "/parsed_plan", 10)

        # Store latest odometry message
        self.latest_odom = None
        # Set your OpenAI API key
        self.api_key = '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'
        openai.api_key = self.api_key

        # Create a ROS 2 service
        self.srv = self.create_service(ChatPrompt, 'chat_gpt_ask', self.service_callback)

        # Variables to store the latest data
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.yaw = 0.0
        self.previous_linear_x = 0.0
        self.previous_angular_z = 0.0


       # Subscription to LaserScan topic
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',  # Change this topic if necessary
            self.laser_scan_callback,
            20
        )

        # Subscription to cmd_vel topic
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            '/cmd_vel',  # Change this topic if necessary
            self.cmd_vel_callback,
            10
        )

        # Subscription to odom topic
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',  # Change this topic if necessary
            self.odom_callback2,
            10
        )


        # Buffers for cmd_vel and odom data
        self.cmd_vel_buffer = deque(maxlen=10)
        self.odom_buffer = deque(maxlen=10)
        self.Lidar_readings = np.zeros(19)  # Allocate for 360 degrees of LiDAR data

        # Initialize state arrays
        self.U_full = np.zeros((2, 100))  # 2×100 for linear and angular velocities
        self.X_0T = np.zeros((3, 100))  # 3×100 for x, y, and yaw
        self.X_1T = np.zeros((3, 100))  # 3×100 for velocities


    def publish_parsed_plan(self, parsed_plan):
       msg = Float64MultiArray()
       msg.data = parsed_plan.flatten().tolist()  # Convert to 1D list
       self.publisher.publish(msg)
        



    def trigger_odom_callback(self):
        if self.latest_odom:
            self.odom_callback(self.latest_odom)

    def odom_callback(self, odo: Odometry):

        # Get the current position from Odometry message
        current_x = odo.pose.pose.position.x
        current_y = odo.pose.pose.position.y
        current_theta = self.yaw
        #Home 
        x_goal = -3
        y_goal = 4

        #World
        #x_goal = 2.0
        #y_goal = 0.5

        reaching_radius = np.sqrt((current_x - x_goal)**2 + (current_y - y_goal)**2)
        LOS =   math.atan2(y_goal - current_y, x_goal - current_x)
        print(reaching_radius)
        # This prompt is sent to ChatGPT to move the robot to the goal position.
        prompt = (
            f"You are the motion controller of a 2D differential drive robot (TurtleBot3). "
            f"Generate control inputs (linear and angular velocities) to move the robot to the target position (x={x_goal}, y={y_goal}) "
            "while avoiding obstacles detected by a Lidar sensor. "
            "The robot's state is (x, y, theta) with reaching radius R. The LOS angle to the target is given as LOS. "
            f"Current state: x={current_x}, y={current_y}, theta={current_theta}, R={reaching_radius}, LOS={LOS}. "
            "The robot is controlled by: "
            "- Linear velocity (LVel) in [0.1, 0.5] m/s. "
            "- Angular velocity (AVel) in [-0.5, 0.5] rad/s which positive value turn CCW."
            "Lidar has 18 beams (-90° to +90°), values in meters (3.5 means no obstacle). "
            f"Current Lidar readings: {self.Lidar_readings}. "
            "Move the robot to the target position by minimizing R while maintaining the yaw angle along the line of sight (LOS)."
            "Rmemeber the colation of obstacles an do not move the robot to the obstales which you already have seen."
            "Output exactly in this format: LVel:[v1,v2,v3] AVel:[w1,w2,w3] "
            "where v1-v10 are linear velocities and w1-w10 are angular velocities. "
            "Do NOT include any extra words, only return the required format."
        )


        

                # Get ChatGPT response
        chat_response = self.get_chat_gpt_response(prompt)
        parsed_plan = self.parse_chat_response(chat_response)

        # Perform safety check and publish commands
        if reaching_radius < 0.5:
            parsed_plan[0, 0] = 0.0
            parsed_plan[0, 1] = 0.0
        
        self.publish_parsed_plan(parsed_plan)
        #print(parsed_plan)

    
    def service_callback(self, request, response):
        response.response = self.get_chat_gpt_response(request.prompt)
        return response

    def get_chat_gpt_response(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4o',
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1

            )
            return response.choices[0].message.content
        except Exception as e:
            self.get_logger().error(f'Error calling OpenAI API: {e}')
            return "Error retrieving response from ChatGPT."



    def parse_chat_response(self, response):
        """
        Parse the ChatGPT response to extract linear and angular velocities as 2D arrays.
        The response is assumed to be a string like: "LVel:[0.1, 0.1, 0.1] AVel:[0.0, 0.0, 0.0]".
        """
        try:
            # Log the raw response for debugging
            self.get_logger().info(f"Raw response: {response}")

            # Use regex to match both LVel and AVel arrays
            match_lvel = re.search(r'LVel:\s*\[([^\]]+)\]', response)
            match_avel = re.search(r'AVel:\s*\[([^\]]+)\]', response)

            if match_lvel and match_avel:
                # Log the matched portions (optional for debugging)
                ##self.get_logger().info(f"Matched AVel: {match_avel.group(1)}")

                # Extract the matched strings containing numbers for both LVel and AVel
                lvel_str = match_lvel.group(1)
                avel_str = match_avel.group(1)

                # Convert the strings to lists of floats
                lvel_values = [float(x.strip()) for x in lvel_str.split(',')]
                avel_values = [float(x.strip()) for x in avel_str.split(',')]

                # Ensure both arrays have the same length (3 elements each)
                if len(lvel_values) == 3 and len(avel_values) == 3:
                    # Combine both lists into a 2D array: [[LVel_x, LVel_y, LVel_z], [AVel_x, AVel_y, AVel_z]]
                    velocities = np.array([lvel_values, avel_values]).T
                    return velocities
                else:
                    raise ValueError("Both LVel and AVel should contain exactly 10 elements.")
            else:
                raise ValueError("Invalid response format: Could not find 'LVel' or 'AVel' arrays.")

        except Exception as e:
            # Log the error (optional for debugging)
            self.get_logger().error(f"Error parsing response: {e}")

            # Return a safe default velocities array (2x3 zero array) if parsing fails
            return np.zeros((3, 2))

    
    def laser_scan_callback(self, msg):
        # Filter and store valid LiDAR range values
        self.Lidar_readings = np.array(msg.ranges)

        for i in range(len(self.Lidar_readings)):
                if self.Lidar_readings[i] == float('Inf'):
                    self.Lidar_readings[i] = 3.5  # Replace infinite readings with 3.5
                elif np.isnan(self.Lidar_readings[i]):
                    self.Lidar_readings[i] = 0.0001  # Replace NaN values with 0

        #self.get_logger().info(f"Filtered LiDAR readings: {self.Lidar_readings}")



    def cmd_vel_callback(self, msg):
        self.linear_x = msg.linear.x
        self.angular_z = msg.angular.z

    def odom_callback2(self, msg):
        self.x_pos = msg.pose.pose.position.x
        self.y_pos = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        # Update X_0T with the latest position and orientation
        self.X_0T[0, :-1] = self.X_0T[0, 1:]  # Shift old values
        self.X_0T[1, :-1] = self.X_0T[1, 1:]
        self.X_0T[2, :-1] = self.X_0T[2, 1:]
        self.X_0T[:, -1] = [self.x_pos, self.y_pos, self.yaw]


def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
