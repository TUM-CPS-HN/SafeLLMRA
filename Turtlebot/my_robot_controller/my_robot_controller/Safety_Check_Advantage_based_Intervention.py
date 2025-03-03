#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from collections import deque
import numpy as np
import time
from pyzonotope.Zonotope import Zonotope
from pyzonotope.SafetyLayer import SafetyLayer
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import openai
from custom_msgs.srv import ChatPrompt

class DataLoggerNode(Node):
    def __init__(self):
        super().__init__('ABI_Safety_Check')
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Subscriptions
        self.scan_subscription = self.create_subscription(
            LaserScan, '/scan', self.laser_scan_callback, 10
        )
        self.cmd_vel_subscription = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.subscription = self.create_subscription(
            Float64MultiArray,
            "/parsed_plan",
            self.parsed_plan_callback,
            10
        )


        self.parsed_plan = None  # Store received plan
        self.old_safe_plan = np.zeros((1, 2))  # Assuming 10-step plan with [linear, angular] values


        # Variables
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.yaw = 0.0

        self.ranges_buffer = deque(maxlen=100)
        self.Lidar_readings = np.zeros((37,))  # 18 lidar readings
        self.time = 0.0  # 18 lidar readings
        self.counter_LLM = 0.0
        self.Duration_LLM = 0.0
        self.counter_Reachability = 0.0
        self.Duration_Reachability = 0.0

        self.X_0T = np.zeros((3, 100))  # 3 rows: [x, y, yaw]
        




        # Timer to log data (4Hz)
        self.timer = self.create_timer(0.01, self.timer_callback)
        #self.get_logger().info('Data Logger Node (Python) started.')
                # Initialize the figure and axis
        self.fig, self.ax = plt.subplots()
        plt.ion()  # Turn on interactive mode
        plt.show()

    def parsed_plan_callback(self, msg):
        """Store received plan as a 2D matrix."""
        array_length = len(msg.data)
        num_cols = 2  # Assuming two columns: [linear.x, angular.z]
        num_rows = array_length // num_cols

        self.parsed_plan = np.array(msg.data).reshape(num_rows, num_cols)
        #self.get_logger().info(f"Received Parsed Plan:\n{self.parsed_plan}")

    
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

    def odom_callback(self, msg):
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
     
    def backup_policy(self,state):
        """
        Define a safe action when the PID output is unsafe.
        """
        if state["obstacle_detected"]:
            return (0.0, 0.2)  # Stop the robot

        # Limit speed if it exceeds the threshold
        safe_linear = min(state["linear_velocity"], 0.1)  # Max 0.2 m/s
        safe_angular = min(state["angular_velocity"], 0.5)  # Max 0.5 rad/s

        return (safe_linear, safe_angular)


    def Q(self,state, action):
        """
        Estimate the safety cost of an action.
        """
        linear, angular = action

        # Check LiDAR distance
        if state["min_distance"] < 0.3 and linear >0.0:  # If obstacle is too close
            return 1.0  # High risk

        # Penalize high speeds
        if abs(linear) > 0.5 or abs(angular) > 0.5:
            return 0.8  # Moderate risk

        return 0.0  # Safe action

    def safety_filter(self,LLM_plan, state, backup_policy, Q, eta=0.5):
        """
        Modify PID output if it is unsafe.
        """
        # Compute the advantage function A(s, a)
        A = Q(state, LLM_plan) - Q(state, backup_policy(state))

        
        # If the action is unsafe, use the backup policy
        if A > eta:
            print("Action is not safe :(")

            return backup_policy(state)
        
        print("Action is  safe :)")

        return LLM_plan





    def timer_callback(self):
        msg = Twist()

        # Get the latest state and Lidar readings
        latest_x_0t = self.X_0T[:, -1].reshape(-1, 1)  
        yaw2 = float(latest_x_0t[2])
        x = float(latest_x_0t[0])
        y = float(latest_x_0t[1])
        
        
        if self.parsed_plan is None:
            self.get_logger().warn("No parsed plan received yet.")
            return

        if self.parsed_plan is not None:
            plan = self.parsed_plan
            time_LLM = time.time()

            Duration_LLM = time_LLM - self.time
            self.time  = time_LLM
            
            self.counter_LLM = self.counter_LLM + 1
            self.Duration_LLM = self.Duration_LLM + Duration_LLM
            Duration_LLM_Ave = self.Duration_LLM/self.counter_LLM
            #print(f"Duration_LLM:{Duration_LLM_Ave}")



        else:
            plan = self.old_safe_plan




        current_position = np.array([latest_x_0t[0], latest_x_0t[1]], dtype=np.float64).reshape(2, 1) 
        # Motion plan
        """ 
        plan = np.array([
            [.1, 0.],
            [.1, 0.],
            [.1, 0.],



        ])
        """


        # Global state variables
        state = {"linear_velocity": 0.0, "angular_velocity": 0.0, "min_distance": float('inf'), "obstacle_detected": False}



        start_time_Reachability = time.time()
        plan_safe = np.zeros_like(plan)

        # Enforce safety
        state["min_distance"] = min(self.Lidar_readings)  # Closest obstacle
        state["obstacle_detected"] = state["min_distance"] < 0.5
        state["linear_velocity"] = plan[0, 0]
        state["angular_velocity"] = plan[0, 1]


        # Apply safety filter
        safe_linear, safe_angular = self.safety_filter(
                    plan[0,:],  # LLM plan
                    state,  # Current state
                    self.backup_policy,  # Backup policy function
                    self.Q,  # Pass the Q function correctly
                    0.5  # Threshold eta
                )
        plan_safe = np.array([safe_linear,safe_angular])      
        
        stop_time_Reachability = time.time()
        Duration_Reachability = stop_time_Reachability - start_time_Reachability

        # Update counters
        self.counter_Reachability += 1
        self.Duration_Reachability += Duration_Reachability
        Duration_Reachability_Ave = self.Duration_Reachability / self.counter_Reachability

        #print(f"Duration_Reachability Average: {Duration_Reachability_Ave:.4f} seconds")

        print(plan_safe)


        msg.linear.x = plan_safe[0]
        msg.angular.z = plan_safe[1]
        self.cmd_vel_pub.publish(msg)

        self.old_safe_plan = plan_safe

        """
        #print(plan_safe)
        # Clear the previous plot but keep the figure
        self.ax.cla()

        # Function to apply the coordinate transformation (90-degree counterclockwise)
        def rotate_coordinates(coords):
            return np.array([coords[1], -coords[0]])

        # Plot the obstacles
        for i in range(len(obstacles)):
            # Define the Zonotope object X0
            dim_x = 2  # assuming a 2D Zonotope
            center = obstacles[i].center()  # Center of the Zonotope
            generators = obstacles[i].generators()  # Generators of the Zonotope

            # Apply rotation to center and generators
            rotated_center = rotate_coordinates(center)
            rotated_generators = np.array([rotate_coordinates(gen) for gen in generators.T]).T

            X0 = Zonotope(rotated_center.reshape(2, 1), rotated_generators)  # Create the rotated Zonotope

            # Get the polygon of the rotated Zonotope
            polygon_vertices = X0.polygon()

            # Set different edge weights
            weight = 1 + i * 0.5  # Example: Increase line thickness for each obstacle

            # Create the polygon object
            polygon_patch = Polygon(
                polygon_vertices.T, closed=True,
                facecolor='none', edgecolor='red', lw=weight, alpha=1.0, linestyle='-'
            )

            # Add the Polygon patch to the plot
            self.ax.add_patch(polygon_patch)

        # Plot reachable set
        num_sets = len(Reachable_Set_NL)

        for i in range(len(Reachable_Set_NL)):
            X00 = Reachable_Set_NL[i]
            center_2D = np.array(X00.center())
            generators_2D = np.array(X00.generators())

            # Apply rotation to center and generators
            rotated_center = rotate_coordinates(center_2D[:2])
            rotated_generators = np.array([rotate_coordinates(gen) for gen in generators_2D.T]).T

            reachability_state_2D = Zonotope(rotated_center.reshape(2, 1), rotated_generators)

            # Get the polygon vertices
            polygon_vertices = reachability_state_2D.polygon()

            # Color logic for reachable sets
            if i == 0:
                color = 'red'  
                line = '-'
            else:
                color = plt.cm.Greens(i / num_sets)  
                line = '--'

            # Create the polygon patch
            polygon_patch = Polygon(
                polygon_vertices.T, closed=True,
                facecolor=color, edgecolor='black', lw=2, alpha=0.5, linestyle=line
            )

            # Add the Polygon patch to the plot
            self.ax.add_patch(polygon_patch)

        # Set plot limits and refresh
        #self.ax.set_xlim(-5, 5)
        #self.ax.set_ylim(-3, 5)
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(2, 8)
        self.ax.invert_yaxis()  # This makes y-axis point left
        self.ax.invert_xaxis()  # This makes y-axis point left
        self.ax.grid()

        plt.draw()
        plt.pause(0.01)  # Small pause to allow the plot to update

        """
def main(args=None):
    rclpy.init(args=args)
    node = DataLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
