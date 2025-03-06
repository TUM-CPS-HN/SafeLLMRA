#!/home/jetson/ros_venv/bin/python3
import rospy
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
from matplotlib.animation import FuncAnimation
from std_msgs.msg import Float32
from matplotlib.patches import Polygon
import websockets
import asyncio
import json
import threading


class DataLoggerNode:
    def __init__(self):
        # Initialize ROS 1 node
        rospy.init_node("safety_check_nonlinear_reachability", anonymous=True)
        # Safety layer
        self.safety_layer = SafetyLayer()
        # self.fig, self.ax = plt.subplots()
        self.plot_data = None
        self.time_data = []
        self.value_data = []
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Subscriptions
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laser_scan_callback)
        self.cmd_vel_sub = rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.parsed_plan_sub = rospy.Subscriber(
            "/parsed_plan", Float64MultiArray, self.parsed_plan_callback
        )

        # Variables (same as original)
        self.parsed_plan = None
        self.old_safe_plan = np.zeros((5, 2))
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_qx = 0.0
        self.robot_qy = 0.0
        self.robot_qz = 0.0
        self.robot_qw = 0.0
        self.robot_mocap_received = False
        self.yaw = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.ranges_buffer = deque(maxlen=100)
        self.Lidar_readings = np.zeros((20,))
        self.X_0T = np.zeros((3, 100))

        # Timer (replace ROS 2 timer with ROS 1)
        rospy.Timer(rospy.Duration(0.01), self.timer_callback)

        # Start websocket client for motion capture in a separate thread
        self.mocap_thread = threading.Thread(target=self.start_mocap_client)
        self.mocap_thread.daemon = True
        self.mocap_thread.start()

        # Matplotlib setup

        # plt.ion()
        # plt.show()

    def parsed_plan_callback(self, msg):
        array_length = len(msg.data)
        num_cols = 2
        num_rows = array_length // num_cols
        self.parsed_plan = np.array(msg.data).reshape(num_rows, num_cols)

    def start_mocap_client(self):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run the websocket client
        loop.run_until_complete(self.connect_to_mocap())

    async def connect_to_mocap(self, server_ip="192.168.64.147"):
        uri = f"ws://{server_ip}:8765"
        rospy.loginfo(f"Connecting to motion capture system at {uri}")

        while not rospy.is_shutdown():
            try:
                async with websockets.connect(uri) as websocket:
                    rospy.loginfo("Connected to motion capture system")
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

                                self.x_pos = self.robot_x
                                self.y_pos = self.robot_y

                                self.robot_mocap_received = True

                        except websockets.exceptions.ConnectionClosed:
                            rospy.logwarn(
                                "Motion capture connection lost, attempting to reconnect..."
                            )
                            break
            except Exception as e:
                rospy.logerr(f"Motion capture connection error: {e}")

    def laser_scan_callback(self, msg):
        self.Lidar_readings = np.array(msg.ranges)
        ranges = np.array(msg.ranges)
        angles = np.arange(
            msg.angle_min,
            msg.angle_min + len(ranges) * msg.angle_increment,
            msg.angle_increment,
        )

        # Downsample to 5 degree increments (~0.087 radians)
        downsample_factor = int(np.radians(5) / msg.angle_increment)
        angles = angles[::downsample_factor]
        ranges = ranges[::downsample_factor]

        # Filter angles between 135° and 225° (in radians: 3π/4 to 5π/4)
        mask = (angles >= 3 * np.pi / 4) | (angles <= -3 * np.pi / 4)

        self.Lidar_readings = ranges[mask]

        a = self.Lidar_readings[10:]
        b = self.Lidar_readings[:10]

        self.Lidar_readings = np.array([a, b]).flatten()

        for i in range(len(self.Lidar_readings)):
            if self.Lidar_readings[i] == float("inf"):
                self.Lidar_readings[i] = 3.5
            elif np.isnan(self.Lidar_readings[i]):
                self.Lidar_readings[i] = 0.0001

    def cmd_vel_callback(self, msg):
        self.linear_x = msg.linear.x
        self.angular_z = msg.angular.z

    def odom_callback(self, msg):
        self.X_0T[0, :-1] = self.X_0T[0, 1:]
        self.X_0T[1, :-1] = self.X_0T[1, 1:]
        self.X_0T[2, :-1] = self.X_0T[2, 1:]
        self.X_0T[:, -1] = [self.x_pos, self.y_pos, self.yaw]

    def timer_callback(self, event):
        msg = Twist()
        latest_x_0t = self.X_0T[:, -1].reshape(-1, 1)
        yaw2 = float(self.yaw)
        x = float(self.robot_x) / 1000
        y = float(self.robot_y) / 1000
        R0 = Zonotope(
            np.array([x, y, np.cos(yaw2), np.sin(yaw2)]).reshape(4, 1),
            np.diag([0.3, 0.3, 0.01, 0.01]),
        )

        if self.parsed_plan is None:
            rospy.logwarn("No parsed plan received yet.")
            return

        plan = self.parsed_plan if self.parsed_plan is not None else self.old_safe_plan

        # plan = np.array(
        #     [
        #         [0.5, 0],
        #         [0.5, 0],
        #         [0.5, 0],
        #         [0.5, 0],
        #         [0.5, 0],
        #         [0.5, 0],
        #     ]
        # )

        current_position = np.array([x, y], dtype=np.float64).reshape(2, 1)
        print(f"X: {x}, Y: {y}")
        # Safety enforcement
        Reachable_Set_NL, obstacles, safety_check, plan_safe, GPT_Safety_Flag = (
            self.safety_layer.enforce_safety_nonlinear(
                R0, plan, self.Lidar_readings, -yaw2, current_position
            )
        )

        print(
            f"Len Reachable Set: {len(Reachable_Set_NL)}, Len Obstacles: {len(obstacles)}"
        )

        self.plot_data = (Reachable_Set_NL, obstacles)

        # print(f"Lidar readings: {self.Lidar_readings}")
        rospy.loginfo(f"Safety Check: {safety_check}")
        print(f"Plan Safe: {plan_safe}")
        reaching_radius = np.sqrt(
            (self.robot_x - self.goal_x) ** 2 + (self.robot_y - self.goal_y) ** 2
        )
        if reaching_radius < 200:
            rospy.loginfo("Reachability: Goal reached!")
            plan_safe[0, 0] = 0.0
            plan_safe[0, 1] = 0.0
            
        # Clip the linear velocity to be between 0 and 0.1
        if plan_safe[0, 0] < 0:
            plan_safe[0, 0] = 0
        elif plan_safe[0, 0] > 0.1:
            plan_safe[0, 0] = 0.1

        # plan_safe[1, 0] = np.clip(plan_safe[0, 1], 0, 0.1)
        msg.linear.x = float(plan_safe[0, 0])
        msg.angular.z = float(plan_safe[0, 1])

        self.cmd_vel_pub.publish(msg)

        # msg = Twist()

        # msg.linear.x = float(plan_safe[1, 0])
        # msg.angular.z = float(plan_safe[1, 1])
        # time.sleep(0.1)
        # self.cmd_vel_pub.publish(msg)

        self.old_safe_plan = plan_safe

        plt.cla()  # Clear previous plot

        def rotate_coordinates(coords):
            return np.array([coords[1], -coords[0]])

        # Plot obstacles
        for i, obs in enumerate(obstacles):
            center = obs.center()
            generators = obs.generators()
            rotated_center = rotate_coordinates(center)
            rotated_generators = np.array(
                [rotate_coordinates(gen) for gen in generators.T]
            ).T
            X0 = Zonotope(rotated_center.reshape(2, 1), rotated_generators)
            polygon_vertices = X0.polygon()
            weight = 1 + i * 0.5
            polygon_patch = Polygon(
                polygon_vertices.T,
                closed=True,
                facecolor="red",
                edgecolor="red",
                lw=weight,
            )
            plt.gca().add_patch(polygon_patch)

        # Plot reachable sets
        for i, reach in enumerate(Reachable_Set_NL):
            # plot the robot with green X
            if i == 0:
                center = reach.center()
                generators = reach.generators()
                rotated_center = rotate_coordinates(center[:2])
                rotated_generators = np.array(
                    [rotate_coordinates(gen) for gen in generators.T]
                ).T
                reachability_state = Zonotope(
                    rotated_center.reshape(2, 1), rotated_generators
                )
                polygon_vertices = reachability_state.polygon()
                polygon_patch = Polygon(
                    polygon_vertices.T,
                    closed=True,
                    facecolor="green",
                    edgecolor="green",
                    lw=2,
                    alpha=0.5,
                )
                plt.gca().add_patch(polygon_patch)
            else:
                
                center_2D = np.array(reach.center())
                generators_2D = np.array(reach.generators())
                rotated_center = rotate_coordinates(center_2D[:2])
                rotated_generators = np.array(
                    [rotate_coordinates(gen) for gen in generators_2D.T]
                ).T
                reachability_state_2D = Zonotope(
                    rotated_center.reshape(2, 1), rotated_generators
                )
                polygon_vertices = reachability_state_2D.polygon()
                color = "red" if i == 0 else plt.cm.Greens(i / len(Reachable_Set_NL))
                polygon_patch = Polygon(
                    polygon_vertices.T,
                    closed=True,
                    facecolor="none",
                    edgecolor="black",
                    lw=2,
                    alpha=0.5,
                )
                plt.gca().add_patch(polygon_patch)
            
        

        # Add legend containing obstacle and reachable set information and robot(green)
        plt.legend(["Obstacle", "Robot", "Reachable Set"])
        # Set plot limits and display
        plt.xlim(-4, 4)
        plt.ylim(-2, 2)
        # plt.gca().invert_yaxis()
        # plt.gca().invert_xaxis()
        plt.grid()
        plt.draw()
        plt.pause(0.0001)

        # # Plotting (unchanged from original)
        # self.ax.cla()
        # def rotate_coordinates(coords):
        #     return np.array([coords[1], -coords[0]])
        # for i in range(len(obstacles)):
        #     center = obstacles[i].center()
        #     generators = obstacles[i].generators()
        #     rotated_center = rotate_coordinates(center)
        #     rotated_generators = np.array([rotate_coordinates(gen) for gen in generators.T]).T
        #     X0 = Zonotope(rotated_center.reshape(2, 1), rotated_generators)
        #     polygon_vertices = X0.polygon()
        #     weight = 1 + i * 0.5
        #     polygon_patch = Polygon(polygon_vertices.T, closed=True, facecolor='none', edgecolor='red', lw=weight, alpha=1.0, linestyle='-')
        #     self.ax.add_patch(polygon_patch)
        # for i in range(len(Reachable_Set_NL)):
        #     X00 = Reachable_Set_NL[i]
        #     center_2D = np.array(X00.center())
        #     generators_2D = np.array(X00.generators())
        #     rotated_center = rotate_coordinates(center_2D[:2])
        #     rotated_generators = np.array([rotate_coordinates(gen) for gen in generators_2D.T]).T
        #     reachability_state_2D = Zonotope(rotated_center.reshape(2, 1), rotated_generators)
        #     polygon_vertices = reachability_state_2D.polygon()
        #     color = 'red' if i == 0 else plt.cm.Greens(i / len(Reachable_Set_NL))
        #     line = '-' if i == 0 else '--'
        #     polygon_patch = Polygon(polygon_vertices.T, closed=True, facecolor=color, edgecolor='black', lw=2, alpha=0.5, linestyle=line)
        #     self.ax.add_patch(polygon_patch)
        # self.ax.set_xlim(-5, 5)
        # self.ax.set_ylim(2, 8)
        # self.ax.invert_yaxis()
        # self.ax.invert_xaxis()
        # self.ax.grid()
        # plt.draw()
        # plt.pause(0.01)

    # Callback function for subscriber
    def callback(self, msg):
        current_time = rospy.get_time()
        self.time_data.append(current_time)
        self.value_data.append(msg.data)

        # Limit the number of points shown (for performance)
        if len(self.time_data) > 100:
            self.time_data.pop(0)
            self.value_data.pop(0)

    # Function to update plot
    def update_plot(self, frame):
        plt.cla()
        plt.plot(self.time_data, self.value_data, label="Sensor Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend(loc="upper right")
        plt.title("Real-Time Data Plot")

    # Main function
    def plot_node(self):
        rospy.Subscriber("/sensor_data", Float32, self.callback)

        # Set up non-blocking plot
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        while not rospy.is_shutdown():
            ax.clear()
            ax.plot(self.time_data, self.value_data, label="Sensor Data")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value")
            ax.legend(loc="upper right")
            ax.set_title("Real-Time Data Plot")
            plt.pause(0.1)
        plt.ioff()  # Disable interactive mode when node is shut down
        plt.show()


def main():
    node = DataLoggerNode()
    # rospy.spin()
    # node.plot_node()


if __name__ == "__main__":
    try:
        node = DataLoggerNode()
        # node.plot_node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
