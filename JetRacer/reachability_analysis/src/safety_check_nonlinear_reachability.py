#!/home/jetson/ros_venv/bin/python3
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from collections import deque
import numpy as np
from pyzonotope.Zonotope import Zonotope
from pyzonotope.MatZonotope import MatZonotope
from pyzonotope.reachability_analysis import get_AB
from pyzonotope.SafetyLayer import SafetyLayer

class DataLoggerNode:
    def __init__(self):
        rospy.init_node('data_logger', anonymous=True)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.laser_scan_callback, queue_size=10)
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=10)
        self.cmd_vel_buffer = deque(maxlen=100)
        self.odom_buffer = deque(maxlen=100)
        self.ranges_buffer = deque(maxlen=100)
        self.Lidar_readings = np.zeros((1, 5))
        self.U_full = np.zeros((2, 100))
        self.X_0T = np.zeros((3, 100))
        self.X_1T = np.zeros((3, 100))
        rospy.Timer(rospy.Duration(0.25), self.timer_callback)
        rospy.loginfo('Data Logger Node (Python) started.')

    def laser_scan_callback(self, msg):
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        angle_range_start = -math.pi / 4
        angle_range_end = math.pi / 4
        angle_step = math.radians(10)
        index_min = int((angle_range_start - angle_min) / angle_increment)
        index_max = int((angle_range_end - angle_min) / angle_increment)
        selected_ranges = []
        for i in range(index_min, index_max + 1):
            if i % int(angle_step / angle_increment) == 0:
                range_value = msg.ranges[i]
                if not math.isinf(range_value) and not math.isnan(range_value):
                    selected_ranges.append(range_value)
        self.ranges_buffer.append(list(selected_ranges))
        self.Lidar_readings = np.array(selected_ranges)[:5]

    def cmd_vel_callback(self, msg):
        self.linear_x = msg.linear.x
        self.linear_y = msg.linear.y
        self.angular_z = msg.angular.z
        self.cmd_vel_buffer.append((self.linear_x, self.linear_y, self.angular_z))
        if len(self.cmd_vel_buffer) > 0:
            linear_x_values = [cmd[0] for cmd in self.cmd_vel_buffer]
            angular_z_values = [cmd[2] for cmd in self.cmd_vel_buffer]
            self.U_full[0, :len(linear_x_values)] = linear_x_values
            self.U_full[1, :len(angular_z_values)] = angular_z_values

    def odom_callback(self, msg):
        self.x_pos = msg.pose.pose.position.x
        self.y_pos = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        self.odom_buffer.append((self.x_pos, self.y_pos, self.yaw))
        if len(self.odom_buffer) > 0:
            x_values = [odom[0] for odom in self.odom_buffer]
            y_values = [odom[1] for odom in self.odom_buffer]
            yaw_values = [odom[2] for odom in self.odom_buffer]
            self.X_0T[0, :len(x_values)] = x_values
            self.X_0T[1, :len(y_values)] = y_values
            self.X_0T[2, :len(yaw_values)] = yaw_values
        self.X_1T[0, :-1] = self.X_1T[0, 1:]
        self.X_1T[1, :-1] = self.X_1T[1, 1:]
        self.X_1T[2, :-1] = self.X_1T[2, 1:]
        self.X_1T[0, -1] = msg.twist.twist.linear.x
        self.X_1T[1, -1] = msg.twist.twist.linear.y
        self.X_1T[2, -1] = msg.twist.twist.angular.z

    def timer_callback(self, event):
        if len(self.cmd_vel_buffer) > 0 and len(self.odom_buffer) > 0 and len(self.ranges_buffer) > 0:
            cmd_vel_data = self.cmd_vel_buffer[-1]
            odom_data = self.odom_buffer[-1]
            ranges_data = self.ranges_buffer[-1]
            dim_x = 3
            dim_u = 2
            dt = 0.05
            initpoints = 100
            steps = 1
            totalsamples = initpoints * steps
            W = Zonotope(np.array(np.zeros((dim_x, 1))), 0.005 * np.ones((dim_x, 1)))
            GW = []
            for i in range(W.generators().shape[1]):
                vec = np.reshape(W.Z[:, i + 1], (dim_x, 1))
                dummy = []
                dummy.append(np.hstack((vec, np.zeros((dim_x, totalsamples - 1)))))
                for j in range(1, totalsamples, 1):
                    right = np.reshape(dummy[i][:, 0:j], (dim_x, -1))
                    left = dummy[i][:, j:]
                    dummy.append(np.hstack((left, right)))
                GW.append(np.array(dummy))
            GW = np.array(GW)
            Wmatzono = MatZonotope(np.zeros((dim_x, totalsamples)), GW)
            AB = get_AB(self.U_full, self.X_0T, self.X_1T, Wmatzono)
            total_steps = 4
            latest_x_0t = self.X_0T[:, -1].reshape(-1, 1)
            X0 = Zonotope(latest_x_0t, 0.1 * np.diag(np.ones((dim_x, 1)).T[0]))
            U = Zonotope(np.array([0, 0]).reshape(-1, 1), [[0.25, 0], [0, 0.25]])
            center_2D = X0.center()
            generators_2D = X0.generators()
            new_center = center_2D[:2, :]
            new_generators = generators_2D[:2, :]
            reachability_state_2D = [Zonotope(new_center, new_generators)]
            reachability_state = [X0]
            for i in range(total_steps):
                reachability_state[i] = reachability_state[i].reduce('girard', 100)
                new_state = AB * reachability_state[i].cart_prod(U) + W
                reachability_state.append(new_state.reduce('girard', 1))
                reachability_state_2D[i] = reachability_state_2D[i].reduce('girard', 100)
                center_2D = new_state.center()
                generators_2D = new_state.generators()
                new_center = center_2D[:2, :]
                new_generators = generators_2D[:2, :]
                new_state_2D = Zonotope(new_center, new_generators)
                reachability_state_2D.append(new_state_2D)
            Safety_layer = SafetyLayer()
            plan = np.array([[.4, .5], [.7, .8], [.10, .11]])
            Safety_chack, obstacles = Safety_layer.enforce_safety(reachability_state_2D, plan, self.Lidar_readings)
            rospy.loginfo(f"Safety Check: {Safety_chack}")

def main():
    node = DataLoggerNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()