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
        self.buffer_size = 500
        self.cmd_vel_buffer = deque(maxlen=self.buffer_size)
        self.odom_buffer = deque(maxlen=self.buffer_size)
        self.ranges_buffer = deque(maxlen=self.buffer_size)
        self.U_full = np.zeros((2, self.buffer_size))
        self.X_0T = np.zeros((3, self.buffer_size))
        self.X_1T = np.zeros((3, self.buffer_size))
        self.data_collected = False
        rospy.loginfo('Data Logger Node Initialized.')

    def laser_scan_callback(self, msg):
        if self.data_collected:
            return
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
        self.check_data_collected()

    def cmd_vel_callback(self, msg):
        if self.data_collected:
            return
        self.cmd_vel_buffer.append((msg.linear.x, msg.linear.y, msg.angular.z))
        self.check_data_collected()

    def odom_callback(self, msg):
        if self.data_collected:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom_buffer.append((x, y, yaw))
        self.check_data_collected()

    def check_data_collected(self):
        if (len(self.cmd_vel_buffer) == self.buffer_size and 
            len(self.odom_buffer) == self.buffer_size and 
            len(self.ranges_buffer) == self.buffer_size):
            self.data_collected = True
            self.timer_callback()

    def timer_callback(self):
        rospy.loginfo('Timer Callback Executed.')
        linear_x = [cmd[0] for cmd in self.cmd_vel_buffer]
        angular_z = [cmd[2] for cmd in self.cmd_vel_buffer]
        self.U_full[0, :] = linear_x
        self.U_full[1, :] = angular_z
        x_positions = [odom[0] for odom in self.odom_buffer]
        y_positions = [odom[1] for odom in self.odom_buffer]
        yaws = [odom[2] for odom in self.odom_buffer]
        self.X_0T[0, :] = x_positions
        self.X_0T[1, :] = y_positions
        self.X_0T[2, :] = yaws
        self.X_1T[0, :] = self.U_full[0, :]
        self.X_1T[1, :] = np.zeros_like(self.U_full[0, :])
        self.X_1T[2, :] = self.U_full[1, :]
        np.save('U_full_NL.npy', self.U_full)
        np.save('X_0T_NL.npy', self.X_0T)
        np.save('X_1T_NL.npy', self.X_1T)
        rospy.loginfo('Data saved as U_full.npy, X_0T.npy, X_1T.npy.')
        rospy.signal_shutdown('Data collection complete.')

def main():
    node = DataLoggerNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()