#!/home/jetson/ros_venv/bin/python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import numpy as np
import os

class DataLoggerNode:
    def __init__(self):
        rospy.init_node('data_logger', anonymous=True)
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=10)
        self.data_array = []
        rospy.Timer(rospy.Duration(1.0), self.timer_callback)
        self.last_cmd_vel = None
        self.last_odom = None
        rospy.loginfo('Data Logger Node Initialized.')

    def cmd_vel_callback(self, msg):
        self.last_cmd_vel = msg

    def odom_callback(self, msg):
        self.last_odom = msg

    def timer_callback(self, event):
        if self.last_odom is not None and self.last_cmd_vel is not None:
            x = self.last_odom.pose.pose.position.x
            y = self.last_odom.pose.pose.position.y
            q = self.last_odom.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            linear_velocity = self.last_cmd_vel.linear.x
            angular_velocity = self.last_cmd_vel.angular.z
            current_time = rospy.get_time()
            self.data_array.append([current_time, x, y, yaw, linear_velocity, angular_velocity])
            rospy.loginfo(f'Data Logged: {current_time}, {x}, {y}, {yaw}, {linear_velocity}, {angular_velocity}')
            if len(self.data_array) >= 501:
                self.save_data()

    def save_data(self):
        save_directory = os.path.expanduser('~/catkin_ws/src/system_identification/system_identification/')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        np.save(os.path.join(save_directory, 'real_time_data.npy'), np.array(self.data_array))
        if os.path.exists(os.path.join(save_directory, 'real_time_data.npy')):
            rospy.loginfo('✅ real_time_data.npy saved successfully.')
        else:
            rospy.logerr('❌ Failed to save real_time_data.npy.')
        self.data_array = []

def main():
    node = DataLoggerNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()