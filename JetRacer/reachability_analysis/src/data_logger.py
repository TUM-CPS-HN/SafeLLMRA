#!/home/jetson/ros_venv/bin/python3
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
from collections import deque

class DataLoggerNode:
    def __init__(self):
        # Initialize the ROS 1 node
        rospy.init_node('data_logger', anonymous=True)

        # Set up subscribers for sensor data
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.laser_scan_callback, queue_size=10)
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=10)

        # Buffers to store incoming data
        self.cmd_vel_buffer = deque(maxlen=100)
        self.odom_buffer = deque(maxlen=100)
        self.ranges_buffer = deque(maxlen=100)

        # Timer for periodic processing (e.g., every 0.25 seconds)
        rospy.Timer(rospy.Duration(0.25), self.timer_callback)

        # Log that the node has started
        rospy.loginfo('Data Logger Node started.')

    def laser_scan_callback(self, msg):
        """Callback for LiDAR scan data."""
        self.ranges_buffer.append(np.array(msg.ranges))

    def cmd_vel_callback(self, msg):
        """Callback for velocity commands."""
        self.cmd_vel_buffer.append([msg.linear.x, msg.angular.z])

    def odom_callback(self, msg):
        """Callback for odometry data."""
        position = msg.pose.pose.position
        self.odom_buffer.append([position.x, position.y, position.z])

    def timer_callback(self, event):
        """Periodic callback to process or log data."""
        rospy.loginfo("Processing data...")
        # Example: Print buffer sizes (replace with actual analysis or logging)
        rospy.loginfo(f"Ranges buffer size: {len(self.ranges_buffer)}")
        rospy.loginfo(f"Cmd vel buffer size: {len(self.cmd_vel_buffer)}")
        rospy.loginfo(f"Odom buffer size: {len(self.odom_buffer)}")

    def run(self):
        """Keep the node running."""
        rospy.spin()

if __name__ == '__main__':
    try:
        node = DataLoggerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
