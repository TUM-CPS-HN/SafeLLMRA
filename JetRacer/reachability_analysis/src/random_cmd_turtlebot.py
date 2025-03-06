#!/home/jetson/ros_venv/bin/python3
import rospy
from geometry_msgs.msg import Twist
import random
import time

class RandomTurtlebot3Command:
    def __init__(self):
        rospy.init_node('random_turtlebot3_commander', anonymous=True)
        self.publisher_ = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.timer_period = 0.1
        rospy.Timer(rospy.Duration(self.timer_period), self.send_random_command)
        rospy.loginfo('Random TurtleBot3 Commander Node Initialized!')

    def send_random_command(self, event):
        twist = Twist()
        twist.linear.x = random.uniform(0.0, 0.25)
        twist.angular.z = random.uniform(-0.5, 0.5)
        self.publisher_.publish(twist)
        rospy.loginfo(f'Published command: linear.x={twist.linear.x:.2f}, angular.z={twist.angular.z:.2f}')

def main():
    node = RandomTurtlebot3Command()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down...')
        rospy.signal_shutdown('User interrupted')

if __name__ == '__main__':
    main()