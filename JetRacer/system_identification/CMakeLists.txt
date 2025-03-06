cmake_minimum_required(VERSION 2.8.3)
project(system_identification)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy geometry_msgs nav_msgs std_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

install(PROGRAMS
  system_identification/data_logger_turtulebot_dynamic.py
  system_identification/random_cmd_turtlebot.py
  system_identification/plot.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY system_identification
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)