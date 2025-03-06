cmake_minimum_required(VERSION 2.8.3)
project(reachability_analysis)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs geometry_msgs nav_msgs std_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

install(PROGRAMS
  src/data_logger.py
  src/data_logger_turtulebot_dynamic.py
  src/random_cmd_turtlebot.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY src/example
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/example
)