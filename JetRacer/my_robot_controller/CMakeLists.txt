cmake_minimum_required(VERSION 2.8.3)
project(my_robot_controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  custom_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy geometry_msgs nav_msgs sensor_msgs std_msgs custom_msgs
)

install(PROGRAMS
  my_robot_controller/LLM_robot_controller.py
  my_robot_controller/safety_check_nonlinear_reachability.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)