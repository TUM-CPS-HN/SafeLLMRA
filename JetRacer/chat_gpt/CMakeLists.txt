cmake_minimum_required(VERSION 2.8.3)
project(chat_gpt)

find_package(catkin REQUIRED COMPONENTS
  rospy
  custom_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy custom_msgs std_msgs
)

install(PROGRAMS
  chat_gpt/chat_gpt_service.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)