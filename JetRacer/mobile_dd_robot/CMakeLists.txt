cmake_minimum_required(VERSION 2.8.3)
project(mobile_dd_robot)
find_package(catkin REQUIRED)
catkin_package()
include_directories(${catkin_INCLUDE_DIRS})
install(DIRECTORY launch model
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)