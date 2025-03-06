# JetRacer

## Requirements
- Ubuntu 18.04
- ROS Melodic
- Python 3.7+
- OpenAI Python package


## How to run
1. Run the following commands:
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
roslaunch my_robot_controller safe_LLM_contoller
```

2. In a new terminal, run the reachability analysis:
```bash
python my_robot_controller/safety_check_nonlinear_reachability.py
```

# Turtlebot3 
First, the following OS and software need to be installed:
- Ubuntu 22.04.5 LTS
- ROS2 Humble
- Gazebo 11.10.2
- TurtleBot3

Then, the following commands need to be executed to install required dependencies:
- source /opt/ros/humble/setup.bash
- sudo apt update
- sudo apt upgrade
- sudo apt-get install gedit
- sudo apt install ros-humble-joint-state-publisher
- sudo apt install ros-humble-joint-state-publisher-gui
- sudo apt install ros-humble-xacro
- sudo apt install ros-humble-ros-gz
- sudo apt install ros-humble-gazebo-ros-pkgs
- sudo apt install ros-humble-ros-core
- sudo apt install ros-humble-geometry2
- sudo apt install ros-humble-gazebo-msgs
- sudo apt install ros-humble-gazebo-plugins
- sudo apt install ros-humble-ros-ign-bridge
- sudo apt install ros-humble-teleop-twist-keyboard
- sudo apt install ros-humble-teleop-twist-keyboard
- pip install openai==0.28
- cd Turtlebot/pyzonotope && pip install .

How to run: 
- source ~/<<"workspace">>/install/setup.bash
- In the path "src/my_robot_controller/my_robot_controller/LLM_robot_controller.py" write you api key
- colcon build
- 
- Run "export TURTLEBOT3_MODEL=waffle_pi" and " ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py" in a terminal.
- Then run "ros2 run chat_gpt chat_gpt_service" and "ros2 run my_robot_controller LLM_robot_controller" and "ros2 run my_robot_controller safety_check_nonlinear_reachability" in three terminals.

To prompt the ChatGPT use the following command:
- ros2 service call /chat_gpt_ask custom_msgs/srv/ChatPrompt "prompt: 'Your should write here'"

References: 
- Differential robot: https://www.youtube.com/watch?v=V9ztoMuSX8w
- ChatGPT: https://www.youtube.com/watch?v=xfggW8OuHDQ
- ROS2 Tutorial: https://www.youtube.com/watch?v=vCTbUgw6k8U&list=PLLSegLrePWgJudpPUof4-nVFHGkB62Izy&index=12
- ROS2 Humble: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
- Gazebo 11.10.2 : https://classic.gazebosim.org/tutorials?tut=install_ubuntu
- Turtlebot3 : https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/






