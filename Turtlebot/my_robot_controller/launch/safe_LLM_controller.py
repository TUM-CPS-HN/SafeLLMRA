from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='chat_gpt',
            executable='chat_gpt_service',
            name='chat_gpt_service'
        ),


        Node(
            package='my_robot_controller',
            executable='LLM_robot_controller',
            name='LLM_robot_controller'
        ),

        Node(
            package='my_robot_controller',
            executable='safety_check_nonlinear_reachability',
            name='safety_check_nonlinear_reachability'
        ),
    ])
