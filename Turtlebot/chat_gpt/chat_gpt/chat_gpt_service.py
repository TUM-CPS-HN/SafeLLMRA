import rclpy
from rclpy.node import Node
from custom_msgs.srv import ChatPrompt
import openai

class ChatGPTServer(Node):

    def __init__(self):
        super().__init__('chat_gpt_server')

        # Set your OpenAI API key
        self.api_key = 'Write Your APi key here'
        openai.api_key = self.api_key

        # Create a ROS 2 service
        self.srv = self.create_service(ChatPrompt, 'chat_gpt_ask', self.service_callback)

    def service_callback(self, request, response):
        # Print the incoming prompt
        self.get_logger().info(f'Received prompt: {request.prompt}')
        
        # Get the ChatGPT response
        response.response = self.get_chat_gpt_response(request.prompt)

        return response

    def get_chat_gpt_response(self, prompt):
        try:
            # Make a call to OpenAI's API
            response = openai.ChatCompletion.create(
                #model='gpt-3.5-turbo',
                model='gpt-4o',

                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract and print the response
            chat_response = response.choices[0].message.content
            self.get_logger().info(f'ChatGPT response: {chat_response}')
            
            return chat_response
        except Exception as e:
            self.get_logger().error(f'Error calling OpenAI API: {e}')
            return "Error retrieving response from ChatGPT."

def main(args=None):
    rclpy.init(args=args)
    chat_gpt_node = ChatGPTServer()
    rclpy.spin(chat_gpt_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
