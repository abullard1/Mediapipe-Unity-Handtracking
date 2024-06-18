# Author: Samuel Ruairi Bullard
# GitHub: https://github.com/abullard1
# Description:
# This script contains the UnitySocket class that is used to handle the socket connection between the handtracking.py
# script and Unity. The class provides methods to format the hand landmarks and handedness data into a JSON string and
# send it to Unity. It also includes a callback function that receives the hand landmarks and handedness data from the
# MediaPipe Handtracking callback and sends it to Unity. Additionally, the class includes methods to format the table
# dimensions into a JSON string and send it to Unity, as well as a callback function for sending the table dimensions to
# Unity. The UnitySocket class is designed to be used in conjunction with the MediaPipe Handtracking library to send hand
# tracking data to Unity for real-time interaction of objects using one's hands, with objects on a TUI table in the context
# of interactive collaborative file-sharing.

import json
import socket


# Class that handles the socket connection to Unity
class UnitySocket:
    def __init__(self, host="localhost", port=8080):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Formats the hand landmarks and handedness data into a JSON string, ready to be sent to Unity
    def format_hand_data(self, hand_landmarks_list, handedness_list):
        hands_data = []
        for hand_index, (handedness, landmarks) in enumerate(zip(handedness_list, hand_landmarks_list)):
            hand_data = {
                "handID": hand_index,
                "handedness": handedness[0].category_name if handedness else "Unknown",
                "landmarks": [
                    # Rounds the landmark coordinates to 4 decimal places (for performance reasons -> smaller data size)
                    # Rest of the digits not needed, four decimal places precision is enough for hand tracking
                    {"id": index, "x": round(landmark.x, 4), "y": round(landmark.y, 4), "z": round(landmark.z, 4)}
                    for index, landmark in enumerate(landmarks)
                ],
            }
            hands_data.append(hand_data)
        return json.dumps({"hands": hands_data})

    # Sends the formatted JSON string to Unity
    def send_data(self, data):
        try:
            self.socket.sendto(data.encode(), (self.host, self.port))
        except Exception as e:
            print(f"Error sending data: {e}")

    # Closes the socket connection
    def close(self):
        self.socket.close()

    # Callback that receives the hand landmarks and handedness data from the MediaPipe Handtracking callback
    def landmark_result_second_callback(self, hand_landmarks_list, handedness_list):
        try:
            formatted_json = self.format_hand_data(hand_landmarks_list, handedness_list)
            self.send_data(formatted_json)
        except Exception as e:
            print(f"An error occurred: {e}")

    # Formats the table dimensions into a JSON string ready to be sent to Unity
    # todo: Implement Unity Camera Size Modification in Unity (based on the table dimensions)
    def format_table_dimensions(self, top_left, bottom_right):
        return json.dumps({
            "top_left": {"x": top_left[0], "y": top_left[1]},
            "bottom_right": {"x": bottom_right[0], "y": bottom_right[1]},
        })

    # Callback for sending the table dimensions (top_left_x, top_left_y, bottom_right_x, bottom_right_y) to Unity
    # todo: Put into handtracking.py mediapipe callback
    def table_dimensions_callback(self, top_left, bottom_right):
        try:
            formatted_json = self.format_table_dimensions(top_left, bottom_right)
            self.send_data(formatted_json)
        except Exception as e:
            print(f"An error occurred: {e}")
