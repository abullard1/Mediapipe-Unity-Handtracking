# Author: Samuel Ruairi Bullard
# GitHub: https://github.com/abullard1
# Description:
# This script contains the HandsLandmarker class that is used to detect hand landmarks using the MediaPipe Hand Landmarker
# model. The class provides methods to detect hand landmarks in a frame, visualize the landmarks and handedness, and
# calculate the frames per second (FPS) of the video feed. The HandsLandmarker class is designed to be used in conjunction
# with the UnitySocket class to send the hand landmarks and handedness data to Unity for interaction on a TUI table in the context
# of interactive collaborative file-sharing.

import threading
import queue
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import table_calibration

import unity_socket

# Hand Landmark Detection Configuration
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode

# Constants
# Path to the Hand Landmarker model file
# Model Source: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
HAND_LANDMARKER_MODEL_PATH = "./mediapipe_hand_landmarker_model/hand_landmarker.task"

# RESOLUTION has to be set to something reasonable for the camera.
# If the camera does not support the resolution, it will default to the closest supported resolution.
# So if your camera is marketed as supporting a full HD 1920x1080 Resolution,
# and you have the hardware power available to run the script at that resolution,
# then set it to that. If you are unsure, 1280x720 is a good default value for most webcams.
RESOLUTION = (1280, 720)  # (Width, Height) -> (16:9 HD)

# Handedness Text Constants
HANDEDNESS_MARGIN = 10  # (Pixels)
HANDEDNESS_FONT_SIZE = 1
HANDEDNESS_FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # (Green)

# FPS Text Constants
FPS_FONT_SIZE = 1
FPS_FONT_THICKNESS = 1
FPS_TEXT_COLOR = (255, 255, 255)  # (White)

# Performance Constants
# (Note: Camera Hardware also has a limit, so if the camera's FPS is lower than this, it will be the limiting factor)
FPS_LIMIT = 60  # Maximum FPS
FPS_AVERAGE_AMOUNT = 10  # Number of frames included in the average FPS calculation
SHOW_ANNOTATIONS = True  # Flag to show landmarks and handedness visualizations. Can be toggled using the space bar
CROP_MODE = False  # Flag to crop the frame to the table calibration area


# Hand Landmarker Class
class HandsLandmarker:
    # Constructor
    def __init__(self, model_path=None, socket=None) -> None:

        # todo: Implement GPU acceleration for MediaPipe Hand Landmarker using Linux and the delegate property
        # Hand Landmarker Configuration/Options
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,  # Maximum number of hands to detect
            min_hand_detection_confidence=0.3,  # Minimum confidence to detect a hand
            min_hand_presence_confidence=0.5,  # Minimum confidence to determine if a hand is present
            min_tracking_confidence=0.3,
            result_callback=self.result_callback  # Callback function for the detection result
        )
        self.detector = HandLandmarker.create_from_options(options)

        # Camera Setup
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

        # Temporary Variables / Data Members
        self.current_frame = None
        self.frame_timestamps = []
        self.socket = socket
        self.table_calibration = table_calibration.TableCalibration()
        self.hand_landmarks_list = None
        self.handedness_list = None

        # Checks if a camera is found
        if not self.video_capture.isOpened():
            print("Error: Camera not found")
            exit()
        else:
            print(f"Camera resolution: {self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)}x"
                  f"{self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    # Retrieves the current frame from the camera
    def get_frame_data(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            return frame, np.array(frame)
        else:
            print("Error: Frame not found")
            return None, None

    # Converts a numpy array to a MediaPipe Image
    def np_array_to_mp_image(self, np_array):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np_array)

    # Gets the current time in milliseconds
    def get_current_time_ms(self):
        return int(time.time() * 1000)

    # Detects hand landmarks in the current frame
    def detect_hand_landmarks(self):
        frame, np_array_frame = self.get_frame_data()
        if frame is None:
            return
        mp_image = self.np_array_to_mp_image(np_array_frame)
        current_timestamp_ms = self.get_current_time_ms()
        self.detector.detect_async(mp_image, round(current_timestamp_ms))

    # Runs the hand landmarker in a loop
    def run_threads(self):
        frame_queue = queue.Queue()
        visualization_queue = queue.Queue()

        # Capture Thread to get the frame data
        # Note: Separating frame capture, detection and frame display (Main Thread) into different threads can improve performance and will allow the
        # video feed to be smooth regardless of the detection speed.
        def capture_thread_function():
            while True:
                frame, np_array_frame = self.get_frame_data()
                if frame is not None:
                    frame_queue.put((frame, np_array_frame))  # Queues the frame for detection

        # Detection Thread to detect hand landmarks and aruco markers and visualize the results
        def detection_thread_function():
            detection_interval = 1.0 / FPS_LIMIT  # Detection interval based on the FPS limit
            last_detection_time = 0

            while True:
                original_frame, np_array_frame = frame_queue.get()  # Gets the frame from the frame queue
                current_time = time.perf_counter()
                self.calculate_and_display_fps(original_frame, current_time,
                                               FPS_AVERAGE_AMOUNT)  # Calculates and displays the feeds FPS

                if current_time - last_detection_time > detection_interval:  # If the detection interval has passed...
                    # Detects ArUco markers on the full frame to get the cropping coordinates
                    corners, marker_ids, markers_detected = self.table_calibration.detect_aruco_markers(original_frame)

                    # Initializes frame_for_mediapipe with the full frame
                    frame_for_mediapipe = original_frame

                    # Crops the frame if in CROP_MODE and valid coordinates are available
                    if CROP_MODE and self.table_calibration.top_left_avg is not None and self.table_calibration.bottom_right_avg is not None:
                        top_left_avg = self.table_calibration.top_left_avg
                        bottom_right_avg = self.table_calibration.bottom_right_avg

                        top_left_x = min(top_left_avg[0], bottom_right_avg[0])
                        bottom_right_x = max(top_left_avg[0], bottom_right_avg[0])
                        top_left_y = min(top_left_avg[1], bottom_right_avg[1])
                        bottom_right_y = max(top_left_avg[1], bottom_right_avg[1])

                        # Calculates the offset to adjust the Aruco marker positions
                        offset_x = top_left_x
                        offset_y = top_left_y

                        # Applies the offset to the Aruco marker positions
                        corners = [corner - np.array([offset_x, offset_y]) for corner in corners]

                        # Crops the frame based on the calculated coordinates
                        if bottom_right_x - top_left_x > 0 and bottom_right_y - top_left_y > 0:
                            # Update frame_for_mediapipe to the cropped area for MediaPipe detection
                            frame_for_mediapipe = original_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                    # Converts the frame_for_mediapipe to MediaPipe format and runs the detection
                    mp_image = self.np_array_to_mp_image(np.array(frame_for_mediapipe))
                    current_timestamp_ms = self.get_current_time_ms()
                    self.detector.detect_async(mp_image,
                                               round(current_timestamp_ms))  # Asynchronous MediaPipe detection

                    if SHOW_ANNOTATIONS and self.hand_landmarks_list is not None and self.handedness_list is not None:
                        # Ensures that we have the corresponding handedness information for each set of landmarks
                        min_length = min(len(self.hand_landmarks_list), len(self.handedness_list))
                        self.visualize_hand_landmarks(frame_for_mediapipe,
                                                      self.hand_landmarks_list[:min_length],
                                                      self.handedness_list[:min_length])

                    # Applies the Aruco marker visualizations based on the Aruco detection results
                    if SHOW_ANNOTATIONS and markers_detected:
                        frame_for_mediapipe = self.table_calibration.visualize_aruco_markers(frame_for_mediapipe,
                                                                                             corners, marker_ids,
                                                                                             markers_detected)

                    # Updates frame_to_display based on whether CROP_MODE is active
                    frame_to_display = frame_for_mediapipe

                    last_detection_time = current_time

                # Always puts a frame into the queue for visualization
                visualization_queue.put(frame_to_display)

        # Converts the thread functions to a thread object (Deamon Thread will close when the main thread closes)
        capture_thread = threading.Thread(target=capture_thread_function, daemon=True)
        detection_thread = threading.Thread(target=detection_thread_function, daemon=True)

        # Starts the capture and detection threads
        capture_thread.start()
        detection_thread.start()

        # Main Loop
        while True:
            # Gets the frame from the visualization queue and displays it in a window
            if not visualization_queue.empty():
                frame = visualization_queue.get()
                cv2.imshow("FSMMI Tangible Interaction: Drag & Drop", frame)

            # Waits for a key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # If the "q" key is pressed, the program will close (ends the main loop)
                self.close()
                break
            elif key == ord(" "):  # If the space bar is pressed, the annotations will be toggled
                global SHOW_ANNOTATIONS
                SHOW_ANNOTATIONS = not SHOW_ANNOTATIONS
            elif key == ord("c"):  # If the "c" key is pressed, the program will toggle the CROP_MODE
                global CROP_MODE
                CROP_MODE = not CROP_MODE

        self.close()  # Closes the program after the main loop ends

    # Function to calculate and display the Video Feeds FPS on the frame
    def calculate_and_display_fps(self, frame, current_time, fps_average_amount=10):
        # Adds the current time to the frame_timestamps list
        # and removes the first element if the list longer than fps_average_amount
        self.frame_timestamps.append(current_time)
        if len(self.frame_timestamps) > fps_average_amount:
            self.frame_timestamps.pop(0)

        # Calculates the average frame time and FPS
        if len(self.frame_timestamps) > 1:
            avg_frame_time = (self.frame_timestamps[-1] - self.frame_timestamps[0]) / (
                    len(self.frame_timestamps) - 1)
            fps = 1 / avg_frame_time if avg_frame_time > 0 else 0

            # Displays the FPS on the frame
            fps_text = f"FPS: {fps:.2f}"  # Formats the FPS to two decimal places

            # Configures the text properties (e.g., font, size, color, thickness) and displays the text on the frame
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, FPS_FONT_SIZE,
                        FPS_TEXT_COLOR, FPS_FONT_THICKNESS, cv2.LINE_AA)

    # Closes the camera and the window
    def close(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
        self.detector.close()
        print("Hand Landmarker Closed")

    # Callback function for the hand landmarker asynchronous detection result
    def result_callback(self, detection_result, rgb_image, _):
        print(detection_result)

        # Updates the hand landmarks data
        self.hand_landmarks_list = detection_result.hand_landmarks
        print(self.hand_landmarks_list)
        # Updates the handedness data
        self.handedness_list = detection_result.handedness

        # Sends the hand landmarks and handedness data to the Unity socket
        if self.socket is not None:
            self.socket.landmark_result_second_callback(self.hand_landmarks_list, self.handedness_list)

    # Visualizes the hand landmarks and handedness detection results
    def visualize_hand_landmarks(self, annotated_frame, hand_landmarks_list, handedness_list):

        # Iterates through the results
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draws the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            # Draws the hand landmarks and handedness on the frame using MediaPipe drawing utilities
            solutions.drawing_utils.draw_landmarks(
                annotated_frame,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Gets the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_frame.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - HANDEDNESS_MARGIN

            # Configures and draws the handedness text (left or right hand) on the frame.
            cv2.putText(annotated_frame, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        HANDEDNESS_FONT_SIZE, HANDEDNESS_TEXT_COLOR, HANDEDNESS_FONT_THICKNESS, cv2.LINE_AA)


if __name__ == "__main__":
    # Unity Socket (if needed)
    local_unity_socket = unity_socket.UnitySocket()

    # Hand Landmarker Instance
    hand_landmarker = HandsLandmarker(model_path=HAND_LANDMARKER_MODEL_PATH, socket=local_unity_socket)
    hand_landmarker.run_threads()
