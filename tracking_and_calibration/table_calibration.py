# Author: Samuel Ruairi Bullard
# GitHub: https://github.com/abullard1
# Description:
# This script contains the TableCalibration class that is used to calibrate the table/tracking area by detecting
# Aruco Markers. The class provides methods to generate Aruco Markers, detect the markers in an image, and visualize
# the markers by drawing a rectangle around them. The TableCalibration class is designed to be used in conjunction with
# the MediaPipe Handtracking library to calibrate the tracking area for hand tracking in Unity.

import os
import cv2
from cv2 import aruco
import numpy as np


# Class that handles the calibration of the table/tracking area by detecting Aruco Markers
class TableCalibration:

    # Initializes the TableCalibration object with the specified parameters
    # (Adjust smoothing_frames for smoother tracking)
    def __init__(self, aruco_type=aruco.DICT_4X4_100, corner_ids=None, persistence_frames=1, smoothing_frames=1):
        self.aruco_type = aruco_type
        self.corner_ids = corner_ids or {"top_left": 1, "bottom_right": 0}  # Default Aruco Marker IDs
        self.persistence_frames = persistence_frames
        self.persistence_counter = 0
        self.smoothing_frames = smoothing_frames
        self.corner_history = {"top_left": [], "bottom_right": []}
        self.top_left_avg = None
        self.bottom_right_avg = None

        # Initializes the Aruco Detector with optimized parameters
        self.dic = aruco.getPredefinedDictionary(self.aruco_type)
        self.aruco_detector = aruco.ArucoDetector(self.dic)
        self.parameters = self.aruco_detector.getDetectorParameters()
        self._set_detector_parameters()

    # Sets the detector parameters for the Aruco Detector (Removed for simplicity, because not needed for this task)
    def _set_detector_parameters(self):
        # Possible parameters to adjust go here

        self.aruco_detector.setDetectorParameters(self.parameters)

    # Generates Aruco Markers for the top_left and bottom_right corners and saves them to a specified directory
    def generate_aruco_markers(self, side_pixels=200, border_bits=1):
        output_dir = "aruco_markers"
        os.makedirs(output_dir, exist_ok=True)
        for corner, corner_id in self.corner_ids.items():
            marker_image = aruco.drawMarker(self.dic, corner_id, side_pixels)
            cv2.imwrite(f"{output_dir}/aruco_marker_{corner}.png", marker_image)
        print(f"Generated aruco markers for: {', '.join(self.corner_ids.keys())}")

    # Detects the Aruco Markers in the image and updates the corner history
    def detect_aruco_markers(self, image):
        greyscale_np_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, marker_ids, _ = self.aruco_detector.detectMarkers(greyscale_np_image)

        markers_detected = False

        # Goes through the detected markers and updates the corner history
        if marker_ids is not None:
            for i, marker_id in enumerate(marker_ids.flatten()):
                if marker_id in self.corner_ids.values():
                    corner_name = [key for key, value in self.corner_ids.items() if value == marker_id][0]
                    self.corner_history[corner_name].append(corners[i][0])

                    if len(self.corner_history[corner_name]) > self.smoothing_frames:
                        self.corner_history[corner_name].pop(0)

                    markers_detected = True

        if markers_detected:
            self.persistence_counter = self.persistence_frames
        else:
            if self.persistence_counter > 0:
                self.persistence_counter -= 1

        return corners, marker_ids, markers_detected

    # Visualizes the Aruco Markers by drawing a rectangle that encompasses the markers and circles at the corners
    def visualize_aruco_markers(self, image, corners, marker_ids, markers_detected, offset=(0, 0)):
        offset_x, offset_y = offset
        if markers_detected or self.persistence_counter > 0:
            cv2.aruco.drawDetectedMarkers(image, corners, marker_ids)
            for corner_name, history in self.corner_history.items():
                if history:
                    avg_corner = np.mean(np.array(history), axis=0).astype(int) - np.array([offset_x, offset_y])
                    if corner_name == "top_left":
                        self.top_left_avg = tuple(avg_corner[0])
                    elif corner_name == "bottom_right":
                        self.bottom_right_avg = tuple(avg_corner[2])

            # Adjusts the rectangle and circle positions based on the offset
            if self.top_left_avg and self.bottom_right_avg:
                cv2.rectangle(image, self.top_left_avg, self.bottom_right_avg, (0, 255, 0), 2)
                cv2.circle(image, self.top_left_avg, 10, (0, 0, 255), -1)
                cv2.circle(image, self.bottom_right_avg, 10, (0, 0, 255), -1)

        return image
