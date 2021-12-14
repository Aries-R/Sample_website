import cv2
import mediapipe as mp
import numpy as np
from math import sqrt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def time_diff(t1, t2):
    t1 = t1.split(":")
    t2 = t2.split(":")
    time1 = int(t1[0]) * 60 * 60 + int(t1[1]) * 60 + int(t1[2])
    time2 = int(t2[0]) * 60 * 60 + int(t2[1]) * 60 + int(t2[2])

    diff = time2 - time1
    h, m, s = 0, 0, 0
    if diff > 3600:
        h = diff // 3600
        diff %= 3600
    if diff > 60:
        m = diff // 60
        diff %= 60
    s = diff
    return ("{}:{}:{}".format(h, m, s))


import time
import os

cap = cv2.VideoCapture(0)

# Curl counter variables
n = 1
stage = "None"
# frames=0
hour = 0
minute = 0
second = 0

starting_time = "0:0:0"
current_time = "0:0:0"

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle_shoulder = calculate_angle(hip, shoulder, elbow)
            angle_hip = calculate_angle(shoulder, hip, knee)
            angle_knee = calculate_angle(hip, knee, ankle)
            angle_elbow = calculate_angle(shoulder, elbow, wrist)

            # Visualize angle
            cv2.putText(image, str(angle_hip),
                        tuple(np.multiply(hip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, str(angle_knee),
                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            coor_leftankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            coor_rightankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            coor_leftankle = tuple(np.multiply(coor_leftankle, [640, 480]).astype(int))
            coor_rightankle = tuple(np.multiply(coor_rightankle, [640, 480]).astype(int))

            x = coor_leftankle[0] - coor_rightankle[0]
            y = coor_leftankle[1] - coor_rightankle[1]

            # Curl counter logic
            if angle_shoulder < 30 and angle_hip < 125 and angle_hip > 60 and angle_knee < 30 and angle_elbow > 150 and x < 70 and x > -70 and y < 50 and y > -50:
                stage = "right"
                if starting_time == "0:0:0":
                    starting_time = time.strftime("%H:%M:%S", time.localtime())
                if starting_time != "0:0:0":
                    current_time = time.strftime("%H:%M:%S", time.localtime())
            else:
                stage = "wrong"
                if starting_time != "0:0:0":
                    difference= time_diff(starting_time, current_time)
                    if int(difference[0])!=0 or int(difference[2])!=0 or int(difference[4])!=0:
                        print("Trial ", n, "->", "Duration is", difference)
                        n += 1
                starting_time = "0:0:0"
                current_time = "0:0:0"
        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (250, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'Time', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, time_diff(starting_time, current_time),(10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        if stage == "wrong":
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(45, 70, 256), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(24, 24, 256), thickness=2, circle_radius=2)
                                      )
            # Render detections
        else:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 256, 0), thickness=2, circle_radius=2)
                                      )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()