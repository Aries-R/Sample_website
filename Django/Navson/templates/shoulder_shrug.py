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


def straight():
    coor_leftshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    coor_leftshoulder = tuple(np.multiply(coor_leftshoulder, [640, 480]).astype(int))
    coor_leftelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    coor_leftindex = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
    coor_leftelbow = tuple(np.multiply(coor_leftelbow, [640, 480]).astype(int))
    coor_leftindex = tuple(np.multiply(coor_leftindex, [640, 480]).astype(int))

    return coor_leftshoulder, coor_leftelbow, coor_leftindex


# straight()
c = 0

cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0
stage = None
count = 0
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
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # Calculate angle
            angle_elbow = calculate_angle(shoulder, elbow, wrist)
            angle_shoulder = calculate_angle(hip, shoulder, elbow)

            # Visualize angle
            cv2.putText(image, str(angle_elbow),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            cv2.putText(image, str(angle_shoulder),
                        tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if angle_elbow > 160 and angle_shoulder < 30:
                count += 1
                if count > 5:
                    stage = "down"
                    count = 0

            if angle_elbow > 160 and angle_shoulder > 80 and angle_shoulder < 110 and stage == 'down':
                count += 1
                if count > 5:
                    stage = "up"
                    count = 0
                    counter += 1
                    print(counter)

            elif angle_shoulder > 110:
                error = "Bring down your hands down"
                # cv2.line(image,(left_cord,right_cord),(left_cord+20,right_cord),(0,255,0),3)

            elif angle_shoulder < 110:
                error = " "
        except:
            pass

            # Render curl counter
            # Setup status box
        cv2.rectangle(image, (0, 0), (275, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (80, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (80, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        if error != " ":
            cv2.rectangle(image, (400, 450), (640, 480), (245, 117, 16), -1)
            cv2.putText(image, error, (405, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            shoulder1, elbow1, index1 = straight()
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                      )
        else:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                      )
        if error == "Bring down your hands":
            d1 = sqrt(((elbow1[0] - shoulder1[0]) ** 2) + ((elbow1[1] - shoulder1[1]) ** 2))
            d2 = sqrt(((elbow1[0] - index1[0]) ** 2) + ((elbow1[1] - index1[1]) ** 2))
            d1 = int(d1)
            d2 = int(d2)
            cv2.line(image, (shoulder1[0], shoulder1[1]), (shoulder1[0] + d1, shoulder1[1]), (66, 245, 230), 3)
            cv2.circle(image, (shoulder1[0] + d1, shoulder1[1]), 2, (255, 0, 0), 2)
            cv2.line(image, (shoulder1[0] + d1, shoulder1[1]), (shoulder1[0] + d1 + d2, shoulder1[1]), (66, 245, 230),
                     3)

            # Render detections

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()