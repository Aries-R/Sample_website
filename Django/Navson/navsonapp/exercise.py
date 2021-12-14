import cv2
import mediapipe as mp
import numpy as np
from math import sqrt
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
class Bicep(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        #ret,frame=self.video.read()
        def calculate_angle(a, b, c):
            a = np.array(a)  # First
            b = np.array(b)  # Mid
            c = np.array(c)  # End

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle

            return angle
        global counter
        counter = 0
        stage = None
        error = "None"
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            ret,frame=self.video.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                angle_elbow = calculate_angle(shoulder, elbow, wrist)
                angle_shoulder = calculate_angle(hip, shoulder, elbow)

                cv2.putText(image, str(angle_elbow),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                if angle_elbow > 160 and angle_shoulder < 30:
                    stage = "down"
                if angle_elbow < 30 and stage == 'down' and angle_shoulder < 30:
                    stage = "up"
                    counter += 1
                    print(counter)
                if angle_shoulder > 25:
                    error = "Keep your elbow close to the body"
                else:
                    error = " "
            except:
                pass

            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'STAGE', (80, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            if error == "Keep your elbow close to the body":

                cv2.rectangle(image, (315, 450), (640, 480), (245, 117, 16), -1)
                cv2.putText(image, error, (330, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 70, 255), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(45, 70, 255), thickness=2, circle_radius=2))
            else:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


        ret,jpg=cv2.imencode(".jpg",image)
        return jpg.tobytes()
class Shoulder_Shrug(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
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
            ret,frame=self.video.read()

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
        ret,jpg=cv2.imencode(".jpg",image)
        return jpg.tobytes()
class Squat(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        def calculate_angle(a, b, c):
            a = np.array(a)  # First
            b = np.array(b)  # Mid
            c = np.array(c)  # End

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle

            return angle
        cap = cv2.VideoCapture(0)

        # Curl counter variables
        counter = 0
        stage = "up"
        from PIL import Image
        img = Image.open(r"squat_crt_pos.jpeg")
        #img = cv2.resize(img, (640, 480))
        window_name = "correct_position"
        error_1, error_2 = None, None

        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            error = " "
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle_hip = calculate_angle(shoulder, hip, knee)
                angle_shoulder = calculate_angle(hip, shoulder, elbow)
                angle_knee = calculate_angle(hip, knee, ankle)

                # Visualize angle
                cv2.putText(image, str(angle_hip),
                            tuple(np.multiply(hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(angle_shoulder),
                            tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(angle_knee),
                            tuple(np.multiply(knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic
                error_1, error_2 = "", ""
                if angle_hip > 40 and angle_hip < 90 and stage == "up":
                    if angle_shoulder < 150 and angle_shoulder > 70:
                        if angle_knee < 110:
                            stage = "down"
                            counter += 1
                            cv2.destroyWindow(window_name)
                            print(counter)

                    elif angle_shoulder < 70 or angle_shoulder > 150:
                        error_1 = "hands straight in front"

                elif angle_hip < 40 or angle_hip > 90:
                    error_2 = "Keep your thighs parallel to the floor"

                if angle_hip > 150 and stage == 'down':
                    stage = "up"
                    error_1, error_2 = "", ""
                    global errors
                    errors=True
                    cv2.imshow(window_name, img)
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

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

            if error_1 != "":
                cv2.rectangle(image, (150, 420), (640, 450), (245, 117, 16), -1)
                cv2.putText(image, error_1, (155, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                            )
            if error_2 != "":
                cv2.rectangle(image, (150, 450), (640, 480), (245, 117, 16), -1)
                cv2.putText(image, error_2, (155, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                            )
            # Render detections
            if error_1 == "" and error_2 == "":
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                            )
        ret,jpg=cv2.imencode(".jpg",image)
        return jpg.tobytes()
class Yoga(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
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
        ret,jpg=cv2.imencode(".jpg",image)
        return jpg.tobytes()
def err():
    if errors==True:
        answer=True
    else:
        answer=False
    return answer