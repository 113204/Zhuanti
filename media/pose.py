
import datetime
import time
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


cap = cv2.VideoCapture(0)

counter = 0
stage = None
begin = []
start = []
danger = []
exercise_time = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, img = cap.read()

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        try:

            landmarks = results.pose_landmarks.landmark

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            if right_elbow_angle > 150:
                if len(start) == 0:
                    start_time = time.time()
                    start.append(start_time)
                started_time = time.time() - start[0]
                # print(started_time)
            elif right_elbow_angle < 150 and started_time < 3:
                start.clear()

            if len(begin) == 0:
                begin_time = datetime.datetime.now().replace(microsecond=0)
                begin.append(begin_time)
            exercise_time = datetime.datetime.now().replace(microsecond=0) - begin[0]

            if int(started_time) == 1:
                show_time = 3
            elif int(started_time) == 2:
                show_time = 2
            elif int(started_time) == 3:
                show_time = 1
            else:
                show_time = ''
            cv2.putText(img, str(show_time), (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 7,
                        cv2.LINE_AA)

            if started_time > 3:
                if right_elbow_angle < 20:
                    warning_message = 'elbow wrong'
                    joint_color = (255, 255, 0)
                    color = (255, 255, 0)

                    if len(danger) == 0:
                        danger_time = time.time()
                        danger.append(danger_time)
                    die_time = time.time() - danger[0]

                    if die_time > 10:
                        die_message = 'die'
                        joint_color = (0, 0, 255)
                        color = (0, 0, 255)
                    elif die_time > 7:
                        die_message = 'maybe die'
                        joint_color = (0, 255, 255)
                        color = (0, 255, 255)

                elif 20 < right_elbow_angle < 50 and right_shoulder_angle < 60:
                    danger.clear()
                    stage = 'down'
                    warning_message = ''
                    joint_color = (37, 216, 195)
                    color = (141, 190, 104)
                elif 50 < right_elbow_angle < 150 and right_shoulder_angle > 60:
                    stage = 'null'
                    warning_message = ''
                    joint_color = (37, 216, 195)
                    color = (141, 190, 104)

                if stage == 'down' and (right_shoulder_angle < 30 or right_shoulder_angle > 60):
                    warning_message = 'shoulder wrong'
                    joint_color = (255, 255, 0)
                    color = (255, 255, 0)
                elif stage == 'down' and right_elbow_angle > 50:
                    warning_message = 'elbow wrong'
                    joint_color = (255, 255, 0)
                    color = (255, 255, 0)

                if right_elbow_angle > 150 and stage == 'null':
                    stage = 'up'
                    warning_message = ''
                    counter += 1
                    joint_color = (37, 216, 195)
                    color = (141, 190, 104)
                    print(counter)
                print(stage)
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=joint_color, thickness=5, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                cv2.rectangle(img, (0, 0), (140, 40), (255, 0, 0), -1)
                cv2.rectangle(img, (200, 0), (450, 40), (0, 255, 255), -1)
                cv2.rectangle(img, (550, 0), (650, 40), (0, 0, 255), -1)
                cv2.putText(img, str(exercise_time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(img, str(warning_message), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(img, str(counter), (580, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            else:
                warning_message = ''
                begin.clear()
                empty = datetime.datetime.now().replace(microsecond=0) - datetime.datetime.now().replace(microsecond=0)
                exercise_time = empty

                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(37, 216, 195), thickness=5, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(141, 190, 104), thickness=2, circle_radius=2),
                    # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                cv2.rectangle(img, (0, 0), (140, 40), (255, 0, 0), -1)
                cv2.rectangle(img, (200, 0), (450, 40), (0, 255, 255), -1)
                cv2.rectangle(img, (550, 0), (650, 40), (0, 0, 255), -1)
                cv2.putText(img, str(exercise_time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, str(warning_message), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(img, str(counter), (580, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.putText(img, str(int(left_elbow_angle)), tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.putText(img, str(int(left_shoulder_angle)),
                        tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.putText(img, str(int(left_knee_angle)), tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.putText(img, str(int(right_elbow_angle)), tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.putText(img, str(int(right_shoulder_angle)),
                        tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.putText(img, str(int(right_knee_angle)), tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
        except:
            pass

        img = cv2.resize(img, (960, 720))
        cv2.imshow('usb_camera', img)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
