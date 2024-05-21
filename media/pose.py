import datetime
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
start = []
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

            cv2.putText(img, str(int(left_elbow_angle)), tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.putText(img, str(int(left_shoulder_angle)), tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.putText(img, str(int(left_knee_angle)), tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.putText(img, str(int(right_elbow_angle)), tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.putText(img, str(int(right_shoulder_angle)), tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            cv2.putText(img, str(int(right_knee_angle)), tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            # print(tuple(np.multiply(elbow, [640, 480]).astype(int)))

        # if right_knee_angle < 130 or left_knee_angle < 130:

            if right_elbow_angle < 60 and left_elbow_angle < 60:
                stage = 'down'
            if right_elbow_angle > 150 and left_elbow_angle > 150 and stage == 'down':
                stage = 'up'
                counter += 1
                print(counter)

            if len(start) == 0:
                start_time = datetime.datetime.now().replace(microsecond=0)
                start.append(start_time)

            warning_message = ''
            exercise_time = datetime.datetime.now().replace(microsecond=0) - start[0]

            if right_shoulder_angle < 45 or right_shoulder_angle > 60 or left_shoulder_angle < 45 or left_shoulder_angle > 60:
                warning_message = 'shoulder wrong'
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=5, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            if right_elbow_angle < 60 or left_elbow_angle < 60:
                warning_message = 'elbow wrong'
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=5, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
        # else:
            warning_message = ''
            # start.clear()
            # empty = datetime.datetime.now().replace(microsecond=0) - datetime.datetime.now().replace(microsecond=0)
            # exercise_time = empty

            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=5, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            cv2.rectangle(img, (0, 0), (140, 40), (255, 0, 0), -1)
            cv2.rectangle(img, (200, 0), (450, 40), (0, 255, 255), -1)
            cv2.putText(img, str(exercise_time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, str(warning_message), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        except:
            pass

        img = cv2.resize(img, (960, 720))
        cv2.imshow('pose', img)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
