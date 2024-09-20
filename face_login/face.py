import face_recognition
import numpy as np
import cv2


def extract_face_encoding(image_path):
    # 讀取圖像
    image = cv2.imread(image_path)

    # 提取臉部特徵
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) > 0:
        return face_encodings[0]
    else:
        raise ValueError("No face found in the image")


# 範例：提取並保存某位用戶的人臉特徵
face_encoding = extract_face_encoding("face_photo\chloe.jpg")
np.save("user1_face_encoding.npy", face_encoding)
face_encoding = extract_face_encoding("face_photo\chen.jpg")
np.save("user2_face_encoding.npy", face_encoding)
face_encoding = extract_face_encoding("face_photo\jamie.JPG")
np.save("user3_face_encoding.npy", face_encoding)
face_encoding = extract_face_encoding("face_photo\jenny.JPG")
np.save("user4_face_encoding.npy", face_encoding)
face_encoding = extract_face_encoding("face_photo\epoch.JPG")
np.save("user5_face_encoding.npy", face_encoding)

saved_encoding = np.load("user1_face_encoding.npy")
saved_encoding = np.load("user2_face_encoding.npy")
saved_encoding = np.load("user3_face_encoding.npy")
saved_encoding = np.load("user4_face_encoding.npy")
saved_encoding = np.load("user5_face_encoding.npy")


def verify_user_login(input_image, user_id):
    input_encoding = extract_face_encoding(input_image)

    saved_encoding = load_face_encoding_from_db(user_id)

    matches = face_recognition.compare_faces([saved_encoding], input_encoding)

    return matches[0]


