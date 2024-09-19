from flask import Flask, request, jsonify
import cv2
import face_recognition

app = Flask(__name__)

# 假設有一個保存人臉特徵的資料庫
database = {
    "chloe": "path_to_user1_face_encoding.npy",
    "chen": "path_to_user2_face_encoding.npy",
    "jamie": "path_to_user3_face_encoding.npy",
    "jenny": "path_to_user4_face_encoding.npy",
    "epoch": "path_to_user5_face_encoding.npy"
}


@app.route('/login', methods=['POST'])
def login():
    img = request.files['image'].read()
    np_img = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # 獲取人臉特徵
    face_encodings = face_recognition.face_encodings(img)

    if not face_encodings:
        return jsonify({"message": "No face found"}), 400

    # 比對資料庫中的人臉特徵
    for user, encoding_path in database.items():
        known_encoding = np.load(encoding_path)
        matches = face_recognition.compare_faces([known_encoding], face_encodings[0])
        if matches[0]:
            return jsonify({"message": "Login successful", "user": user}), 200

    return jsonify({"message": "Login failed"}), 401


if __name__ == '__main__':
    app.run(debug=True)
