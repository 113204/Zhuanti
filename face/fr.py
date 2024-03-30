import cv2
import face_recognition
from main import add

known_faces_encodings = []
known_faces_names = []
# add("face_0316\chloe.jpg",'chloe')
#
# print(known_faces_encodings)
# print(known_faces_names)
known_person_image = face_recognition.load_image_file("face_0316\chloe.jpg")
known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
known_faces_encodings.append(known_person_encoding)
known_faces_names.append('chloe')

# known_person_image = face_recognition.load_image_file("face_0316\chen.jpg")
# known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
# known_faces_encodings.append(known_person_encoding)
# known_faces_names.append('chen')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for(top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces_encodings, face_encodings)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_faces_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("abc", frame)

    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()