import cv2
import face_recognition
from main import add

known_faces_encodings = []
known_faces_names = []

add("face_photo\chloe.jpg",'chloe')
add("face_photo\chen.jpg",'chen')
add("face_photo\jamie.JPG",'jamie')
add("face_photo\jenny.JPG",'jenny')
add("face_photo\epoch.JPG",'epoch')

print(known_faces_encodings)
print(known_faces_names)

known_person_image = face_recognition.load_image_file("face_photo\chloe.jpg")
known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
known_faces_encodings.append(known_person_encoding)
known_faces_names.append('chloe')

known_person_image = face_recognition.load_image_file("face_photo\chen.jpg")
known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
known_faces_encodings.append(known_person_encoding)
known_faces_names.append('david')

known_person_image = face_recognition.load_image_file("face_photo\jamie.JPG")
known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
known_faces_encodings.append(known_person_encoding)
known_faces_names.append('jamie')

known_person_image = face_recognition.load_image_file("face_photo\jenny.JPG")
known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
known_faces_encodings.append(known_person_encoding)
known_faces_names.append('jenny')

known_person_image = face_recognition.load_image_file("face_photo\epoch.JPG")
known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
known_faces_encodings.append(known_person_encoding)
known_faces_names.append('epoch')


video_capture = cv2.VideoCapture(1)

while True:
    ret, frame = video_capture.read()

    face_locations = face_recognition.face_locations(frame, model='cnn')
    face_encodings = face_recognition.face_encodings(frame, face_locations, model='large')

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
