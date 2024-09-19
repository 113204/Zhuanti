import cv2
import face_recognition

known_faces_encodings = []
known_faces_names = []

def add(image,name):
    known_person_image = face_recognition.load_image_file(image)
    known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
    known_faces_encodings.append(known_person_encoding)
    known_faces_names.append(name)
    return known_faces_encodings,known_faces_names