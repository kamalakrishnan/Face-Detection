from deepface import DeepFace
import cv2
#face detection and face alignment
face_detected = DeepFace.detectFace(img_path="dataset/testing/modi1.jpg",
                                    detector_backend="opencv")
face_detected = cv2.cvtColor(face_detected,cv2.COLOR_BGR2RGB)
cv2.imshow("face_detected",face_detected)
