from deepface import DeepFace
face_recognition = DeepFace.find(img_path="dataset/testing/modi1.jpg",
                                    db_path="dataset/training",
                                    detector_backend="opencv",
                                    model_name="VGG-Face",
                                    distance_metric="cosine")

print(face_recognition)

