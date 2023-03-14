from deepface import DeepFace
face_verified = DeepFace.verify(img1_path="dataset/testing/modi1.jpg",
                                    img2_path="dataset/testing/joe1.jpg",
                                    detector_backend="opencv",
                                    model_name="VGG-Face",
                                    distance_metric="cosine")

print(face_verified)

