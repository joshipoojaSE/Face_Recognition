import os

import cv2
import numpy as np

# Create a Camera Object
cam = cv2.VideoCapture(0)

# Model
model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Settings
file_name = input("Enter the name of the person: ")
dataset_path = "./data/"
offset = 20  # Extra pixels around the face (hair, beard, etc.)
skip = 0  # Frame counter, used to save every 10th frame
face_data = []

# Read image from Camera Object
while True:
    success, img = cam.read()
    if not success:
        print("Reading Camera Failed!")
        break

    faces = model.detectMultiScale(img, 1.3, 5)

    # Pick the largest face (closest to the camera) by area w * h
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = faces[-1]

        # Crop the face: rows are y, columns are x. Clamp to the image boundaries.
        crop_face = img[
            max(0, y - offset):min(img.shape[0], y + h + offset),
            max(0, x - offset):min(img.shape[1], x + w + offset),
        ]
        crop_face = cv2.resize(crop_face, (100, 100))

        skip += 1
        if skip % 10 == 0:
            face_data.append(crop_face)
            print("Saved so far:", len(face_data))

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Cropped Face", crop_face)

    cv2.imshow("Image Window", img)
    key = cv2.waitKey(1)  # Pause here for 1 ms before you read the next image
    if key == ord('q'):
        break

# Release Camera, and Destroy Window
cam.release()
cv2.destroyAllWindows()

# Convert the face list into a data matrix: one row per image (m, 100*100*3)
if len(face_data) > 0:
    face_data = np.array(face_data)
    m = face_data.shape[0]
    face_data = face_data.reshape(m, -1)
    print("Data shape:", face_data.shape)

    os.makedirs(dataset_path, exist_ok=True)
    np.save(dataset_path + file_name + ".npy", face_data)
    print("Data saved at:", dataset_path + file_name + ".npy")
else:
    print("No faces collected, nothing saved.")
