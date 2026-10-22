import os

import cv2
import numpy as np

# Settings
dataset_path = "./data/"
offset = 20  # Extra pixels around the face, same as when the data was collected
face_data = []  # One (m, 30000) array per person
labels = []  # One array of class ids per person
name_map = {}  # class_id -> person name
class_id = 0

# Load every saved .npy file. Sort the filenames so each person always gets the same class_id.
for file in sorted(os.listdir(dataset_path)):
    if not file.endswith(".npy"):
        continue

    data_item = np.load(os.path.join(dataset_path, file))
    number_of_images = data_item.shape[0]
    print(file, "->", data_item.shape)

    face_data.append(data_item)
    labels.append(np.full(number_of_images, class_id))  # Same label for all images of this person
    name_map[class_id] = file[:-4]  # Remove ".npy" to get the name
    class_id += 1

# Join all people along the rows: X_train is (total_images, 30000), y_train is (total_images, 1)
# Named X_train / y_train (not X / y) so the face's y coordinate below can't overwrite the labels.
X_train = np.concatenate(face_data, axis=0)
y_train = np.concatenate(labels, axis=0).reshape(-1, 1)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Name map:", name_map)


# KNN classifier
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        # Use float so subtracting uint8 pixels doesn't wrap around
        self.X = X.astype(np.float32)
        self.y = y.ravel()

    def predict(self, X_test):
        predictions = []
        for x in X_test.astype(np.float32):
            distances = np.sqrt(((self.X - x) ** 2).sum(axis=1))  # Distance to every training face
            nearest = self.y[np.argsort(distances)[:self.k]]  # Labels of the k closest faces
            values, counts = np.unique(nearest, return_counts=True)
            predictions.append(values[np.argmax(counts)])  # Majority vote
        return np.array(predictions)


knn = KNN(k=5)
knn.fit(X_train, y_train)

# Prediction: detect faces in the webcam and predict who they are
cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    success, img = cam.read()
    if not success:
        print("Reading Camera Failed!")
        break

    faces = model.detectMultiScale(img, 1.3, 5)

    for x, y, w, h in faces:
        # Crop the face the same way as during data collection
        crop_face = img[
            max(0, y - offset):min(img.shape[0], y + h + offset),
            max(0, x - offset):min(img.shape[1], x + w + offset),
        ]
        crop_face = cv2.resize(crop_face, (100, 100))

        # Flatten the face into 30,000 features
        X_test = crop_face.reshape(1, -1)

        # Predict the person's class and convert the class ID to a name
        class_predicted = knn.predict(X_test)
        name_predicted = name_map[class_predicted[0]]

        # Display the name and a box around the face
        cv2.putText(img, name_predicted, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Prediction Window", img)
    key = cv2.waitKey(1)  # Pause here for 1 ms before you read the next image
    if key == ord('q'):
        break

# Release Camera, and Destroy Window
cam.release()
cv2.destroyAllWindows()
