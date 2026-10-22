import os

import numpy as np

# Settings
dataset_path = "./data/"
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

# Join all people along the rows: X is (total_images, 30000), y is (total_images, 1)
X = np.concatenate(face_data, axis=0)
y = np.concatenate(labels, axis=0).reshape(-1, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Name map:", name_map)
