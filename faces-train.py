import numpy as np
import os
import cv2
from PIL import Image #python pillow library, PIL = Python Image Library
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
recognizer = cv2.face.LBPHFaceRecognizer_create()

y_labels = []
x_train = []
label_ids = {}
current_id = 0
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() # or label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)
            # y_labels.append(label)
            # x_train.append(path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id = current_id + 1
            id = label_ids[label]
            # print(label_ids)
            pil_image = Image.open(path).convert("L") #.convert("L") converts to grayscale
            image_array = np.array(pil_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id)
# print(y_labels)
# print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")