import cv2
import os
import numpy as np
import sys

# Path to dataset folder
dataset_path = "dataset"

if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset folder '{dataset_path}' not found. Please create it with subfolders per person.")
    sys.exit(1)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("ERROR: Could not load Haar Cascade xml file.")
    sys.exit(1)

# Prepare training data
faces = []
labels = []
label_names = []
label_id = 0

print("Preparing dataset and training...")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"Processing {person_name}...")

    label_names.append(person_name)
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces_rects) == 0:
            print(f"Warning: No face found in image {img_path}, skipping.")
            continue

        # Take the first face found
        (x, y, w, h) = faces_rects[0]
        face_roi = gray[y:y+h, x:x+w]

        faces.append(face_roi)
        labels.append(label_id)

    label_id += 1

if len(faces) == 0:
    print("ERROR: No faces found in dataset. Please add face images.")
    sys.exit(1)

# Create recognizer - requires opencv-contrib-python
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("ERROR: LBPHFaceRecognizer not found. Make sure opencv-contrib-python is installed.")
    sys.exit(1)

recognizer.train(faces, np.array(labels))

print("Training completed. Starting webcam for recognition...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_rects:
        face_roi = gray_frame[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face_roi)

        if confidence < 70:
            name = label_names[label]
            label_text = f"{name} ({int(confidence)})"
        else:
            label_text = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
