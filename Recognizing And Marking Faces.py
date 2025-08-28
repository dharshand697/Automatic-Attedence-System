import cv2 
import pickle 
import numpy as np 
import os 
import csv 
from datetime import datetime 
from sklearn.neighbors import KNeighborsClassifier 
from utils import speak # Ensure utils.py is present and has a speak function 
# Setup 
video = cv2.VideoCapture(0) 
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 
'haarcascade_frontalface_default.xml') 
# Load face data 
with open ('data/names.pkl', 'rb') as f: 
labels = pickle.load(f) 
with open ('data/faces_data.pkl', 'rb') as f: 
faces = pickle.load(f) 

# Train classifier 
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(faces, labels) 
 
# Attendance folder 
os.makedirs("Attendance", exist_ok=True) 
COL_NAMES = ['NAME', 'TIME'] 
 
THRESHOLD = 10000 # Increased threshold to account for raw pixel distances 
 
while True: 
    ret, frame = video.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces_coord = facedetect.detectMultiScale(gray, 1.3, 5) 
 
    for (x, y, w, h) in faces_coord: 
        crop_img = frame [y:y+h, x:x+w] 
        resized_img = cv2.resize(crop_img, (50, 50)). flatten (). reshape (1, -1) 
 
        # Get nearest neighbor distance 
        distances, indices = knn.kneighbors(resized_img, n_neighbors=1) 
        min_distance = distances [0][0] 
        predicted_label = knn.predict(resized_img)[0] 
 
        print(f"[DEBUG] Distance: {min_distance}, Prediction: {predicted_label}") 
 
        if min_distance <= THRESHOLD: 
            output = predicted_label 
        else: 
            output = "Unknown" 
 
        # Draw bounding box 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2) 
 
        # Show identity 
        cv2.putText(frame, output, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, 
                    (0, 255, 0) if output! = "Unknown" else (0, 0, 255), 2) 
 
        # Only mark attendance if known 
        if output! = "Unknown": 
            timestamp = datetime.now (). strftime("%H: %M: %S") 
            date = datetime.now (). strftime("%d-%m-%Y") 
            attendance_file = f"Attendance/Attendance_{date}.csv" 
 
            # Check if already marked 
            eligible_tag = "NOT ELIGIBLE" 
            already_present = False 
 
            if os.path.isfile(attendance_file): 
                with open (attendance_file, 'r') as f: 
                    reader = csv.reader(f) 
                    next (reader, None) 
                    for row in reader: 

 
 
 
                        if row and row [0] == output: 
                            eligible_tag = "ELIGIBLE" 
                            already_present = True 
                            break 
            # Display eligibility 
            cv2.putText(frame, eligible_tag, (x, y + h + 30), cv2.FONT_HERSHEY_COMPLEX, 
0.8, (0, 255, 0) if eligible_tag == "ELIGIBLE" else (0, 0, 255), 2) 
 
            # Mark attendance if not already marked 
            if not already_present: 
                speak (f"Welcome {output}") 
                if not os.path.isfile(attendance_file): 
                    with open (attendance_file, 'w', newline='') as f: 
                        writer = csv.writer(f) 
                        writer.writerow(COL_NAMES) 
                with open(attendance_file, 'a', newline='') as f: 
                    writer = csv.writer(f) 
                    writer.writerow([output, timestamp]) 
 
    cv2.imshow("Face Recognition Attendance", frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
 
video.release() 
cv2.destroyAllWindows()