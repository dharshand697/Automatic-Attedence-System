import cv2 
import pickle 
import os 
name = input("Enter your name: ") 
# Setup 
video = cv2.VideoCapture(0) 
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 
'haarcascade_frontalface_default.xml') 
faces_data = [] 
labels = []
print("[INFO] Collecting data. Press 'q' to quit...") 
while True: 
ret, frame = video.read() 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
faces = facedetect.detectMultiScale(gray, 1.3, 5) 
for (x, y, w, h) in faces: 
crop_img = frame[y:y+h, x:x+w] 
resized_img = cv2.resize(crop_img, (50, 50)).flatten() 
faces_data.append(resized_img) 
labels.append(name) 
cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
cv2.imshow("Collecting Face Data", frame) 
if cv2.waitKey(1) & 0xFF == ord('q'): 
break 
video.release() 
cv2.destroyAllWindows() 
# Ensure 'data' folder exists 
os.makedirs("data", exist_ok=True) 
# Save faces and labels 
with open("data/faces_data.pkl", "wb") as f: 
pickle.dump(faces_data, f) 
with open("data/names.pkl", "wb") as f: 
pickle.dump(labels, f) 
print("[INFO] Data collection complete. Files saved to 'data/'") 