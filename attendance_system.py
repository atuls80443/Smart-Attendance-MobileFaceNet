import cv2
import insightface
import numpy as np
import os
from scipy.spatial.distance import cosine
from datetime import datetime
import csv
import time 

# Load model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)

# Load all stored embeddings
embeddings_db = {}

for name in os.listdir('embeddings'):
    student_folder = f'embeddings/{name}'
    embeddings_db[name] = []
    
    for emb_file in os.listdir(student_folder):
        embedding = np.load(f'{student_folder}/{emb_file}')
        embeddings_db[name].append(embedding)

print(f"Loaded embeddings for: {list(embeddings_db.keys())}")

# Create attendance CSV file 
csv_file = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
marked_today = {}  # Track who's already marked
HOLD_SECONDS = 3

if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Time', 'Confidence'])

# Open camera
cap = cv2.VideoCapture(0)
print("Attendance System Running - Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = model.get(frame)
    
    for face in faces:
        bbox = face.bbox.astype(int)
        current_embedding = face.embedding
        
        # Compare with stored embeddings
        best_match = None
        best_distance = float('inf')
        
        for name, embeddings in embeddings_db.items():
            for stored_embedding in embeddings:
                distance = cosine(current_embedding, stored_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
        
        # Check if match and not already marked today
        if best_distance < 0.6 and best_match not in marked_today:
            color = (0, 255, 0)  # Green - new attendance
            label = f"{best_match} marked"
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([best_match, timestamp, f"{best_distance:.2f}"])
            
            marked_today[best_match] = time.time()  # save mark time
            print(f"Attendance marked: {best_match} at {timestamp}")
        
        elif best_distance < 0.6 and (time.time() - marked_today[best_match]) < HOLD_SECONDS:
            color = (0, 255, 0)  # Still green - hold period active
            label = f"{best_match} marked"
        
        elif best_distance < 0.6:
            color = (255, 255, 0)  # Yellow - already marked (after hold)
            label = f"{best_match} (already marked)"
        
        else:
            color = (0, 0, 255)  # Red - unknown face
            label = "Unknown"
        
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow("Attendance System - Press Q to quit", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Attendance saved to {csv_file}")