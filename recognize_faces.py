import cv2
import insightface
import numpy as np
import os
from scipy.spatial.distance import cosine

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

# Open camera
cap = cv2.VideoCapture(0)

print("Press 'Q' to quit")
print("Threshold: 0.6 (lower = stricter match)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = model.get(frame)
    
    for face in faces:
        bbox = face.bbox.astype(int)
        current_embedding = face.embedding
        
        # Compare with all stored embeddings
        best_match = None
        best_distance = float('inf')
        
        for name, embeddings in embeddings_db.items():
            for stored_embedding in embeddings:
                # Calculate similarity 
                distance = cosine(current_embedding, stored_embedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
        
        # Draw box and name
        if best_distance < 0.6:  
            color = (0, 255, 0)  # Green
            label = f"{best_match} ({best_distance:.2f})"
        else:  # No match
            color = (0, 0, 255)  # Red
            label = f"Unknown ({best_distance:.2f})"
        
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("Face Recognition - Press Q to quit", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
