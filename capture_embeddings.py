import cv2
import insightface
import numpy as np
import os

# Load model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)

# Create folder to store embeddings
os.makedirs('embeddings', exist_ok=True)

# student name
name = input("Enter student name: ")
student_folder = f'embeddings/{name}'
os.makedirs(student_folder, exist_ok=True)

# Open camera
cap = cv2.VideoCapture(0)
count = 0

print(f"Capturing embeddings for {name}")
print("Press 'SPACE' to capture, 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = model.get(frame)
    
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Save embedding 
        if count > 0:
            embedding = face.embedding
            np.save(f'{student_folder}/embedding_{count}.npy', embedding)
            print(f"✓ Embedding {count} saved!")
    
    cv2.imshow(f"Capturing for {name} - Press SPACE to capture, Q to quit", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # SPACE key
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✓ Captured {count} embeddings for {name}")