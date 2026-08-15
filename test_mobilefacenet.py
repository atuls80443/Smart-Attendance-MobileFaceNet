import cv2
import insightface
import numpy as np

# Load MobileFaceNet model (downloads automatically)
print("Loading MobileFaceNet model...")
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)  # ctx_id=0 uses CPU

print("✓ MobileFaceNet loaded successfully!")
print("Model ready for face embedding!")