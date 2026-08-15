import cv2

for i in range(3):
    print(f"Testing camera {i}")
    cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"SUCCESS: Camera {i} - Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"Opened but cannot read frame - Camera {i}")
    else:
        print(f"Not opened - Camera {i}")
    
    cap.release()

print("Scan complete!")