# Import opencv
import cv2
# Import operating sys
import os
# Import matplotlib
from matplotlib import pyplot as plt

# Connect to capture device
cap = cv2.VideoCapture(2)

# Loop through every frame until we close our webcam
while cap.isOpened(): 
    ret, frame = cap.read()
    
    # Show image 
    cv2.imshow('Webcam', frame)
    
    # Checks whether q has been hit and stops the loop
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Releases the webcam
cap.release()
# Closes the frame
cv2.destroyAllWindows()