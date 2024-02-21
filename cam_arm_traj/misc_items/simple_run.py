import cv2

# Open handles to the webcams
cap1 = cv2.VideoCapture(2)

while True:
    # Capture each frame from both cameras
    ret1, frame1 = cap1.read()

    # If the frames were successfully captured
    if ret1 :
    # if True:
        # Display the resulting frames
        cv2.imshow('Camera 1', frame1)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Unable to capture frame")
        break

# After the loop release the cap objects
cap1.release()

# Destroy all the windows
cv2.destroyAllWindows()