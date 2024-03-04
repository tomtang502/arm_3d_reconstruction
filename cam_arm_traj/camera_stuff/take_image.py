import cv2

# Open handles to the webcams
cap1 = cv2.VideoCapture(2)
cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, -4.25)  # Example value

idx = 0
while True:
    # Capture each frame from both cameras
    ret1, frame1 = cap1.read()

    # If the frames were successfully captured
    if ret1 :
        #print(frame1.shape)
        cv2.imshow('Camera 1', frame1)

        # Break the loop on pressing 'q'
        keyval = cv2.waitKey(1) & 0xFF
        if keyval == ord('q'):
            print("quit!")
            break
        elif keyval == ord(' '):  # 32 is the ASCII code for the space bar
        # Save the frame as an image file
            print("start image save")
            cv2.imwrite('../arm_images/init_arm_t'+str(idx)+'.jpg', frame1)
            print("image saved!")
            idx = idx+1

    else:
        print("Unable to capture frame")
        break

# After the loop release the cap objects
cap1.release()

# Destroy all the windows
cv2.destroyAllWindows()