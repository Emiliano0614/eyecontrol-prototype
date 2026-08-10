import cv2

# Connect to the webcam (0 = default camera)
cap = cv2.VideoCapture(0)
# cap is now an object representing that open connection.
while True:
    # Ask the webcam for one frame
    #it returns 2 things 
    # frame(the actual NumPy array of pixels)
    #ret a true / false value whether the read actually succeeded.
    ret, frame = cap.read()

    # Show that frame in a window
    cv2.imshow("Webcam Feed",frame)
    #shows the actual (height, width, 3) numbers of the frame
    #print(frame.shape)
    #gets the exact pixel from that frame and 
    #returns the  value for each  bgr
    print(frame[540,990])
    #check if the user pressed q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()