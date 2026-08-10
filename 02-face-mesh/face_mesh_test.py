import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# Point to the model file
model_path = 'face_landmarker.task'

# Configure and create the landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    #tells it "expect a continuous stream of frames,"
    running_mode=vision.RunningMode.VIDEO,
    #means only track one face at a time
    num_faces=1
)
#this actually builds the model object, loading
# those weights from the file into memory, once, 
# before the loop starts.
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

while True:
    ret,frame = cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap the frame in MediaPipe's own Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


    # Video mode requires an increasing timestamp per frame
    #the model expects you to tell it when each 
    # frame occurred, in increasing milliseconds, 
    # so it can reason about motion between frames
    frame_timestamp_ms += 33
    # Run the face mesh model on this frame
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    #ckecks to see if there is a face or not
        #if there is a face then print out the landmarks
    if result.face_landmarks:
        #its a list of faces and each face is itself of that faces landmark points
        #a list containing one (or more) lists inside it
        #print(result.face_landmarks)
        #since the options i just made it for 1 face only just grab the first one
        #since indices are fixed the tip of the nose is always 1
        #nose_tip = result.face_landmarks[0][1]
        #it only prits the coords
        #print(nose_tip.x, nose_tip.y, nose_tip.z)
        left_iris_center = result.face_landmarks[0][468]
        right_iris_center = result.face_landmarks[0][473]
        left_eye_outer_corner = result.face_landmarks[0][33]
        left_eye_inner_corner = result.face_landmarks[0][133]
        right_eye_outer_corner = result.face_landmarks[0][362]
        right_eye_inner_corner = result.face_landmarks[0][263]
        #print(left_iris_center.x, left_iris_center.y, right_iris_center.x, right_iris_center.y, left_eye_outer_corner.x, left_eye_outer_corner.y, left_eye_inner_corner.x,left_eye_inner_corner.y ,right_eye_outer_corner.x, right_eye_outer_corner.y, right_eye_inner_corner.x, right_eye_inner_corner.y)
        # --- Calculate horizontal gaze ratio (t) for the left eye ---
        # t tells us where the iris sits between the two eye corners:
        #   t = 0   -> iris is at the outer corner (looking outward)
        #   t = 1   -> iris is at the inner corner (looking toward nose)
        #   t = 0.5 -> iris is roughly centered
        #the formula is t = (P(left_iris_center.x) - A(left_eye_outer_corner.x)) / (B(left_eye_inner_corner.x) - A(left_eye_outer_corner.x ))
        # Step 1: compute how far the iris has moved from the outer corner (numerator)
        numerator = left_iris_center.x - left_eye_outer_corner.x
        # Step 2: compute the total distance between outer and inner corner (denominator)
        denominator = left_eye_inner_corner.x -  left_eye_outer_corner.x 
        # Step 3: divide numerator by denominator to get t
        t = numerator/denominator
        # Step 4: print t rounded to 2 decimal places, so we can watch it change liv
        # since ur iris dont go all the way to the outer corner or inner corner
        #it only goes around 0.43 and 0.56 so this
        print(round(t, 2))
    cv2.imshow("Face Mesh", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()