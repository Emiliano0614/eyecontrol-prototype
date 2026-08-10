import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = '../02-face-mesh/face_landmarker.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# --- Blink detection tuning ---
# Both values were derived from real captured data (see 04-blink-detection
# session notes), not guessed:
#   - BLINK_THRESHOLD = 0.10: set just above the observed floor of a
#     relaxed, fully-closed eye (~0.08-0.10 in testing), while sitting
#     well below normal open-eye readings (0.4+) and below the ambiguous
#     "partially closed / squinting" middle band (~0.15-0.25) that showed
#     up during normal use.
#   - BLINK_FRAMES_REQUIRED = 7: originally set to 15 assuming a 30fps
#     camera (frame_timestamp_ms += 33 below reflects that original
#     assumption, used only as a fake increasing timestamp for MediaPipe
#     — it does NOT reflect real elapsed time). Real frame rate was
#     measured directly (counting loop iterations over a real 5-second
#     wall-clock window) and came out to ~13fps, not 30fps — most likely
#     because each loop iteration also runs a full face-landmark
#     inference call (detect_for_video), not just a camera read, so the
#     loop is slower than a bare camera capture would be. At ~13fps
#     (~77ms/frame), 15 frames was actually ~1.05 real seconds (measured
#     directly, holding a blink and timing it), far longer than the
#     intended ~0.5s hold. 7 frames at ~77ms/frame is ~540ms, matching
#     the ~0.5s target — confirmed live at 0.47s and 0.48s on repeat
#     tests. Still comfortably longer than a natural involuntary blink
#     (~100-150ms, or ~1-2 frames at this rate), so normal blinking
#     still won't false-trigger a selection.
BLINK_THRESHOLD = 0.10
BLINK_FRAMES_REQUIRED = 7


def run_camera(shared_data):
    """
    Same shared_data pattern as camera_thread.py from 03: expects
    shared_data["running"] to control the loop, and writes results back
    into shared_data for whatever UI/main thread is watching it.

    shared_data["eye_openness"]: live eye_openness value, updated every
        frame (useful for debugging/tuning, or a live UI readout).
    shared_data["blink_detected"]: flips to True for exactly one frame
        the moment a deliberate blink completes (BLINK_FRAMES_REQUIRED
        consecutive closed frames). The caller is responsible for
        reading it and then resetting it back to False after handling
        it, so the same blink doesn't get processed twice.
    """
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0
    closed_frame_count = 0

    while shared_data["running"] == True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # NOTE: +=33 assumes 30fps, which is NOT the real measured frame
        # rate (~13fps, see BLINK_FRAMES_REQUIRED comment above). This
        # only needs to be a strictly increasing number for MediaPipe's
        # VIDEO mode to accept it as "the next frame" — it does not need
        # to match real elapsed time for detection to work correctly.
        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks:
            left_eye_outer_corner = result.face_landmarks[0][33]
            left_eye_inner_corner = result.face_landmarks[0][133]
            left_upper_eyelid = result.face_landmarks[0][159]
            left_lower_eyelid = result.face_landmarks[0][145]

            eye_corner_distance = abs(left_eye_outer_corner.x - left_eye_inner_corner.x)
            eye_gap = abs(left_upper_eyelid.y - left_lower_eyelid.y)
            eye_openness = eye_gap / eye_corner_distance

            shared_data["eye_openness"] = eye_openness

            if eye_openness < BLINK_THRESHOLD:
                closed_frame_count += 1
            else:
                closed_frame_count = 0

            if closed_frame_count == BLINK_FRAMES_REQUIRED:
                shared_data["blink_detected"] = True

    cap.release()


if __name__ == "__main__":
    # Standalone test mode: run this file directly (not imported) to
    # watch blink detection live in the terminal without needing a
    # main.py or UI wired up yet.
    shared_data = {
        "running": True,
        "eye_openness": 0,
        "blink_detected": False,
    }

    import threading
    camera_thread = threading.Thread(target=run_camera, args=(shared_data,))
    camera_thread.start()

    try:
        while True:
            if shared_data["blink_detected"]:
                print("BLINK DETECTED")
                shared_data["blink_detected"] = False  # reset so it only fires once per blink
    except KeyboardInterrupt:
        shared_data["running"] = False
        camera_thread.join()