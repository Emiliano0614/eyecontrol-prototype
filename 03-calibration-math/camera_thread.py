import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import statistics

model_path = '../02-face-mesh/face_landmarker.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

# created once at module load — reused every call, not recreated per-frame
landmarker = vision.FaceLandmarker.create_from_options(options)

# scratch buffers used only during an active capture — hold up to 15
# raw t_x/t_y samples, get cleared once a median is computed. These
# don't need to live in shared_data since thread B never needs to see
# them mid-collection, only the final median.
capture_buffer_x = []
capture_buffer_y = []
# NEW — same idea as capture_buffer_x/y above, but for pitch_signal.
# Collects 15 raw pitch_signal readings during a single calibration-point
# capture burst, gets median'd and cleared the same way as x/y.
capture_buffer_pitch_signal = []

# NEW — separate from capture_buffer_x/y above. These are ALWAYS-ON
# sliding windows that smooth the live t_x/t_y signal shown at every
# moment, not just during a 9-point calibration burst. Declared at
# module level (not inside run_camera) so they persist across every
# frame of the whole program's lifetime, the same reason
# capture_buffer_x/y live up here instead of inside the function.
buffer_x = []
buffer_y = []
# NEW — live sliding-window smoothing buffer for pitch_signal, mirrors
# buffer_x/buffer_y exactly (trim-oldest-then-append every frame).
buffer_pitch = []


def run_camera(shared_data):
    """
    Thread A's job: runs forever in a background daemon thread.
    Reads webcam frames, runs MediaPipe FaceLandmarker on each one,
    and computes the current gaze ratios (t_x, t_y) using the
    interpolation formula t = (P - A) / (B - A).

    Every frame, writes the live t_x/t_y into shared_data so thread B
    (tkinter) can read the latest value at any time.

    shared_data is a dict created once in the main script and passed
    into both threads by reference — NOT recreated here — so both
    threads are reading/writing the same object in memory.
    """
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0

    # NEW — changed from `while True`. Checking shared_data["running"] lets
    # calibration_window.py signal this loop to stop cleanly once
    # calibration finishes, instead of relying on daemon=True to kill the
    # thread abruptly during interpreter shutdown. Before this change, the
    # thread could get torn out mid-frame (mid cv2/MediaPipe call) right as
    # the main program exited, throwing an "Exception in thread Thread-1"
    # on close. Now this loop gets a chance to finish its current iteration,
    # check the flag, and exit gracefully on its own before shutdown starts
    while shared_data["running"] == True:
        ret, frame = cap.read()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks:
            # NEW — nose tip landmark (index 1), needed as the moving
            # reference point for pitch_signal below. Head tilting
            # up/down shifts the nose tip's position relative to the
            # eyes more than it shifts the eyes themselves.
            nose_tip = result.face_landmarks[0][1]
            left_iris_center = result.face_landmarks[0][468]
            left_eye_outer_corner = result.face_landmarks[0][33]
            left_eye_inner_corner = result.face_landmarks[0][133]
            # t = (P - A) / (B - A): where the iris sits between outer(0) and inner(1) corner
            x_numerator = left_iris_center.x - left_eye_outer_corner.x
            x_denominator = left_eye_inner_corner.x - left_eye_outer_corner.x
            t_X = x_numerator / x_denominator

            # same idea vertically: upper eyelid(0) to lower eyelid(1)
            left_upper_eyelid = result.face_landmarks[0][159]
            left_lower_eyelid = result.face_landmarks[0][145]
            y_numerator = left_iris_center.y - left_upper_eyelid.y
            y_denominator = left_lower_eyelid.y - left_upper_eyelid.y
            t_Y = y_numerator / y_denominator

            # NEW — pitch_signal: a head-pose feature built to fix the
            # weak vertical signal in t_Y (eyelid gap barely moves with
            # pure up/down eye rotation, so t_Y couldn't reliably tell
            # top/mid/bottom rows apart). Formula derived the same way as
            # t_X/t_Y — a ratio using stable reference points so the
            # value isn't thrown off by moving closer to/further from
            # the camera:
            #   eye_level_y = midpoint between the two eye corners' y —
            #     a stable "how high are my eyes" reference that doesn't
            #     move with pitch itself
            #   eye_corner_distance = horizontal span between eye
            #     corners — the normalizer, same role as t_X/t_Y's
            #     denominators, cancels out camera distance
            #   pitch_signal = how far the nose tip sits below eye
            #     level, relative to that normalizer — grows as you tilt
            #     your chin down, shrinks as you tilt your chin up
            # Left-eye-only (not both eyes) since both eyes rotate
            # together — no signal gained from doubling the landmarks,
            # and a second eye is planned for the real (non-practice)
            # project later.
            # KNOWN LIMITATION: at extreme combined diagonal gaze (e.g.
            # looking hard bottom-right), eye_corner_distance gets
            # distorted since the eye corners no longer present a flat,
            # consistent horizontal span to the camera — confirmed via
            # repeated live testing to produce outlier pitch_signal
            # values specifically at that corner. Documented as an
            # accepted edge-case limitation, not fixed at the code level.
            eye_level_y = (left_eye_outer_corner.y + left_eye_inner_corner.y) / 2
            eye_corner_distance = abs(left_eye_outer_corner.x - left_eye_inner_corner.x)
            pitch_signal = (nose_tip.y - eye_level_y) / eye_corner_distance

            # SLIDING WINDOW SMOOTHING, runs every single frame,
            # unconditionally (not just during calibration capture).
            # Goal: reduce frame-to-frame jitter/noise in the raw t_X/t_Y
            # reading before anyone (calibration OR future cursor control)
            # ever sees it, by using the median of the last 15 frames
            # instead of a single frame's possibly-noisy value.
            #
            # Trim-then-append keeps this a rolling window that always
            # holds at most the 15 MOST RECENT frames. This is NOT the
            # same as filling to 15 and clearing — clearing would cause a
            # repeating "smooth, then sudden jump back to raw, then
            # smooth again" sawtooth pattern every 15 frames. Popping only
            # the single oldest reading before appending the new one keeps
            # the window sliding forward continuously with no resets.
            if len(buffer_x) >= 15:
                buffer_x.pop(0)
            if len(buffer_y) >= 15:
                buffer_y.pop(0)
            # NEW — same trim rule applied to pitch_signal's buffer
            if len(buffer_pitch) >= 15:
                buffer_pitch.pop(0)

            buffer_x.append(t_X)
            buffer_y.append(t_Y)
            # NEW — same append pattern for pitch_signal
            buffer_pitch.append(pitch_signal)

            # statistics.median() works fine even when the buffer has
            # fewer than 15 items (e.g. the very first few frames after
            # the program starts) — no crash, no special-casing needed.
            # It just returns partial smoothing until the window fills up,
            # which in practice takes well under a second at ~20-30fps —
            # faster than a user could realistically find the calibration
            # dot and press spacebar.
            buffer_x_median = statistics.median(buffer_x)
            buffer_y_median = statistics.median(buffer_y)
            # NEW — smoothed pitch_signal value, same median-of-window approach
            buffer_pitch_median = statistics.median(buffer_pitch)

            # this smoothed value is what thread B reads whenever it
            # checks the LIVE gaze position (separate from the
            # capture/median logic below, which does its own independent
            # 15-frame burst median from raw t_X/t_Y — deliberately NOT
            # reading from buffer_x_median/buffer_y_median here, to avoid
            # smoothing an already-smoothed value / blurring which stage
            # is responsible for what)
            shared_data["t_x"] = buffer_x_median
            shared_data["t_y"] = buffer_y_median
            # NEW — live smoothed pitch_signal, published the same way as t_x/t_y
            shared_data["pitch_signal"] = buffer_pitch_median

            # thread B flips "capturing" to True when spacebar is pressed.
            # while True, buffer up raw samples instead of just using the
            # live value — once we have 15, take the median (resistant to
            # blinks/glances-away) and use THAT as the calibration result
            if (shared_data["capturing"] == True):
                capture_buffer_x.append(t_X)
                capture_buffer_y.append(t_Y)
                # NEW — pitch_signal joins the same one-shot capture burst
                # as t_X/t_Y, so each calibration point gets a matching
                # median pitch reading alongside its gaze reading
                capture_buffer_pitch_signal.append(pitch_signal)

                if (len(capture_buffer_y) == 15 and len(capture_buffer_x) == 15
                        and len(capture_buffer_pitch_signal) == 15):
                    # overwrite the live value with the finished median —
                    # this is what thread B will read once it sees
                    # "capturing" flip back to False
                    shared_data["t_x"] = statistics.median(capture_buffer_x)
                    shared_data["t_y"] = statistics.median(capture_buffer_y)
                    # NEW — finished burst median for pitch_signal, published
                    # the same way as t_x/t_y right above
                    shared_data["pitch_signal"] = statistics.median(capture_buffer_pitch_signal)

                    # signals thread B the median is ready to read
                    shared_data["capturing"] = False

                    # reset buffers so the NEXT capture (next calibration
                    # point) starts fresh instead of appending onto stale data
                    capture_buffer_y.clear()
                    capture_buffer_x.clear()
                    # NEW — clear pitch_signal's capture buffer too, same reason
                    capture_buffer_pitch_signal.clear()