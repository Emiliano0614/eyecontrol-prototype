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

landmarker = vision.FaceLandmarker.create_from_options(options)

# --- Blink detection tuning (unchanged from 04) ---
# BLINK_THRESHOLD=0.10 and BLINK_FRAMES_REQUIRED=7 were both derived
# from real measured data in 04, not guessed — see 04's
# camera_thread.py comments for the full derivation (BLINK_FRAMES_REQUIRED
# specifically was retuned from an assumed-30fps value of 15 down to 7,
# after directly measuring this loop's real frame rate at ~13fps).
#
# IMPORTANT: this loop does MORE per frame than 04's did (gaze + pitch
# math on top of the eyelid-gap math), so it will likely be even slower
# than 04's measured ~13fps. BLINK_FRAMES_REQUIRED=7 should be
# re-verified live in this file specifically (hold a deliberate blink,
# time it, confirm it still lands close to ~0.5s) rather than assumed
# to carry over unchanged, the same way 04 didn't trust the original
# 30fps assumption without measuring.
BLINK_THRESHOLD = 0.16
BLINK_FRAMES_REQUIRED = 3
# How many consecutive "clearly open" frames are needed after a blink
# fires before we'll start counting toward the NEXT blink. This is what
# stops a noisy 40-80 frame semi-closed stretch from re-triggering
# blink_detected multiple times on the way through — a single frame
# poking back above BLINK_THRESHOLD is not proof the eye actually
# reopened, so we require a short sustained run of open frames instead.
# 3 is a starting guess (unverified the rigorous way BLINK_THRESHOLD
# was) — revisit if legit rapid double-blinks stop registering.
REOPEN_FRAMES_REQUIRED = 3
# PITCH-FREEZE FIX: separate, higher threshold from BLINK_THRESHOLD.
#
# BLINK_THRESHOLD (0.10) is tuned to CONFIRM a blink is happening —
# it's deliberately strict/low so a normal open eye (eye_openness
# ~0.4-0.6 in this file's own logged data) never falsely counts as
# closed.
#
# But pitch_signal depends on eye_level_y, which is built from
# left_eye_outer_corner.y and left_eye_inner_corner.y — landmarks
# that shift as the eyelid physically lowers, BEFORE eye_openness
# drops anywhere near 0.10. Live testing confirmed this directly:
# pitch_signal was seen climbing steadily (e.g. 2.13 -> 2.25, a
# +0.10+ delta swing) while eye_openness was still only easing down
# through 0.5 -> 0.44, well above BLINK_THRESHOLD. By the time
# eye_openness actually crossed 0.10 and the buffer-skip kicked in,
# pitch_signal had already drifted up and got FROZEN at that already-
# elevated value — which was then enough on its own to cross
# scroll_delta and flip SELECT -> SCROLL, even with the original
# buffer-skip fix working exactly as designed.
#
# Fix: freeze the smoothing buffers earlier, at the first sign of the
# eyelid lowering, rather than waiting until the eye is nearly shut.
# 0.35 is a starting estimate, not yet rigorously measured the way
# BLINK_THRESHOLD was in 04 — it sits below the ~0.44-0.5+ range this
# file's own logs show for a normally open eye, but above the point
# a full blink is already close to complete. Revisit with the same
# real-data derivation BLINK_THRESHOLD got if pitch still drifts
# before the freeze catches it, or if the freeze now catches too much
# (e.g. firing during normal blinking-adjacent eye movement that was
# never going to distort pitch_signal in the first place).
PITCH_FREEZE_THRESHOLD = 0.35
# --- Live smoothing buffers (unchanged pattern from 03) ---
# Module-level, not inside run_camera(), so they persist across every
# frame of the whole program's lifetime — same reasoning as 03.
buffer_x = []
buffer_y = []
buffer_pitch = []


def run_camera(shared_data):
    """
    Merged version of 03's and 04's run_camera(): one webcam read, one
    detect_for_video() call per frame, feeding BOTH the gaze/pitch
    pipeline (03) and the blink pipeline (04) from the same landmark
    result. 05 doesn't need calibration-burst capturing (capturing/
    capture_buffer_x/y/pitch from 03) since it reuses existing
    calibration_run*.json centroids via zone_classifier.py instead of
    collecting new ones — so that logic is deliberately left out here,
    not forgotten.

    Expects/writes the same shared_data keys as before:
      shared_data["running"]        - controls the loop (set False to stop)
      shared_data["t_x"]            - live smoothed horizontal gaze ratio
      shared_data["t_y"]            - live smoothed vertical gaze ratio
      shared_data["pitch_signal"]   - live smoothed head-pitch signal
      shared_data["eye_openness"]   - live eye_openness value (debug/tuning)
      shared_data["blink_detected"] - flips True for one blink; caller
                                       resets it back to False after handling
    """
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0
    closed_frame_count = 0
    open_frame_count = 0
    blink_armed = True  # False right after a blink fires, until reopen confirmed
    while shared_data["running"] == True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Same caveat as both source files: +=33 assumes 30fps and is
        # only used as a strictly-increasing fake timestamp for
        # MediaPipe's VIDEO mode — it does not reflect real elapsed time.
        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # --- Landmarks (same indices as 03 and 04) ---
            nose_tip = landmarks[1]
            left_iris_center = landmarks[468]
            left_eye_outer_corner = landmarks[33]
            left_eye_inner_corner = landmarks[133]
            left_upper_eyelid = landmarks[159]
            left_lower_eyelid = landmarks[145]

            # --- Gaze ratio: t = (P - A) / (B - A), from 03 ---
            x_numerator = left_iris_center.x - left_eye_outer_corner.x
            x_denominator = left_eye_inner_corner.x - left_eye_outer_corner.x
            t_X = x_numerator / x_denominator

            y_numerator = left_iris_center.y - left_upper_eyelid.y
            y_denominator = left_lower_eyelid.y - left_upper_eyelid.y
            t_Y = y_numerator / y_denominator

            # --- pitch_signal, from 03 ---
            eye_level_y = (left_eye_outer_corner.y + left_eye_inner_corner.y) / 2
            eye_corner_distance = abs(left_eye_outer_corner.x - left_eye_inner_corner.x)
            pitch_signal = (nose_tip.y - eye_level_y) / eye_corner_distance

            # --- eye_openness, from 04 (reuses eye_corner_distance
            # already computed above for pitch_signal — 04 originally
            # computed this separately since it never had pitch_signal
            # in the same file, but the value is identical, so no need
            # to compute it twice) ---
            eye_gap = abs(left_upper_eyelid.y - left_lower_eyelid.y)
            eye_openness = eye_gap / eye_corner_distance

            # --- Sliding window smoothing for gaze/pitch, from 03 ---
            # BLINK-CORRUPTION FIX: skip feeding this frame's gaze/pitch
            # into the smoothing buffers if the eye is currently closed.
            #
            # Root cause: eye_corner and eyelid landmarks (used for BOTH
            # pitch_signal and t_X/t_Y) become unreliable mid-blink — the
            # face mesh doesn't track a closing/closed eyelid as cleanly as
            # an open one, so pitch_signal can spike for the duration of
            # the blink, not just a single frame.
            #
            # Why the median didn't already handle this: BLINK_FRAMES_REQUIRED
            # is 7, meaning a real blink holds eye_openness below threshold
            # for 7+ consecutive frames. The buffer is only 15 frames long.
            # A median is only robust against outliers that are a MINORITY
            # of the window — once 7+ out of 15 frames are blink-affected,
            # that's nearly half the window, which is enough to drag the
            # median itself, not just get filtered out as noise. That's why
            # pitch_signal was spiking hard enough to flip SELECT -> SCROLL
            # on every blink instead of the blink cleanly resolving.
            #
            # Fix: don't let those unreliable frames into the buffer at all.
            # We already computed eye_openness above for blink-frame-counting,
            # so we reuse it here rather than compute anything new.

            # PITCH-FREEZE FIX: this now checks PITCH_FREEZE_THRESHOLD,
            # NOT BLINK_THRESHOLD. BLINK_THRESHOLD stays reserved for
            # blink CONFIRMATION below (closed_frame_count / blink_detected)
            # — that logic is untouched and still needs the strict low
            # bar so a normal open eye never counts as a blink. This
            # buffer-skip check has a different job (stop trusting
            # pitch_signal/t_X/t_Y before they visibly drift) and so it
            # deliberately uses a different, higher bar. Using the same
            # constant for both jobs was the root cause of the freeze
            # kicking in too late — see PITCH_FREEZE_THRESHOLD comment
            # above for the live-data evidence.
            if eye_openness >= PITCH_FREEZE_THRESHOLD:
                if len(buffer_x) >= 15:
                    buffer_x.pop(0)
                if len(buffer_y) >= 15:
                    buffer_y.pop(0)
                if len(buffer_pitch) >= 15:
                    buffer_pitch.pop(0)

                buffer_x.append(t_X)
                buffer_y.append(t_Y)
                buffer_pitch.append(pitch_signal)

                # shared_data still updates every frame — even on a skipped-append
                # frame, this just re-reports the last good median rather than
                # freezing shared_data itself. If the buffers are ever empty
                # (e.g. very first frame or a blink at program start before any
                # good frames landed), statistics.median() will raise — not a
                # concern in practice since the pitch-baseline warm-up in main.py
                # runs for several seconds before this ever gets read, but worth
                # knowing if this gets rebuilt from scratch.
                shared_data["t_x"] = statistics.median(buffer_x)
                shared_data["t_y"] = statistics.median(buffer_y)
                shared_data["pitch_signal"] = statistics.median(buffer_pitch)
            # <-- "if eye_openness >= BLINK_THRESHOLD:" BLOCK ENDS HERE.
            # Everything below is back OUT to the same indent level as that
            # "if" line itself (i.e. still INSIDE "if result.face_landmarks:",
            # just a sibling of the buffer-smoothing block, not nested in it),
            # so it runs every frame regardless of whether the eye is open or
            # closed this frame — blink counting must never be skipped.

            # --- Blink detection, from 04 (still uses RAW eye_openness,
            # unsmoothed, unaffected by the buffer change above — the
            # buffer skip only protects t_x/t_y/pitch_signal, not this) ---
            shared_data["eye_openness"] = eye_openness

            if eye_openness < BLINK_THRESHOLD:
                open_frame_count = 0
                if blink_armed:
                    closed_frame_count += 1
                # if not armed, ignore further closure until reopen — prevents
                # re-triggering mid-hold
            else:
                closed_frame_count = 0
                open_frame_count += 1
                if open_frame_count >= REOPEN_FRAMES_REQUIRED:
                    blink_armed = True

            if blink_armed and closed_frame_count == BLINK_FRAMES_REQUIRED:
                shared_data["blink_detected"] = True
                blink_armed = False
                closed_frame_count = 0

    cap.release()