import threading
import time

from Camera_thread import run_camera
from Zone_classifier import ZoneClassifier
from State_machine import CursorStateMachine
from Cursor_window import CursorWindow

shared_data = {
    "t_x": 0,
    "t_y": 0,
    "pitch_signal": 0,
    "eye_openness": 0,
    "blink_detected": False,
    "running": True,
}

camera_thread = threading.Thread(target=run_camera, args=(shared_data,), daemon=True)
camera_thread.start()

# --- Establish a personal pitch baseline before starting the UI ---
# scroll detection (state_machine.py's update_pitch) works off DEVIATION
# from a neutral "looking straight ahead" pitch_signal, not an absolute
# number — pitch_signal's actual value depends on face geometry and
# camera angle, which will differ from whatever value happened to show
# up during 03's calibration sessions. This is a simple v1 approach
# (wait for the buffer to fill, then read one live value) — NOT
# measured/tuned the rigorous way BLINK_THRESHOLD was in 04. If scroll
# triggers too easily or not easily enough in testing, this is the
# first place to revisit, the same way BLINK_FRAMES_REQUIRED needed
# revisiting after real measurement.
print("Establishing pitch baseline — look straight at the screen...")
# FIX 2 — warm-up bumped from 0.5s to 3s. Root cause found via live
# testing: two baseline runs minutes apart, same physical setup
# (confirmed with you directly, nothing moved), produced wildly
# different baselines (2.41 vs 1.30) and the second one required
# extreme head movement to trigger anything afterward. That pattern —
# same setup, different result, purely based on WHEN sampling started —
# points at webcam autoexposure/autofocus still ramping up in the first
# ~1-2 seconds after cv2.VideoCapture(0) opens, during which MediaPipe's
# landmark precision (and therefore pitch_signal) can be genuinely
# unstable. 0.5s wasn't enough clearance from that startup period. 3s
# gives the camera time to settle before we start trusting its readings.
time.sleep(3)

baseline_samples = []
sample_start = time.time()
while time.time() - sample_start < 2.0:
    baseline_samples.append(shared_data["pitch_signal"])
    time.sleep(0.1)
pitch_baseline = sum(baseline_samples) / len(baseline_samples)

# Print the spread, not just the average — this is the actual evidence
# for whether the samples were stable (small min-max spread, trustworthy
# baseline) or still bouncing around (large spread, still unstable,
# needs more warm-up or better lighting/positioning before running).
sample_spread = max(baseline_samples) - min(baseline_samples)
print(f"Baseline pitch_signal: {pitch_baseline} (averaged over {len(baseline_samples)} samples, "
      f"spread {sample_spread:.3f} — should be small, comparable to pitch's normal noise floor ~0.05-0.1)")

zc = ZoneClassifier(calibration_dir="../03-calibration-math")
print("Zone centroids:", zc.zone_centroids)
print("stdevs — gx:", zc.std_gx)
sm = CursorStateMachine(pitch_baseline=pitch_baseline, scroll_delta=0.15)

window = CursorWindow(shared_data, zc, sm, pitch_baseline)
window.run()