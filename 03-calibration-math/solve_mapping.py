import json
import numpy as np

# --- Load calibration (training) data ---
# 4 separate calibration sweeps were recorded to average out run-to-run
# inconsistency (a single 15-point sweep wasn't stable enough on its own).
# .extend() is used instead of .append() so all 4 files flatten into one
# single list of rows, rather than nesting into a list of 4 separate lists.
train_data = []
for i in range(4):
    train_data.extend(json.load(open(f"calibration_run{i}.json")))

# Held-out data, collected in a totally separate session, used ONLY for
# evaluation. Never used to build zone centroids. This is what makes the
# hit rate below a genuine generalization test, not a same-data test.
test_data = json.load(open('calibration_results_test.json'))

# --- Zone grid configuration ---
# ZONE_COUNT is the single source of truth for how many horizontal zones
# the screen is split into. zone_width and the cap in get_zone() both
# derive from it, so changing zone granularity only requires editing
# this one number (this used to be two numbers that had to be kept in
# sync by hand, which caused a real bug: zone_width was updated to test
# 4 zones but the cap was left at the old 3-zone value, silently merging
# two real zones into one and making test results look better than they
# actually were).
ZONE_COUNT = 3  # confirmed clean via testing: 3 horizontal zones classify
                # as well as 2 did (60% hit rate both times), so the
                # horizontal signal has real headroom here. 4 zones was
                # attempted but isn't a valid test yet — the calibration
                # grid only has 3 real x-columns (0, 735, 1470), so no
                # calibration data exists for a 4th zone's region. Testing
                # 4 zones for real would require re-collecting calibration
                # data with a 4-column target grid first.

zone_width = 1470 / ZONE_COUNT

# Only one row for now — vertical is the known weak axis (eyelid-gap
# signal is inherently faint for up/down eye rotation), so the vertical
# split isn't being tested yet. zone_height spans the full screen height,
# meaning every point currently counts as the same row regardless of y.
zone_height = 956


def get_zone(x, y):
    """
    Convert a screen coordinate into a (zone_x, zone_y) zone label.
    Used two ways in this file:
      1. On calibration target coordinates, to group calibration
         readings into zones for building centroids.
      2. On test target coordinates, to get the ground-truth zone
         label to check predictions against.
    NOT used for live classification — that's predict_zone()'s job.
    """
    # Floor division buckets x into 0, 1, ... but the maximum x value
    # (1470) floor-divides exactly to ZONE_COUNT, one past the last
    # valid zone index (zones are 0-indexed). The min(..., ZONE_COUNT-1)
    # clamp pulls that edge case back into the last real zone instead
    # of silently creating a phantom extra zone.
    zone_x = min(int(x // zone_width), ZONE_COUNT - 1)
    zone_y = 0  # single row until vertical zones are added
    return (zone_x, zone_y)


# --- Build zone centroids from calibration data ---
# For each zone, average all calibration readings that landed in it into
# one representative (gaze_x, gaze_y, pitch_signal) centroid. This is the
# entire "model" — no regression coefficients, no curve fitting. Nearest-
# centroid was adopted after lstsq regression (linear and polynomial) was
# root-caused to be fundamentally unstable here: raw gaze measurement
# noise within a single calibration target is nearly as large as the real
# signal range across the whole screen, which makes any coefficient-based
# fit unreliable regardless of which terms are included.
zone_readings = {}
for row in train_data:
    gaze_x, gaze_y, target_x, target_y, pitch_signal = row
    zone = get_zone(target_x, target_y)
    zone_readings.setdefault(zone, []).append((gaze_x, gaze_y, pitch_signal))

zone_centroids = {}
for zone, readings in zone_readings.items():
    avg_gaze_x = sum(r[0] for r in readings) / len(readings)
    avg_gaze_y = sum(r[1] for r in readings) / len(readings)
    avg_pitch = sum(r[2] for r in readings) / len(readings)
    zone_centroids[zone] = (avg_gaze_x, avg_gaze_y, avg_pitch)

# --- Diagnostic output ---
# Sanity-check block: confirms the calibration grid's real x-columns,
# and that every expected zone got a roughly even number of calibration
# points. If a zone is missing or lopsided, something upstream (grid
# setup or the zone math above) is wrong. Comment out once trusted.
print("Calibration target x-values:", sorted(set(row[2] for row in train_data)))
print("Test target x-values:       ", sorted(set(row[2] for row in test_data)))
print("Points per zone:            ", {z: len(r) for z, r in zone_readings.items()})
print("Zones with a centroid:      ", list(zone_centroids.keys()))


def predict_zone(gaze_x, gaze_y, pitch_signal):
    """
    Classify a live gaze reading by finding its nearest centroid
    (squared Euclidean distance across gaze_x, gaze_y, pitch_signal).
    No regression, no coefficients — just "which zone's average
    reading is this closest to."
    """
    best_zone = None
    best_dist = float('inf')
    for zone, (cx, cy, cp) in zone_centroids.items():
        dist = (gaze_x - cx) ** 2 + (gaze_y - cy) ** 2 + (pitch_signal - cp) ** 2
        if dist < best_dist:
            best_dist = dist
            best_zone = zone
    return best_zone


# --- Evaluate on held-out test data ---
hits = 0
misses = 0
for row in test_data:
    gaze_x, gaze_y, actual_x, actual_y, pitch_signal = row
    actual_zone = get_zone(actual_x, actual_y)  # ground truth, from known target position
    predicted_zone = predict_zone(gaze_x, gaze_y, pitch_signal)  # model's guess, from gaze reading
    if actual_zone == predicted_zone:
        hits += 1
    else:
        misses += 1

print(f"HITS: {hits}  MISSES: {misses}  ({hits/(hits+misses)*100:.0f}% hit rate on unseen test data)")