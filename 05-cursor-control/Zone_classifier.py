import json
import os
import statistics

ZONE_COUNT = 2
OLD_ZONE_COUNT = 3
zone_width = 1470 / OLD_ZONE_COUNT
zone_height = 956


def get_zone(x, y):
    old_zone_x = min(int(x // zone_width), OLD_ZONE_COUNT - 1)
    new_zone_x = 0 if old_zone_x in (0, 1) else 1
    return (new_zone_x, 0)


class ZoneClassifier:
    """
    Nearest-centroid classifier. LEFT vs RIGHT, on gaze_x ONLY.

    PITCH REMOVED (previous fix): pitch_rel was contributing near-equal
    weight to gaze_x in the distance calc, causing flips from head tilt
    alone with gaze_x barely moving. See earlier commit/comment.

    GAZE_Y REMOVED (this fix): same category of bug, different feature.
    Live testing found points where gaze_x clearly favored one zone
    (e.g. distance 0.15 vs 1.61 — not close) but gaze_y's opposing vote
    dragged the total distance into a near-tie (1.623 vs 1.629),
    letting tiny gaze_y noise flip the result. LEFT/RIGHT is a
    horizontal-only decision; gaze_y has no business voting on it,
    same reasoning as removing pitch. Classifying on gaze_x alone
    removes the tie entirely.

    calibration_dir: folder containing calibration_run0.json ...
        calibration_run3.json.
    """

    def __init__(self, calibration_dir="../03-calibration-math", num_runs=4):
        self.zone_centroids = {}
        zone_readings = {}
        all_gx = []

        for i in range(num_runs):
            path = os.path.join(calibration_dir, f"calibration_run{i}.json")
            with open(path) as f:
                run_data = json.load(f)

            for row in run_data:
                gaze_x, gaze_y, target_x, target_y, pitch_signal = row
                zone = get_zone(target_x, target_y)
                zone_readings.setdefault(zone, []).append(gaze_x)
                all_gx.append(gaze_x)

        for zone, readings in zone_readings.items():
            self.zone_centroids[zone] = sum(readings) / len(readings)

        if len(self.zone_centroids) != ZONE_COUNT:
            raise ValueError(
                f"Expected {ZONE_COUNT} zone centroids, got "
                f"{len(self.zone_centroids)}. Check calibration_dir and "
                f"that all {num_runs} calibration_run*.json files are present."
            )

        self.std_gx = statistics.stdev(all_gx)

    def predict_zone(self, gaze_x, gaze_y=None, pitch_signal=None, pitch_baseline=None):
        """
        gaze_y/pitch_signal/pitch_baseline kept as accepted (ignored)
        args so callers (cursor_window.py) don't need to change their
        call signature. Classifies purely on gaze_x now.
        """
        best_zone = None
        best_dist = float("inf")
        for zone, cx in self.zone_centroids.items():
            dist = ((gaze_x - cx) / self.std_gx) ** 2
            if dist < best_dist:
                best_dist = dist
                best_zone = zone
        return best_zone