import threading
import time
from camera_thread import run_camera
from calibration_window import CalibrationWindow

# shared_data is created ONCE, here, and passed by reference into both
# threads below — this is the one dict both threads read/write, which
# is what lets thread A's live gaze data reach thread B's tkinter window
shared_data = {
    "t_x": 0,
    "t_y": 0,
    # NEW — seeded here for the same reason t_x/t_y/capturing/running
    # are: camera_thread.py writes to shared_data["pitch_signal"] every
    # frame starting on the very first frame, so the key needs to exist
    # in the dict before that first write happens.
    "pitch_signal": 0,
    "capturing": False,
    "running": True
}

# thread A: runs run_camera() in the background, forever, reading the
# webcam and updating shared_data every frame. daemon=True means this
# thread gets killed automatically when the main program exits — it
# doesn't need its own break condition, since mainloop() below is what
# actually controls when the whole program ends.
# the trailing comma in args=(shared_data,) matters: without it, Python
# wouldn't treat this as a tuple, and Thread wouldn't accept it correctly
camera_thread = threading.Thread(target=run_camera, args=(shared_data,), daemon=True)
camera_thread.start()

# thread B: tkinter's mainloop() runs directly on the main thread (not
# spawned separately) — this is the actual main thread of the whole
# program, and it's what run() will block on
cal = CalibrationWindow(shared_data)

# draws the first (center) target dot BEFORE run() is called, since
# run() blocks on mainloop() and nothing after it executes until the
# window closes
cal.draw_target()
cal.run()