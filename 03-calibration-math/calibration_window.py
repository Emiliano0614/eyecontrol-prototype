import tkinter as tk
import json  # NEW — needed to save self.results to a .json file on disk

class CalibrationWindow:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.current_target_index = 0
        self.results = []
        self.calibration_step = 0

        # creates the actual tkinter window object — must exist before
        # we can ask it anything about screen size, or create a canvas on it
        self.root = tk.Tk()

        # IMPORTANT: tkinter has its own internal coordinate space ("points"),
        # which on Retina Macs is NOT the same as physical pixel count
        # (e.g. this printed 1470x956 even though the real display is 2560x1664).
        # We ask tkinter directly for its own truth here, instead of hardcoding
        # or trusting pyautogui/screeninfo, which reported numbers that didn't
        # match what tkinter actually draws with — that mismatch was the bug
        # that made the dot appear off-center earlier.
        print(self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        self.root.attributes("-fullscreen", True)

        # using tkinter's own reported screen size ensures our calculated
        # target positions always agree with the coordinate space tkinter
        # actually draws into — no mismatch, regardless of the real physical
        # resolution or display scaling on whatever machine this runs on
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # gets the center of the screen
        center_x = screen_width / 2
        center_y = screen_height / 2

        # NEW — two extra row references added when the grid expanded
        # from 9 points (3 rows) to 15 points (5 rows). Same
        # screen_height-based formula pattern as every other target
        # coordinate here, just at the 1/4 and 3/4 marks instead of
        # 0/half/full, so these sit evenly between the existing rows.
        # Added specifically to get more calibration samples per row,
        # to test whether pitch_signal's row separation would tighten
        # up with more data.
        upper_mid_y = screen_height / 4
        lower_mid_y = screen_height * 3 / 4

        # top row (existing, y=0)
        top_left_x = 0
        top_left_y = 0
        top_middle_x = screen_width / 2
        top_middle_y = 0
        top_right_x = screen_width
        top_right_y = 0

        # NEW row — upper-mid (y = screen_height / 4)
        upper_left_x = 0
        upper_middle_x = screen_width / 2
        upper_right_x = screen_width

        # center row (existing, y = screen_height / 2)
        center_left_x = 0
        center_left_y = screen_height / 2
        center_right_x = screen_width
        center_right_y = screen_height / 2

        # NEW row — lower-mid (y = screen_height * 3 / 4)
        lower_left_x = 0
        lower_middle_x = screen_width / 2
        lower_right_x = screen_width

        # bottom row (existing, y = screen_height)
        bottom_left_x = 0
        bottom_left_y = screen_height
        bottom_center_x = screen_width / 2
        bottom_center_y = screen_height
        bottom_right_x = screen_width
        bottom_right_y = screen_height

        # NEW — grid expanded from 9 entries (3 rows x 3 cols) to 15
        # entries (5 rows x 3 cols): the 6 new upper_*/lower_* tuples
        # appended at the end. Order doesn't need to match a visual
        # top-to-bottom sweep — current_target_index just walks this
        # list in whatever order it's written, draw_target() and
        # check_capture_done() don't care about row/column position,
        # only about index.
        self.calibration_targets = [
            (center_x, center_y),
            (top_middle_x, top_middle_y),
            (top_left_x, top_left_y),
            (top_right_x, top_right_y),
            (center_left_x, center_left_y),
            (center_right_x, center_right_y),
            (bottom_left_x, bottom_left_y),
            (bottom_center_x, bottom_center_y),
            (bottom_right_x, bottom_right_y),
            (upper_left_x, upper_mid_y),
            (upper_middle_x, upper_mid_y),
            (upper_right_x, upper_mid_y),
            (lower_left_x, lower_mid_y),
            (lower_middle_x, lower_mid_y),
            (lower_right_x, lower_mid_y),
        ]

        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # spacebar triggers start_capture — no lambda needed here since
        # start_capture is a real method (lambdas can't contain assignment
        # statements, only expressions, so this couldn't be a one-line lambda
        # like the Escape binding below)
        self.root.bind("<space>", self.start_capture)

        # bind Escape key to close the window, just for testing
        self.root.bind("<Escape>", lambda event: self.root.destroy())

    def check_capture_done(self):
        """
        Polls shared_data["capturing"] without blocking the tkinter event
        loop. If thread A (camera_thread.py) is still mid-capture, this
        reschedules itself 50ms later via .after() instead of using a
        blocking while loop — mainloop() needs to keep running the whole
        time, so we can't just sit and wait here.

        Once thread A flips "capturing" back to False, the median it
        computed is sitting in shared_data["t_x"]/["t_y"], ready to read.
        """
        if (self.shared_data["capturing"] is True):
            # still capturing — check again in 50ms instead of blocking
            self.root.after(50, self.check_capture_done)
        else:
            # thread A finished — the median is ready in shared_data
            median_X = self.shared_data["t_x"]
            median_Y = self.shared_data["t_y"]
            # NEW — pitch_signal's finished median, published by
            # camera_thread.py the same way t_x/t_y are
            median_pitch_signal = self.shared_data["pitch_signal"]

            #  grab the (x, y) of whichever target this capture was
            # taken at. Must happen BEFORE current_target_index gets
            # incremented below, or this would pair the median with the
            # NEXT target instead of the one the user actually just looked at
            target_x, target_y = self.calibration_targets[self.current_target_index]

            #  bundle the gaze reading with the screen truth it belongs
            # to, and save that pair. This is the actual training data
            # solve_mapping.py will use later — without pairing gaze to a
            # known screen position, there'd be nothing to fit a formula to
            # NEW — median_pitch_signal appended as a 5th element, after
            # the original 4. Appending at the end (not inserting between
            # existing values) keeps the first four positions unchanged,
            # so nothing that already reads target_x/target_y by index
            # breaks.
            self.results.append((median_X, median_Y, target_x, target_y, median_pitch_signal))

            #  moved BEFORE the >8 check (previously this was after,
            # which caused calibration_targets[9] to be accessed on the
            # last point — an IndexError, since valid indices only go 0-8)
            self.current_target_index += 1

            # NEW — boundary changed from `> 8` to `> 14` when the grid
            # expanded from 9 targets (valid indices 0-8) to 15 targets
            # (valid indices 0-14). Same increment-then-check logic as
            # before: after the last point (index 14) is captured and
            # incremented, current_target_index becomes 15, and
            # 15 > 14 correctly triggers the finish branch. Traced
            # through explicitly to avoid repeating the original 8/9
            # off-by-one bug at the new boundary.
            if (self.current_target_index > 14 ):
                with open(f"calibration_run{self.calibration_step}.json", "w") as f:
                    json.dump(self.results, f)
                self.results = []
                self.current_target_index = 0
                self.calibration_step += 1
                if (self.calibration_step > 3 ): 
                    print("Finished Calibration")
                    # tells camera_thread.py's run_camera loop to stop on
                    # its NEXT iteration, BEFORE the window actually closes.
                    # Must be set before self.root.destroy(), so thread A gets a
                    # chance to exit cleanly instead of getting cut off mid-frame
                    # when the whole program starts shutting down
                    self.shared_data["running"] = False
                    self.root.destroy()
                else:
                    self.draw_target()
                # write all 15 (gaze, screen, pitch) tuples to disk as JSON so
                # they survive after this script exits (Python has no memory
                # between runs — without this, all captured data would be lost
                # the moment the window closes)
            else:
                self.draw_target()

            print("capture done, median:", self.shared_data["t_x"], self.shared_data["t_y"])

    def start_capture(self, event):
        """
        Fired when spacebar is pressed. tkinter automatically passes an
        'event' object to any bound function, so this needs to accept it
        as a parameter even though we don't use it here.

        Sets the flag thread A is watching for, then immediately kicks off
        the first check_capture_done() call to start polling for when
        thread A finishes collecting its 15 frames.
        """
        self.shared_data["capturing"] = True
        self.check_capture_done()
        print("space pressed, capturing set to True:", self.shared_data)

    def run(self):
        # blocks here until the window is closed — nothing after this
        # call runs until then, which is why draw_target() must be
        # called BEFORE run(), not after
        self.root.mainloop()

    def draw_target(self):
        # looks up the (x, y) position for whichever target is currently
        # active, based on current_target_index
        self.x, self.y = self.calibration_targets[self.current_target_index]

        # radius bumped from 10 to 55. At true screen edges (y=0 or
        # y=screen_height) a small dot got clipped/invisible — confirmed via
        # live testing that macOS's auto-hide Dock reserves a sliver of space
        # at the screen edge even when hidden. Kept the target COORDINATES
        # at the true edges (best for calibration accuracy) and just made
        # the visible circle bigger so enough of it always pokes into view,
        # instead of insetting the coordinates (which would be an arbitrary
        # hardcoded pixel offset unrelated to screen size)
        self.raduis = 55

        # create_oval needs a bounding box (x0,y0 top-left, x1,y1 bottom-right),
        # not a center point + radius — so we compute the box from the center
        # point and radius ourselves
        self.x0 = self.x - self.raduis
        self.x1 = self.x + self.raduis
        self.y0 = self.y - self.raduis
        self.y1 = self.y + self.raduis

        # clears the previous dot BEFORE drawing the new one. Must
        # come before create_oval, not after — clearing after would wipe
        # out the dot that was just drawn, leaving nothing visible at all
        self.canvas.delete("all")

        # create_oval belongs to the canvas object, not to CalibrationWindow
        # itself — that's why it's self.canvas.create_oval(...), not just
        # self.create_oval(...)
        self.canvas.create_oval(self.x0, self.y0, self.x1, self.y1)