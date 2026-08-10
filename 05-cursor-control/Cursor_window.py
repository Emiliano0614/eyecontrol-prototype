import tkinter as tk


class CursorWindow:
    """
    Continuously polls shared_data (gaze/pitch/blink) every POLL_MS via
    root.after(), same non-blocking pattern calibration_window.py used
    for check_capture_done — but here it never stops polling, since 05
    isn't a one-shot capture sequence, it's a live running loop for as
    long as the program is open.

    2-ZONE REDESIGN NOTE: USE_DEFAULT_CENTER and the alternate
    predict_zone_default_center classifier are gone — that workaround
    existed only because A/B were ambiguous under a 3-way split. With
    A+B merged into one LEFT zone at the classifier level, there's no
    more ambiguous middle zone to bias toward, so the plain nearest-
    centroid predict_zone() is the only classifier needed now.
    """

    POLL_MS = 50  # matches calibration_window.py's polling interval

    def __init__(self, shared_data, zone_classifier, state_machine, pitch_baseline):
        self.shared_data = shared_data
        self.zc = zone_classifier
        self.sm = state_machine
        self.pitch_baseline = pitch_baseline
        self.last_gazed_zone = 0  # which of the 2 boxes gaze currently classifies into

        self.root = tk.Tk()
        self.root.title("EyeControl - 05 Cursor Control")

        self.root.attributes("-fullscreen", True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.canvas = tk.Canvas(self.root, bg="black", width=screen_width, height=screen_height * 0.8)
        self.canvas.pack(fill="both", expand=True)

        self.status_label = tk.Label(
            self.root, text="", font=("Courier", 18), anchor="w", justify="left"
        )
        self.status_label.pack(fill="x", padx=10, pady=5)

        self.root.bind("<Escape>", lambda event: self._quit())

        # Box layout: 2 equal-width boxes spanning the FULL real screen
        # width, matching the 2-zone left/right geometry the classifier
        # was trained on (see zone_classifier.py's get_zone — old A+B
        # merged into LEFT, old C is RIGHT).
        self.screen_width = screen_width
        self.box_width = screen_width / 2
        self.box_height = screen_height * 0.75

    def _quit(self):
        self.shared_data["running"] = False
        self.root.destroy()

    def poll(self):
        gaze_x = self.shared_data["t_x"]
        gaze_y = self.shared_data["t_y"]
        pitch = self.shared_data["pitch_signal"]
        eye_openness = self.shared_data["eye_openness"]

        mode = self.sm.update_pitch(pitch)
        delta = pitch - self.pitch_baseline

        if mode == "SELECT":
            zone = self.zc.predict_zone(gaze_x, gaze_y, pitch, self.pitch_baseline)
            self.last_gazed_zone = zone[0]  # zone is (zone_x, 0) — see zone_classifier.get_zone
            print(f"MODE={mode} pitch={pitch:.3f} delta={delta:+.3f} eye_openness={eye_openness:.3f} gaze_x={gaze_x:.3f} -> zone={zone}")

            if self.shared_data["blink_detected"]:
                self.shared_data["blink_detected"] = False  # reset so it only fires once
                result = self.sm.confirm_zone(self.last_gazed_zone)
                print(f"BLINK CONFIRMED -> zone={self.last_gazed_zone} result={result} path={self.sm.path} typed_input={self.sm.typed_input}")
        else:   
            print(f"MODE={mode} pitch={pitch:.3f} delta={delta:+.3f} eye_openness={eye_openness:.3f}")
            self.shared_data["blink_detected"] = False

        self._draw()
        self.root.after(self.POLL_MS, self.poll)

    def _draw(self):
        self.canvas.delete("all")
        options = self.sm.current_options()

        for zone_index in range(2):
            x0 = zone_index * self.box_width
            x1 = x0 + self.box_width
            label = options.get(zone_index, "")

            is_gazed = (self.sm.mode == "SELECT" and zone_index == self.last_gazed_zone)
            fill = "#3a3a3a" if is_gazed else "#111111"

            self.canvas.create_rectangle(
                x0 + 5, 5, x1 - 5, self.box_height,
                fill=fill, outline="#666666", width=2
            )
            self.canvas.create_text(
                (x0 + x1) / 2, self.box_height / 2,
                text=label, fill="white", font=("Courier", 28)
            )

        status = (
            f"MODE: {self.sm.mode}"
            f"{'  (' + self.sm.scroll_direction + ')' if self.sm.mode == 'SCROLL' else ''}"
            f"   DEPTH: {len(self.sm.path)}"
            f"   INPUT: {self.sm.typed_input}"
        )
        self.status_label.config(text=status)

    def run(self):
        self.poll()  # kicks off the recurring poll loop before mainloop blocks
        self.root.mainloop()