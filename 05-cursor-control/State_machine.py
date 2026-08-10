import Drill_tree as dt


class CursorStateMachine:
    """
    Owns two concerns:
      1. SCROLL vs SELECT mode, driven by pitch_signal magnitude relative
         to a neutral baseline captured at startup (see NOTE below —
         this is deliberately NOT a hardcoded magic number, the same way
         BLINK_THRESHOLD in 04 wasn't guessed, it was measured).
      2. Within SELECT mode, position in dt.TREE, tracked as a list of
         zone_index choices made so far (the "path" from the root).

    2-ZONE REDESIGN NOTE: the old version tracked level/row_context/
    pair_context by name, matching a hand-built 3-way tree (Row A/B/C).
    With only 2 zones, the tree is a generic binary tree of arbitrary
    depth (see drill_tree.py), so instead of named levels we just track
    self.path — the sequence of 0/1 choices from the root — and walk
    dt.TREE with it. No BACK/cancel path for now (dropped — no free
    3rd zone to hide it in anymore).

    This class does NOT touch tkinter or the camera thread directly —
    main.py reads shared_data every frame and calls into this class,
    keeping the state machine testable on its own (see
    test_state_machine.py) without a webcam or a display attached.
    """

    def __init__(self, pitch_baseline, scroll_delta=0.15):
        # NOTE on scroll_delta: this is a placeholder, not a measured
        # constant like BLINK_THRESHOLD/BLINK_FRAMES_REQUIRED were.
        # Those were derived from real captured data (see 04's
        # comments). This has NOT been through that same process yet —
        # it needs the same treatment: capture real pitch_signal
        # readings during deliberate up/down head tilts, look at where
        # "clearly tilting" separates from "normal small head movement
        # while reading the screen," and set scroll_delta from that,
        # the same way BLINK_THRESHOLD was set just above the observed
        # floor of a real closed eye. Flagged here so it doesn't get
        # forgotten and quietly treated as trustworthy.
        self.pitch_baseline = pitch_baseline
        self.scroll_delta = scroll_delta

        self.mode = "SELECT"  # "SELECT" or "SCROLL"
        self.scroll_direction = None
        self.path = []         # list of 0/1 choices from the root of dt.TREE
        self.typed_input = ""

    def update_pitch(self, pitch_signal):
        """
        Call every frame with the live (smoothed) pitch_signal reading.
        Returns the resulting mode string, and handles the
        scroll-discards-drill-progress rule agreed on during design.
        """
        delta = pitch_signal - self.pitch_baseline
        was_select = (self.mode == "SELECT")

        if abs(delta) > self.scroll_delta:
            self.mode = "SCROLL"
            self.scroll_direction = "DOWN" if delta > 0 else "UP"
            # pitch_signal grows as chin tilts down (see camera_thread.py
            # comment on pitch_signal in 03) — larger delta = looking
            # down = scroll down. Matches natural head-tilt-to-scroll
            # intuition.
            if was_select:
                # Entering SCROLL from mid-drill: discard progress rather
                # than trying to preserve a half-selected path across
                # a mode switch — agreed as the simplest, safest v1
                # behavior.
                self._reset_drill()
        else:
            self.mode = "SELECT"
            self.scroll_direction = None

        return self.mode

    def _reset_drill(self):
        self.path = []

    def _current_node(self):
        """
        Walk dt.TREE using self.path and return whatever node we land
        on — either a branch (dict) or a leaf (str). Internal helper;
        current_options() and confirm_zone() both need this.
        """
        node = dt.TREE
        for zone_index in self.path:
            node = node[zone_index]
        return node

    def current_options(self):
        """
        Returns the dict of {zone_index: label} the UI should currently
        display/highlight, based on self.path. Pure read — does not
        mutate state.

        Labels: if a child is a leaf (str), show the key itself
        ("7", "ENTER"). If a child is a branch (dict), show a preview of
        every leaf reachable underneath it (e.g. "0135") instead of a
        flat "..." — this is what makes it visible on screen that you've
        actually drilled deeper, since a static "..." looked identical
        before and after a confirmed blink.
        """
        node = self._current_node()
        if not isinstance(node, dict):
            return {}

        options = {}
        for zone_index, child in node.items():
            if isinstance(child, str):
                options[zone_index] = child
            else:
                options[zone_index] = self._leaf_preview(child)
        return options

    def _leaf_preview(self, node):
        """
        Recursively collect every leaf string reachable under `node`,
        joined together (e.g. "0135"). Used by current_options() to give
        branches a label that actually changes as you drill deeper,
        instead of a static "...".
        """
        if isinstance(node, str):
            return node
        leaves = []
        for child in node.values():
            leaves.append(self._leaf_preview(child))
        return "".join(leaves)

    def confirm_zone(self, zone_index):
        """
        Call this ONLY when a blink is detected (shared_data["blink_detected"]
        flips True), passing whichever zone was most recently classified
        by predict_zone(). This is the one place drill state actually
        advances — current_options()/update_pitch() never mutate path
        on their own.

        Returns a dict describing what happened, e.g.:
          {"event": "drilled_in"}                  -> moved deeper, no key yet
          {"event": "key_typed", "key": "4"}        -> digit/enter appended
          {"event": "backspace"}                    -> last char removed
        Caller (main.py) decides what to actually do with typed_input
        (e.g. print it, forward ENTER as a real action) — this method
        only tracks the string internally for convenience/debugging.
        """
        if self.mode != "SELECT":
            return {"event": "ignored_not_in_select_mode"}

        node = self._current_node()
        if not isinstance(node, dict) or zone_index not in node:
            return {"event": "ignored_unknown_zone"}

        child = node[zone_index]

        if isinstance(child, dict):
            self.path.append(zone_index)
            return {"event": "drilled_in"}

        # child is a leaf (str) -> selection complete
        self._append_key(child)
        self._reset_drill()
        if child == "BACKSPACE":
            return {"event": "backspace"}
        return {"event": "key_typed", "key": child}

    def _append_key(self, key):
        if key == "BACKSPACE":
            self.typed_input = self.typed_input[:-1]
        elif key == "ENTER":
            pass  # caller decides what ENTER actually does downstream
        else:
            self.typed_input += key