# drill_tree.py
# Binary drill tree for the 05 numeric keypad (2-zone version).
# Each node is either:
#   - a leaf: a string, the final key ("0".."9", "BACKSPACE", "ENTER")
#   - a branch: a dict {0: <node>, 1: <node>}
# No BACK/cancel path for now — dropped when we went to 2 zones,
# since there's no free 3rd zone to hide it in anymore.

TREE = {
    0: {                                  # left half: 0-5
        0: {0: "0", 1: "1"},
        1: {0: {0: "3", 1: "4"}, 1: "5"},
    },
    1: {                                  # right half: 6-9, BACKSPACE, ENTER
        0: {0: {0: "6", 1: "7"}, 1: "8"},
        1: {0: {0: "9", 1: "BACKSPACE"}, 1: "ENTER"},
    },
}