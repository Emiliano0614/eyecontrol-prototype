# quick_trace.py
from State_machine import CursorStateMachine

sm = CursorStateMachine(pitch_baseline=0.0)  # pitch_baseline irrelevant for this test

def confirm(zone_index):
    options_before = sm.current_options()
    result = sm.confirm_zone(zone_index)
    print(f"  picked {zone_index} (options were {options_before}) -> {result}")
    return result

print("Path 1: 1, 0, 0  (expect: right half -> ??? -> ???)")
confirm(1)
confirm(0)
confirm(0)
print(f"  typed_input so far: {sm.typed_input!r}\n")

sm = CursorStateMachine(pitch_baseline=0.0)  # reset for next trace
print("Path 2: 0, 0, 0  (expect: left half -> ??? -> '0')")
confirm(0)
confirm(0)
confirm(0)
print(f"  typed_input so far: {sm.typed_input!r}\n")

sm = CursorStateMachine(pitch_baseline=0.0)
print("Path 3: 1, 1, 1  (expect: 'ENTER')")
confirm(1)
confirm(1)
confirm(1)
print(f"  typed_input so far: {sm.typed_input!r}")