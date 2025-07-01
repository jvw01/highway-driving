import enum
from dataclasses import dataclass
from typing import Optional

class StateMachine(enum.Enum):
    INITIALIZATION = 1
    ATTEMPT_LANE_CHANGE = 2
    EXECUTE_LANE_CHANGE = 3
    HOLD_LANE = 4
    GOAL_STATE = 5

@dataclass
class StateContext:
    """Context object to hold state-specific data"""
    shortest_path: list
    path_node: int
    num_steps_path: int
    freq_counter: int
    # ... other state-specific variables