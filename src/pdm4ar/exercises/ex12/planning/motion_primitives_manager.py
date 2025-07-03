from dataclasses import dataclass
from typing import List, Tuple
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from pdm4ar.exercises.ex12.planning.motion_primitves import generate_primat
import numpy as np

@dataclass

class MotionPrimitiveManager:
    def __init__(self, vp, vg, dt_integral, scaling_dt):
        self.vp = vp
        self.vg = vg
        self.dt_integral = dt_integral
        self.scaling_dt = scaling_dt

    def generate_control_inputs(
        self,
        current_state: VehicleState, 
        goal_lane: str,
        lane_orientation: float,
        delta_steer: float,
        steps_lane_change: int,
        n_steps: int,
    ) -> Tuple[List[List[VehicleState]], List[List[VehicleCommands]]]:
        """Generate motion primitives for lane change maneuver."""

        end_states = []
        control_inputs = []
        state = current_state

        for i in range(steps_lane_change):
            case = 0 if i == 0 else 2

            end_state, control_input = generate_primat(
                x0=state,
                steer_range=delta_steer,
                goal_lane=goal_lane,
                case=case,
                vp=self.vp,
                vg=self.vg,
                dt=self.dt_integral,
                n_steps=n_steps,
                scaling_dt=self.scaling_dt,
            )

            end_states.append(end_state)
            control_inputs.append(control_input)

            if i == steps_lane_change - 1:
                break

            # Calculate next starting state
            state = self._calculate_next_state(current_state, end_state[-1], lane_orientation)

        return end_states, control_inputs

    def _calculate_next_state(self, current_state: VehicleState, end_state: VehicleState, lane_orientation: float) -> VehicleState:
        """Calculate the next starting state for primitive generation."""

        dx = end_state.x - current_state.x
        dy = end_state.y - current_state.y
        dpsi = end_state.psi - current_state.psi
        delta = end_state.delta - current_state.delta

        delta_pos = np.array(
            [
                dx * np.cos(current_state.psi - lane_orientation)
                - dy * np.sin(current_state.psi - lane_orientation),
                dy * np.cos(current_state.psi - lane_orientation)
                + dx * np.sin(current_state.psi - lane_orientation),
            ]
        )

        state = VehicleState(
            x=current_state.x + delta_pos[0],
            y=current_state.y + delta_pos[1],
            psi=current_state.psi + dpsi,
            vx=current_state.vx,
            delta=current_state.delta + delta,
        )

        return state
