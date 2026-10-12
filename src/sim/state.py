"""Typed state containers for the leader environment.

Exactness label: implementation convenience only.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ChannelState",
    "LeaderAction",
    "ScenarioState",
    "TaskState",
    "VehicleState",
]


@dataclass(frozen=True)
class TaskState:
    task_id: int
    roi_center_m: tuple[float, float]
    beta: float
    tau_max_s: float
    length_m: float
    width_m: float
    candidate_limit: int
    urgency_label: str


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: int
    position_m: tuple[float, float]
    raw_data_rate_bps: float
    resolution_coeff: float
    cpu_cycles_per_s: float
    sensor_power_w: float
    tx_power_w: float
    energy_weight: float


@dataclass(frozen=True)
class ChannelState:
    vehicle_id: int
    distance_to_rsu_m: float
    fading_power: float
    path_loss_exponent: float


@dataclass(frozen=True)
class LeaderAction:
    prices: tuple[float, ...]
    bandwidth: tuple[float, ...]


@dataclass(frozen=True)
class ScenarioState:
    tasks: tuple[TaskState, ...]
    vehicles: tuple[VehicleState, ...]
    channels: tuple[ChannelState, ...]
    distance_to_roi_m: tuple[tuple[float, ...], ...]
    occlusion_factors: tuple[tuple[float, ...], ...]

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def num_vehicles(self) -> int:
        return len(self.vehicles)
