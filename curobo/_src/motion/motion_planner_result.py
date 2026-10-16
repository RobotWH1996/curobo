# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.state.state_joint import JointState


@dataclass
class MotionPlannerResult:
    """Result of a motion planning operation."""
    success: Optional[torch.Tensor] = None


@dataclass
class WaypointPlanResult:
    """Result of sequential waypoint (via-pose) planning.

    Each Cartesian or joint-space via is planned as a TrajOpt segment. The
    returned ``interpolated_trajectory`` is a fly-by retiming of those
    segments: it passes through each via without coming to rest, except at the
    final goal. If that spline collides, the concatenated rest-to-rest
    segments are used instead. ``segment_trajectories`` keeps the original
    rest-to-rest TrajOpt segments.
    """

    success: Optional[torch.Tensor] = None
    interpolated_trajectory: Optional[JointState] = None
    segment_trajectories: Optional[list] = None
    failed_index: Optional[int] = None
    status: Optional[str] = None
    total_time: float = 0.0
    solve_time: float = 0.0
    planning_time: float = 0.0
    #: TrajOpt/IK result of the failed segment, if any (None means IK returned None).
    failed_result: Optional[object] = None


@dataclass
class GraspPlanResult:
    """Result of a grasp planning operation."""

    success: Optional[torch.Tensor] = None
    approach_success: Optional[torch.Tensor] = None
    grasp_success: Optional[torch.Tensor] = None
    lift_success: Optional[torch.Tensor] = None
    approach_trajectory: Optional[JointState] = None
    approach_trajectory_dt: Optional[torch.Tensor] = None
    approach_interpolated_trajectory: Optional[JointState] = None
    grasp_trajectory: Optional[JointState] = None
    grasp_trajectory_dt: Optional[torch.Tensor] = None
    grasp_interpolated_trajectory: Optional[JointState] = None
    lift_trajectory: Optional[JointState] = None
    lift_trajectory_dt: Optional[torch.Tensor] = None
    lift_interpolated_trajectory: Optional[JointState] = None
    approach_interpolated_last_tstep: Optional[torch.Tensor] = None
    grasp_interpolated_last_tstep: Optional[torch.Tensor] = None
    lift_interpolated_last_tstep: Optional[torch.Tensor] = None
    status: Optional[str] = None
    planning_time: float = 0.0
    goalset_index: Optional[torch.Tensor] = None

