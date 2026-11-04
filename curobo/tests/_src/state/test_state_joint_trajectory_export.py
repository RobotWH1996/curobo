# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for full-joint trajectory export."""

from __future__ import annotations

import pytest
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import expand_joint_trajectory
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg


def test_expand_trajectory_holds_unoptimized_joints(cpu_cuda_device_cfg: DeviceCfg) -> None:
    """Preserve batched active motion, derivatives and timing on CPU and CUDA."""
    device = cpu_cuda_device_cfg.device
    position = torch.arange(12, dtype=torch.float32, device=device).reshape(2, 3, 2)
    expected_position = position.clone()
    trajectory = JointState(
        position=position,
        velocity=position + 1.0,
        acceleration=position + 2.0,
        jerk=position + 3.0,
        joint_names=["arm_b", "arm_a"],
        dt=torch.tensor([0.05], device=device),
        knot=position[:, :2].clone(),
        knot_dt=torch.tensor([0.1], device=device),
        control_space=ControlSpace.BSPLINE_3,
        aux_data={"stage": "lift"},
    )
    result = expand_joint_trajectory(
        trajectory,
        {"jaw": -0.03, "torso": 0.4, "arm_a": 123.0},
        ["torso", "arm_a", "jaw", "arm_b"],
    )
    assert result.position.device == position.device
    assert result.position.dtype == position.dtype
    assert result.joint_names == ["torso", "arm_a", "jaw", "arm_b"]
    torch.testing.assert_close(result.position[..., 0], torch.full_like(position[..., 0], 0.4))
    torch.testing.assert_close(result.position[..., 2], torch.full_like(position[..., 0], -0.03))
    for field in ("position", "velocity", "acceleration", "jerk", "knot"):
        torch.testing.assert_close(getattr(result, field)[..., [3, 1]], getattr(trajectory, field))
    for field in ("velocity", "acceleration", "jerk"):
        assert torch.count_nonzero(getattr(result, field)[..., [0, 2]]) == 0
    torch.testing.assert_close(result.knot[..., 2], torch.full_like(result.knot[..., 2], -0.03))
    torch.testing.assert_close(result.dt, trajectory.dt)
    torch.testing.assert_close(result.knot_dt, trajectory.knot_dt)
    assert result.control_space == trajectory.control_space
    assert result.aux_data == trajectory.aux_data
    result.position.zero_()
    torch.testing.assert_close(trajectory.position, expected_position)
    assert trajectory.joint_names == ["arm_b", "arm_a"]


def test_expand_trajectory_optional_derivatives() -> None:
    """Do not invent derivative arrays for a position-only trajectory."""
    trajectory = JointState(torch.tensor([[0.0], [1.0]]), joint_names=["arm"])
    result = expand_joint_trajectory(trajectory, {"jaw": -0.02}, ["arm", "jaw"])
    assert result.velocity is result.acceleration is result.jerk is None
    torch.testing.assert_close(result.position, torch.tensor([[0.0, -0.02], [1.0, -0.02]]))


def test_expand_trajectory_rejects_missing_or_ambiguous_joints() -> None:
    """Fail rather than emitting unknown held positions or dropping optimized joints."""
    trajectory = JointState(torch.tensor([[0.0], [1.0]]), joint_names=["arm"])
    for names, held in (
        (["arm", "jaw"], {}),
        (["arm", "jaw"], {"jaw": float("nan")}),
        (["jaw"], {"jaw": 0.0}),
        (["arm", "arm"], {}),
    ):
        with pytest.raises(ValueError):
            expand_joint_trajectory(trajectory, held, names)
