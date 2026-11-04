# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Trajectory operations on JointState objects."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence, Union

import torch

from curobo._src.types.control_space import ControlSpace
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import clone_if_not_none

from .state_joint_jit_helpers import trim_trajectory_jit

if TYPE_CHECKING:
    from .state_joint import JointState


def expand_joint_trajectory(
    trajectory: "JointState",
    held_positions: Mapping[str, float],
    joint_names: Sequence[str],
) -> "JointState":
    """Export a trajectory with additional joints held at supplied positions.

    Args:
        trajectory: Optimized state with tensors shaped (..., horizon, active_dof).
        held_positions: Measured or predicted positions for every added joint.
        joint_names: Unique output order containing all optimized joints.

    Returns:
        New state shaped (..., horizon, len(joint_names)) on the original device
        and dtype. Added joints have zero derivatives; absent derivatives stay
        absent. Inputs and optimization degrees of freedom are unchanged.
    """
    from .state_joint import JointState

    output_names = list(joint_names)
    active_names = trajectory.joint_names
    if active_names is None or trajectory.position.ndim < 2:
        log_and_raise("A trajectory with joint names and a horizon dimension is required")
    if len(active_names) != trajectory.position.shape[-1]:
        log_and_raise("Trajectory joint names do not match its tensor width")
    if len(set(active_names)) != len(active_names):
        log_and_raise("Trajectory joint names must be unique")
    if not output_names or len(set(output_names)) != len(output_names):
        log_and_raise("Output joint names must be nonempty and unique")
    missing_active = set(active_names).difference(output_names)
    if missing_active:
        log_and_raise(f"Output joint names omit optimized joints: {sorted(missing_active)}")

    active_set = set(active_names)
    held_values = []
    for name in output_names:
        if name in active_set:
            held_values.append(0.0)
        else:
            if name not in held_positions or not isfinite(held_positions[name]):
                log_and_raise(f"A finite held position is required for joint {name!r}")
            held_values.append(held_positions[name])
    active_indices = [output_names.index(name) for name in active_names]

    def expand(
        values: Optional[torch.Tensor], hold_position: bool = False
    ) -> Optional[torch.Tensor]:
        if values is None:
            return None
        if values.shape[-1] != len(active_names):
            log_and_raise("Trajectory tensors must have the same joint width")
        shape = (*values.shape[:-1], len(output_names))
        if hold_position:
            expanded = values.new_tensor(held_values).expand(shape).clone()
        else:
            expanded = values.new_zeros(shape)
        indices = torch.as_tensor(active_indices, device=values.device, dtype=torch.long)
        return expanded.index_copy_(-1, indices, values)

    position_knots = (
        trajectory.control_space is None
        or trajectory.control_space in ControlSpace.position_types()
    )
    return JointState(
        position=expand(trajectory.position, hold_position=True),
        velocity=expand(trajectory.velocity),
        acceleration=expand(trajectory.acceleration),
        jerk=expand(trajectory.jerk),
        joint_names=output_names,
        device_cfg=trajectory.device_cfg,
        dt=clone_if_not_none(trajectory.dt),
        aux_data=trajectory.aux_data.copy(),
        knot=expand(trajectory.knot, hold_position=position_knots),
        knot_dt=clone_if_not_none(trajectory.knot_dt),
        control_space=trajectory.control_space,
    )


def gather_joint_state_by_seed(joint_state: "JointState", idx: torch.Tensor) -> "JointState":
    """Gather joint states by seed index.

    Joint state is of shape (batch_size, num_seeds, horizon, dof).
    idx is a tensor of shape (batch_size, topk).
    Output is (batch_size, topk, horizon, dof).

    Args:
        joint_state: Joint state with shape (batch_size, num_seeds, horizon, dof).
        idx: Index tensor of shape (batch_size, topk).

    Returns:
        Gathered joint state.
    """
    from .state_joint import JointState

    dim = 1
    if idx.ndim != 2:
        log_and_raise("idx is not a 2D tensor")
    if idx.shape[0] != joint_state.position.shape[0]:
        log_and_raise("idx batch size does not match position batch size")
    if joint_state.position.ndim != 4:
        log_and_raise("position is not a 4D tensor (batch_size, num_seeds, horizon, dof)")

    # create a flat index tensor:
    batch_size = joint_state.shape[0]
    num_seeds = joint_state.shape[1]
    horizon = joint_state.shape[2]
    dof = joint_state.shape[3]
    topk_seeds = idx.shape[1]
    offset = torch.arange(batch_size, device=idx.device) * num_seeds
    offset = offset.view(-1, 1)
    flat_indices = (idx + offset).view(-1)

    # gather position:
    position = joint_state.position.view(batch_size * num_seeds, horizon, dof)[flat_indices].view(
        batch_size, topk_seeds, horizon, dof
    )
    # gather velocity:
    velocity = joint_state.velocity.view(batch_size * num_seeds, horizon, dof)[flat_indices].view(
        batch_size, topk_seeds, horizon, dof
    )
    # gather acceleration:
    acceleration = joint_state.acceleration.view(batch_size * num_seeds, horizon, dof)[flat_indices].view(
        batch_size, topk_seeds, horizon, dof
    )
    # gather jerk:
    jerk = joint_state.jerk.view(batch_size * num_seeds, horizon, dof)[flat_indices].view(
        batch_size, topk_seeds, horizon, dof
    )
    # gather dt:
    dt = joint_state.dt.view(batch_size * num_seeds, -1)[flat_indices].view(batch_size, topk_seeds)

    if joint_state.knot is not None:
        knots = joint_state.knot.shape[-2]
        knot = joint_state.knot.view(batch_size * num_seeds, knots, dof)[flat_indices].view(
            batch_size, topk_seeds, knots, dof
        )
    else:
        knot = None

    if joint_state.knot_dt is not None:
        knot_dt = joint_state.knot_dt.view(batch_size * num_seeds, -1)[flat_indices].view(
            batch_size, topk_seeds
        )
    else:
        knot_dt = None
    return JointState(
        position,
        velocity,
        acceleration,
        jerk=jerk,
        joint_names=joint_state.joint_names,
        dt=dt,
        knot=knot,
        knot_dt=knot_dt,
        control_space=joint_state.control_space,
    )


def copy_joint_state_only_index(
    target: "JointState", source: "JointState", idx: Union[int, torch.Tensor]
) -> "JointState":
    """Copy joint state data at specific index from source to target.

    Args:
        target: Target joint state (modified in-place).
        source: Source joint state.
        idx: Index to copy.

    Returns:
        Modified target state.
    """
    if target.position is not None:
        target.position[idx] = source.position[idx]
    if target.velocity is not None:
        target.velocity[idx] = source.velocity[idx]
    if target.acceleration is not None:
        target.acceleration[idx] = source.acceleration[idx]
    if target.jerk is not None:
        target.jerk[idx] = source.jerk[idx]
    if target.dt is not None:
        target.dt[idx] = source.dt[idx]
    return target


def copy_joint_state_at_index(
    target: "JointState", source: "JointState", idx: Union[int, List, torch.Tensor]
) -> None:
    """Copy source joint state to target at specific index.

    Args:
        target: Target joint state (modified in-place).
        source: Source joint state.
        idx: Index to copy to.
    """
    max_idx = 0
    if isinstance(idx, int):
        max_idx = idx
    elif isinstance(idx, List):
        max_idx = max(idx)
    elif isinstance(idx, torch.Tensor):
        max_idx = torch.max(idx)
    if target.position is not None:
        if max_idx >= target.position.shape[0]:
            raise ValueError(
                str(max_idx)
                + " index out of range, current state is of length "
                + str(target.position.shape[0])
            )
        target.position[idx] = source.position
    if target.velocity is not None:
        target.velocity[idx] = source.velocity
    if target.acceleration is not None:
        target.acceleration[idx] = source.acceleration
    if target.jerk is not None:
        target.jerk[idx] = source.jerk
    if target.dt is not None and source.dt is not None:
        target.dt[idx] = source.dt


def copy_joint_state_at_batch_seed_indices(
    target: "JointState",
    source: "JointState",
    batch_idx: torch.Tensor,
    seed_idx: torch.Tensor,
) -> "JointState":
    """Copy joint state at specific batch and seed indices.

    Args:
        target: Target joint state (modified in-place).
        source: Source joint state.
        batch_idx: Batch indices.
        seed_idx: Seed indices.

    Returns:
        Modified target state.
    """
    if target.position is not None:
        target.position[batch_idx, seed_idx] = source.position[batch_idx, seed_idx]
    if target.velocity is not None:
        target.velocity[batch_idx, seed_idx] = source.velocity[batch_idx, seed_idx]
    if target.acceleration is not None:
        target.acceleration[batch_idx, seed_idx] = source.acceleration[batch_idx, seed_idx]
    if target.jerk is not None:
        target.jerk[batch_idx, seed_idx] = source.jerk[batch_idx, seed_idx]
    if target.dt is not None and source.dt is not None:
        target.dt[batch_idx, seed_idx] = source.dt[batch_idx, seed_idx]
    if target.knot_dt is not None and source.knot_dt is not None:
        target.knot_dt[batch_idx, seed_idx] = source.knot_dt[batch_idx, seed_idx]
    if target.knot is not None and source.knot is not None:
        target.knot[batch_idx, seed_idx] = source.knot[batch_idx, seed_idx]
    return target


def get_joint_state_at_horizon_index(joint_state: "JointState", horizon_index: int) -> "JointState":
    """Get joint state at a specific horizon index.

    Args:
        joint_state: Joint state with horizon dimension.
        horizon_index: Index along the horizon dimension.

    Returns:
        Joint state at the specified horizon index.
    """
    from .state_joint import JointState

    if len(joint_state.position.shape) < 2:
        raise ValueError("JointState does not have horizon")
    return JointState(
        position=joint_state.position[..., horizon_index, :],
        velocity=joint_state.velocity[..., horizon_index, :] if joint_state.velocity is not None else None,
        acceleration=(
            joint_state.acceleration[..., horizon_index, :] if joint_state.acceleration is not None else None
        ),
        jerk=joint_state.jerk[..., horizon_index, :] if joint_state.jerk is not None else None,
        joint_names=joint_state.joint_names,
        device_cfg=joint_state.device_cfg,
        dt=joint_state.dt,
        control_space=joint_state.control_space,
        knot=joint_state.knot[..., horizon_index, :] if joint_state.knot is not None else None,
        knot_dt=joint_state.knot_dt,
    )


def trim_joint_state_trajectory(
    joint_state: "JointState", start_idx: int, end_idx: Optional[int] = None
) -> "JointState":
    """Trim joint state trajectory to specified range.

    Args:
        joint_state: Joint state with trajectory.
        start_idx: Start index.
        end_idx: End index (defaults to end of trajectory).

    Returns:
        Trimmed joint state.
    """
    from .state_joint import JointState

    if end_idx is None or end_idx == 0:
        end_idx = joint_state.position.shape[-2]
    if len(joint_state.position.shape) < 2:
        raise ValueError("JointState does not have horizon")
    pos, vel, acc, jerk = trim_trajectory_jit(
        joint_state.position, joint_state.velocity, joint_state.acceleration, joint_state.jerk, start_idx, end_idx
    )

    return JointState(
        pos,
        vel,
        acc,
        joint_names=joint_state.joint_names,
        jerk=jerk,
        device_cfg=joint_state.device_cfg,
        dt=joint_state.dt,
        control_space=joint_state.control_space,
    )


def index_joint_state_dof(joint_state: "JointState", idx: torch.Tensor) -> "JointState":
    """Index joint state along the DOF dimension.

    Args:
        joint_state: Joint state.
        idx: Index tensor for DOF selection.

    Returns:
        Indexed joint state.
    """
    from .state_joint import JointState

    velocity = acceleration = jerk = None
    new_index = idx
    knot = None
    position = torch.index_select(joint_state.position, -1, new_index)
    if joint_state.velocity is not None:
        velocity = torch.index_select(joint_state.velocity, -1, new_index)
    if joint_state.acceleration is not None:
        acceleration = torch.index_select(joint_state.acceleration, -1, new_index)
    if joint_state.jerk is not None:
        jerk = torch.index_select(joint_state.jerk, -1, new_index)
    if joint_state.knot is not None:
        knot = torch.index_select(joint_state.knot, -1, new_index)
    joint_names = [joint_state.joint_names[x] for x in new_index]
    return JointState(
        position=position,
        joint_names=joint_names,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        control_space=joint_state.control_space,
        knot=knot,
        knot_dt=joint_state.knot_dt,
    )

