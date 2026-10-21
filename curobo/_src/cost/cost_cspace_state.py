# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from curobo._src.cost.cost_cspace_base import BaseCSpaceCost
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.cost.wp_cspace_state import StateCSpaceFunction
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg




class StateCSpaceCost(BaseCSpaceCost):
    # One row per intermediate waypoint. The terminal goal stays on the existing
    # single-target cost, so this only stores the waypoints the trajectory must
    # pass through before the goal.
    _MAX_VIAS = 16
    _VIA_WEIGHT = 2.0e5

    def __init__(self, config: CSpaceCostCfg):
        if config.cost_type != CSpaceCostType.STATE:
            log_and_raise("StateCSpaceCost can only be used for state cost")
        super().__init__(config)
        self._target_joint_state = None
        self._idxs_target_joint_state = None
        self._via_position = None
        self._via_idxs = None
        self._via_factor = None
        self._via_enable = None
        self._via_weight = None
        self._via_steps: List[int] = []

    def setup_batch_tensors(self, batch: int, horizon: int):
        if self._batch_size != batch or self._horizon != horizon:
            self._out_c_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            self._out_gp_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._out_gtau_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            self._out_gv_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._out_ga_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._out_gj_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            # create terminal buffers:
            self._run_weight_vel = torch.ones(
                (1, horizon),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._run_weight_acc = torch.ones(
                (1, horizon),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            self._run_weight_jerk = torch.ones(
                (1, horizon),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            self._target_joint_state = JointState.from_position(position=torch.zeros(
                (self.config.dof), device=self.device_cfg.device, dtype=self.device_cfg.dtype
            ))

            self._idxs_target_joint_state = torch.zeros(
                (batch), device=self.device_cfg.device, dtype=torch.int32
            )
            self._joint_torque_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            # Stable storage for the captured kernel. Disabled until a multi-waypoint
            # solve writes into these buffers; single-goal plans leave enable at 0.
            self._via_position = torch.zeros(
                (self._MAX_VIAS, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._via_idxs = torch.zeros(
                (batch, horizon), device=self.device_cfg.device, dtype=torch.int32
            )
            self._via_factor = torch.zeros(
                (horizon,), device=self.device_cfg.device, dtype=self.device_cfg.dtype
            )
            self._via_enable = torch.zeros(
                (1,), device=self.device_cfg.device, dtype=torch.int32
            )
            self._via_weight = torch.zeros(
                (1,), device=self.device_cfg.device, dtype=self.device_cfg.dtype
            )
            self._via_steps: List[int] = []
            super().setup_batch_tensors(batch, horizon)

    def set_via_targets(
        self, positions: torch.Tensor, steps: Optional[List[int]] = None
    ) -> Optional[List[int]]:
        """Pull horizon steps toward intermediate joint targets.

        ``positions`` is ``(n, dof)`` and does not include the terminal goal.
        ``steps`` are the rollout indices chosen from segment travel time.
        When omitted, the indices are spaced by waypoint count.
        Returns the horizon indices that were armed, or None when the buffers
        cannot represent this request.
        """
        if (
            self._horizon is None
            or self._horizon < 3
            or self._via_enable is None
            or positions.ndim != 2
            or positions.shape[-1] != self.config.dof
        ):
            return None
        n = int(positions.shape[0])
        if n <= 0 or n > self._via_position.shape[0]:
            return None
        horizon = int(self._horizon)
        if steps is None:
            n_total = n + 1
            steps = []
            for i in range(n):
                step = int(round((i + 1) / n_total * (horizon - 1)))
                steps.append(min(max(step, 1), horizon - 2))
        if len(steps) != n:
            return None
        resolved: List[int] = []
        used = set()
        for step in steps:
            step = int(step)
            if step < 1 or step > horizon - 2:
                return None
            while step in used and step < horizon - 2:
                step += 1
            if step in used:
                return None
            used.add(step)
            resolved.append(step)
        steps = resolved
        self._via_position[:n].copy_(
            positions.to(device=self._via_position.device, dtype=self._via_position.dtype)
        )
        self._via_idxs.zero_()
        self._via_factor.zero_()
        for row, step in enumerate(steps):
            self._via_idxs[:, step] = row
            self._via_factor[step] = 1.0
        self._via_weight.fill_(self._VIA_WEIGHT)
        self._via_enable.fill_(1)
        self._via_steps = steps
        return steps

    def clear_via_targets(self) -> None:
        if self._via_enable is not None:
            self._via_enable.zero_()
        if self._via_factor is not None:
            self._via_factor.zero_()
        self._via_steps = []

    def forward(
        self,
        state_batch: JointState,
        joint_torque: Optional[torch.Tensor] = None,
        target_joint_state: Optional[JointState] = None,
        idxs_target_joint_state: Optional[torch.Tensor] = None,
        current_joint_state: Optional[JointState] = None,
        idxs_current_joint_state: Optional[torch.Tensor] = None,
        current_state_dt: Optional[torch.Tensor] = None,
    ):
        self.validate_input(state_batch, joint_torque, target_joint_state, idxs_target_joint_state)

        if target_joint_state is None:
            target_joint_state = self._target_joint_state
        if idxs_target_joint_state is None:
            idxs_target_joint_state = self._idxs_target_joint_state
        if target_joint_state is None:
            log_and_raise("target_joint_state is required, call setup_batch_tensors()  first")
        if joint_torque is None:
            joint_torque = self._joint_torque_buffer

        cost = StateCSpaceFunction.apply(
            state_batch.position,
            state_batch.velocity,
            state_batch.acceleration,
            state_batch.jerk,
            joint_torque,
            state_batch.dt,
            target_joint_state.position,
            idxs_target_joint_state,
            self.config.joint_limits.position,
            self.config.joint_limits.velocity,
            self.config.joint_limits.acceleration,
            self.config.joint_limits.jerk,
            self.config.joint_limits.effort,
            self._weight,
            self.config.activation_distance,
            self.config.squared_l2_regularization_weight,
            self._cspace_target_weight,
            self.config.cspace_non_terminal_weight_factor,
            self.config.cspace_target_dof_weight,
            self._out_c_buffer,
            self._out_gp_buffer,
            self._out_gv_buffer,
            self._out_ga_buffer,
            self._out_gj_buffer,
            self._out_gtau_buffer,
            self.config.retime_weights,
            self.config.retime_regularization_weights,
            self.config.use_grad_input,
            self._via_enable,
            self._via_weight,
            self._via_factor,
            self._via_idxs,
            self._via_position,
        )



        return cost
