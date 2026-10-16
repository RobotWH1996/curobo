# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-problem motion planner with retry logic and graph-planner seeding."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.collision.collision_scene import create_scene_collision
from curobo._src.geom.types import SceneCfg
from curobo._src.graph_planner.graph_planner_prm import (
    PRMGraphPlanner,
    TrajInterpolationType,
)
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.motion.motion_planner_result import GraspPlanResult, WaypointPlanResult
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import cat_joint_states
from curobo._src.state.state_joint_trajectory_ops import (
    get_joint_state_at_horizon_index,
    trim_joint_state_trajectory,
)
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise, log_warn


def _peak(t: torch.Tensor) -> float:
    if t is None or t.numel() == 0:
        return 0.0
    return float(t.detach().amax().item())


_CONSTRAINT_ZH = {
    "self_collision": "自碰",
    "scene_collision": "场景碰撞",
    "cspace": "关节限位",
}


def _constraint_violations(metrics, index: Optional[int] = None) -> list:
    if metrics is None or getattr(metrics, "costs_and_constraints", None) is None:
        return []
    cc = metrics.costs_and_constraints
    out = []
    for coll in (cc.constraints, cc.hybrid_costs_constraints):
        if coll is None:
            continue
        for name, val in zip(coll.names, coll.values):
            if val is None or val.numel() == 0:
                continue
            peak = _peak(val)
            if index is not None:
                if index < 0 or index >= val.shape[0]:
                    continue
                peak = _peak(val.reshape(val.shape[0], -1)[index])
            if peak > 1e-6:
                label = _CONSTRAINT_ZH.get(name, name)
                out.append(f"{label}={peak:.4g}")
    return out


def _pose_tol(result, pos_fallback: float = 0.005, rot_fallback: float = 0.05):
    pe_tol = float(getattr(result, "position_tolerance", 0.0) or 0.0)
    re_tol = float(getattr(result, "orientation_tolerance", 0.0) or 0.0)
    if pe_tol <= 0.0:
        pe_tol = pos_fallback
    if re_tol <= 0.0:
        re_tol = rot_fallback
    return pe_tol, re_tol


def format_ik_failure(ik_result) -> str:
    """Classify IK failure as unreachable vs collision."""
    if ik_result is None:
        return (
            "IK 失败：没有找到同时满足所有 tool_frames 目标位姿的无碰关节构型"
            "（无法读取 IK 诊断）"
        )

    pe = getattr(ik_result, "position_error", None)
    re = getattr(ik_result, "rotation_error", None)
    feas = getattr(ik_result, "feasible", None)
    if pe is None or re is None or pe.numel() == 0:
        return "IK 失败：没有有效的位姿误差（求解器未返回诊断）"

    pe_tol, re_tol = _pose_tol(ik_result)
    pe_flat = pe.reshape(-1)
    re_flat = re.reshape(-1)
    pose_ok = (pe_flat <= pe_tol) & (re_flat <= re_tol)
    feas_flat = (
        feas.reshape(-1).to(dtype=torch.bool)
        if feas is not None and feas.numel() == pe_flat.numel()
        else None
    )
    pose_score = pe_flat / max(pe_tol, 1e-9) + re_flat / max(re_tol, 1e-9)
    best = int(torch.argmin(pose_score).item())
    pe_best = float(pe_flat[best].item())
    re_best = float(re_flat[best].item())
    pose_best_ok = pe_best <= pe_tol and re_best <= re_tol

    any_pose = bool(pose_ok.any().item())
    any_feas = bool(feas_flat.any().item()) if feas_flat is not None else False
    both = (
        bool((pose_ok & feas_flat).any().item())
        if feas_flat is not None
        else False
    )

    seed_idx = None
    debug = getattr(ik_result, "debug_info", None) or {}
    raw_idx = debug.get("seed_idx") if isinstance(debug, dict) else None
    if raw_idx is not None:
        try:
            seed_idx = int(raw_idx.reshape(-1)[best].item())
        except Exception:
            seed_idx = None
    viol = _constraint_violations(getattr(ik_result, "metrics", None), seed_idx)
    has_self = any(v.startswith("自碰") for v in viol)
    has_scene = any(v.startswith("场景碰撞") for v in viol)
    has_limit = any(v.startswith("关节限位") for v in viol)
    colliding = has_self or has_scene or has_limit

    if both and not colliding:
        kind = "有种子同时到位且无碰，但未被后续 TrajOpt 采用"
    elif pose_best_ok and colliding:
        reasons = []
        if has_self:
            reasons.append("自碰")
        if has_scene:
            reasons.append("场景碰撞")
        if has_limit:
            reasons.append("关节限位")
        kind = "碰撞：末端误差已达标，失败原因是" + "、".join(reasons)
    elif any_pose and not any_feas:
        kind = "碰撞：末端误差已达标，但所有 IK 构型自碰、碰障碍或超关节限位"
    elif colliding and not any_pose:
        extra = []
        if has_self:
            extra.append("自碰")
        if has_scene:
            extra.append("场景碰撞")
        kind = "不可达且碰撞：末端达不到目标，最好种子还有" + "、".join(extra or ["约束违反"])
    elif any_feas and not any_pose:
        kind = "不可达：存在无碰构型，但末端位置/姿态达不到目标"
    else:
        kind = "不可达且碰撞：没有种子既能到位又无碰"

    parts = [
        f"IK 失败：{kind}",
        (
            f"最好种子 位置误差 {pe_best * 1000:.1f} mm（阈值 {pe_tol * 1000:.1f} mm）, "
            f"姿态误差 {re_best * 180.0 / 3.14159265:.2f} deg"
            f"（阈值 {re_tol * 180.0 / 3.14159265:.2f} deg）"
        ),
    ]
    if feas_flat is not None:
        parts.append("该种子无碰" if bool(feas_flat[best].item()) else "该种子不可行")
    if viol:
        parts.append("约束: " + ", ".join(viol))
    return "; ".join(parts)


def format_plan_failure(
    result: Optional[TrajOptSolverResult] = None,
    ik_result=None,
) -> str:
    """Human-readable reason why ``plan_pose`` / TrajOpt did not succeed."""
    if result is None:
        return format_ik_failure(ik_result)
    success = getattr(result, "success", None)
    if success is not None and bool(success.any().item()):
        return "规划成功"

    parts = []
    feasible = getattr(result, "feasible", None)
    feas_any = bool(feasible.any().item()) if feasible is not None else None
    if feas_any is False:
        parts.append("碰撞：轨迹自碰、碰障碍或超关节限位")
    elif feas_any is True:
        parts.append("不可达/未收敛：轨迹无碰，但末端位姿未到阈值")
    else:
        parts.append("TrajOpt 未成功")

    pe = getattr(result, "position_error", None)
    re = getattr(result, "rotation_error", None)
    pe_tol = float(getattr(result, "position_tolerance", 0.0) or 0.0)
    re_tol = float(getattr(result, "orientation_tolerance", 0.0) or 0.0)
    if pe is not None and pe.numel() > 0:
        pe_best = float(pe.amin().item())
        parts.append(f"位置误差 {pe_best * 1000:.1f} mm（阈值 {pe_tol * 1000:.1f} mm）")
    if re is not None and re.numel() > 0:
        re_best = float(re.amin().item())
        parts.append(
            f"姿态误差 {re_best * 180.0 / 3.14159265:.2f} deg（阈值 {re_tol * 180.0 / 3.14159265:.2f} deg）"
        )

    viol = _constraint_violations(getattr(result, "metrics", None))
    if viol:
        parts.append("约束: " + ", ".join(viol))
    ival = _constraint_violations(getattr(result, "interpolated_metrics", None))
    if ival:
        parts.append("插值轨迹约束: " + ", ".join(ival))
    return "; ".join(parts)


def _axis_string_to_vector(axis: str) -> List[float]:
    if axis == "x":
        return [1.0, 0.0, 0.0]
    elif axis == "y":
        return [0.0, 1.0, 0.0]
    elif axis == "z":
        return [0.0, 0.0, 1.0]
    else:
        log_and_raise(f"Invalid axis: {axis}, must be 'x', 'y', or 'z'")


def _squeeze_joint_state_to_2d(joint_state: JointState) -> JointState:
    """Drop leading size-1 dims so ``plan_pose`` gets ``[1, dof]``."""

    def _squeeze(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        while tensor.ndim > 2:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    return JointState(
        position=_squeeze(joint_state.position),
        velocity=_squeeze(joint_state.velocity),
        acceleration=_squeeze(joint_state.acceleration),
        jerk=_squeeze(joint_state.jerk),
        joint_names=joint_state.joint_names,
        device_cfg=joint_state.device_cfg,
        dt=joint_state.dt,
        control_space=joint_state.control_space,
    )


def _concat_interpolated_segments(segments: List[JointState]) -> JointState:
    """Concatenate interpolated plans, dropping the duplicated junction sample."""
    full = segments[0]
    for segment in segments[1:]:
        if segment.position.shape[-2] <= 1:
            continue
        full = cat_joint_states(
            full, trim_joint_state_trajectory(segment, 1, None), dim=-2
        )
    return full


def _traj_position_2d(traj: JointState) -> torch.Tensor:
    """Return joint positions as ``[T, dof]``."""
    pos = traj.position
    while pos.ndim > 2:
        pos = pos.squeeze(0)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    return pos


def _unsqueeze_like_horizon(traj_2d: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Add leading dims so ``traj_2d`` ``[T, dof]`` matches ``ref`` rank."""
    out = traj_2d
    while out.ndim < ref.ndim:
        out = out.unsqueeze(0)
    return out


def _flyby_spline_through_segments(
    segments: List[JointState],
    interpolation_dt: float,
    max_velocity: torch.Tensor,
    max_acceleration: torch.Tensor,
    min_knot_delta: float = 0.04,
) -> JointState:
    """Retiming spline that passes through via configs without resting.

    TrajOpt segments are still rest-to-rest. This keeps a sparse subset of
    those collision-aware samples (including each via) and fits a clamped
    cubic spline so only the global start/goal have zero velocity.
    """
    from scipy.interpolate import CubicSpline
    import numpy as np

    knots: List[torch.Tensor] = []
    for i, segment in enumerate(segments):
        pos = _traj_position_2d(segment)
        n_steps = int(pos.shape[0])
        if n_steps < 1:
            continue
        start_t = 0 if i == 0 else 1
        for t_idx in range(start_t, n_steps):
            q = pos[t_idx]
            is_via = t_idx == n_steps - 1
            if not knots:
                knots.append(q)
                continue
            delta = torch.max(torch.abs(q - knots[-1])).item()
            if is_via or delta >= min_knot_delta:
                if is_via and delta < 1e-8:
                    continue
                knots.append(q)

    if len(knots) < 3:
        return _concat_interpolated_segments(segments)

    q_knots = torch.stack(knots, dim=0)
    n_knots, dof = q_knots.shape
    q_np = q_knots.detach().cpu().numpy()

    vmax = max_velocity.detach().reshape(-1).cpu().numpy()
    amax = max_acceleration.detach().reshape(-1).cpu().numpy()
    n_lim = min(dof, int(vmax.shape[0]), int(amax.shape[0]))
    vmax = vmax[:n_lim]
    amax = amax[:n_lim]

    t_knots = np.zeros(n_knots, dtype=np.float64)
    for i in range(1, n_knots):
        dq = np.abs(q_np[i, :n_lim] - q_np[i - 1, :n_lim])
        dt_v = float(np.max(dq / np.maximum(vmax, 1e-6)))
        dt_a = float(np.max(np.sqrt(np.maximum(2.0 * dq, 0.0) / np.maximum(amax, 1e-6))))
        t_knots[i] = t_knots[i - 1] + max(dt_v, dt_a, interpolation_dt)

    spline = CubicSpline(t_knots, q_np, axis=0, bc_type="clamped")
    duration = float(t_knots[-1])
    n_eval = max(int(np.ceil(duration / interpolation_dt)) + 1, 2)
    tau = np.linspace(0.0, duration, n_eval)
    q_eval = spline(tau)
    v_eval = spline(tau, 1)
    a_eval = spline(tau, 2)

    s_v = float(np.max(np.abs(v_eval[:, :n_lim]) / np.maximum(vmax, 1e-6)))
    s_a = float(np.max(np.sqrt(np.abs(a_eval[:, :n_lim]) / np.maximum(amax, 1e-6))))
    time_scale = max(1.0, s_v, s_a)
    if time_scale > 1.0:
        duration *= time_scale
        n_eval = max(int(np.ceil(duration / interpolation_dt)) + 1, 2)
        t_new = np.linspace(0.0, duration, n_eval)
        tau = t_new / time_scale
        q_eval = spline(tau)
        v_eval = spline(tau, 1) / time_scale
        a_eval = spline(tau, 2) / (time_scale * time_scale)
        j_eval = spline(tau, 3) / (time_scale ** 3)
    else:
        j_eval = spline(tau, 3)

    ref = segments[0]
    device = ref.position.device
    dtype = ref.position.dtype
    position = _unsqueeze_like_horizon(
        torch.from_numpy(np.ascontiguousarray(q_eval)).to(device=device, dtype=dtype),
        ref.position,
    )
    velocity = _unsqueeze_like_horizon(
        torch.from_numpy(np.ascontiguousarray(v_eval)).to(device=device, dtype=dtype),
        ref.position,
    )
    acceleration = _unsqueeze_like_horizon(
        torch.from_numpy(np.ascontiguousarray(a_eval)).to(device=device, dtype=dtype),
        ref.position,
    )
    jerk = _unsqueeze_like_horizon(
        torch.from_numpy(np.ascontiguousarray(j_eval)).to(device=device, dtype=dtype),
        ref.position,
    )
    if ref.dt is not None:
        dt = torch.full_like(ref.dt, interpolation_dt)
    else:
        dt = torch.tensor([interpolation_dt], device=device, dtype=dtype)

    return JointState(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        joint_names=ref.joint_names,
        device_cfg=ref.device_cfg,
        dt=dt,
        control_space=ref.control_space,
    )


class MotionPlanner:
    """Single-problem motion planner with retry logic and graph-planner seeding.

    Solves one planning problem at a time (``batch_size=1``).  For batched
    planning see :class:`BatchMotionPlanner`.
    """

    def __init__(self, config: MotionPlannerCfg):
        self.config = config
        self.device_cfg = config.device_cfg
        self.scene_collision_checker = None
        self._destroyed = False
        self.last_ik_result = None
        self._initialize_components()

    def _initialize_components(self):
        if self.config.scene_collision_cfg is not None:
            self.scene_collision_checker = create_scene_collision(
                self.config.scene_collision_cfg
            )

        self.ik_solver = IKSolver(self.config.ik_solver_config, self.scene_collision_checker)
        self.trajopt_solver = TrajOptSolver(
            self.config.trajopt_solver_config, self.scene_collision_checker
        )
        self.graph_planner = None
        if self.config.graph_planner_config is not None:
            self.graph_planner = PRMGraphPlanner(
                self.config.graph_planner_config, self.scene_collision_checker
            )

    # -- Lifecycle --

    def destroy(self):
        """Release all CUDA graph resources.

        Call before dropping the last reference to avoid ``cudaMallocAsync``
        warnings about captured allocations freed outside graph replay.

        Does not synchronize, the ``gc.collect()`` + ``synchronize()`` fence
        in :meth:`GraphExecutor._initialize_cuda_graph` ensures stale
        ``cuGraphExecDestroy`` calls complete before any new capture begins.
        """
        if self._destroyed:
            return
        self._destroyed = True
        self.ik_solver.destroy()
        self.trajopt_solver.destroy()
        if self.graph_planner is not None and hasattr(self.graph_planner, "reset_cuda_graph"):
            self.graph_planner.reset_cuda_graph()

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()
        return False

    @property
    def attachment_manager(self) -> AttachmentManager:
        """Attachment manager for attaching/detaching obstacles to robot links."""
        return self.trajopt_solver.core.attachment_manager

    # -- Properties --

    @property
    def joint_names(self) -> List[str]:
        return self.ik_solver.joint_names

    @property
    def action_dim(self) -> int:
        return self.trajopt_solver.action_dim

    @property
    def tool_frames(self) -> List[str]:
        return self.trajopt_solver.tool_frames

    @property
    def default_joint_state(self) -> JointState:
        return self.trajopt_solver.default_joint_state

    @property
    def kinematics(self) -> Kinematics:
        return self.trajopt_solver.kinematics

    def compute_kinematics(self, state: JointState) -> KinematicsState:
        return self.trajopt_solver.compute_kinematics(state)

    # -- Warmup --

    def warmup(
        self,
        enable_graph: bool = True,
        warmup_joint_index: int = 0,
        warmup_joint_delta: float = 0.2,
        num_warmup_iterations: int = 10,
    ) -> bool:
        num_goalset = self.config.ik_solver_config.max_goalset
        og_exit_early = self.ik_solver.config.exit_early
        self.ik_solver.config.exit_early = False

        for _ in range(num_warmup_iterations):
            current_state = self.default_joint_state.clone().unsqueeze(0)

            if num_goalset > 1:
                goalset_poses = self._make_warmup_goalset(
                    current_state, num_goalset, warmup_joint_index, warmup_joint_delta,
                )
                goalset_goal = GoalToolPose.from_poses(
                    goalset_poses, num_goalset=num_goalset,
                )
                result = self.plan_pose(goalset_goal, current_state, max_attempts=1)
                if result is None:
                    goal_tool_poses = GoalToolPose.from_poses(
                        goalset_poses, num_goalset=num_goalset,
                    )
                    seed_config = current_state.position.view(1, 1, -1).repeat(
                        1, self.trajopt_solver.config.num_seeds, 1
                    )
                    self.trajopt_solver.solve_pose(
                        goal_tool_poses, current_state,
                        seed_config=seed_config, use_implicit_goal=True,
                    )
            else:
                goal_state = current_state.clone()
                goal_state.position[..., warmup_joint_index] += warmup_joint_delta
                goal_tool_poses = self.compute_kinematics(goal_state).tool_poses.as_goal()
                self.plan_pose(goal_tool_poses, current_state, max_attempts=1)

            self.reset_seed()

        if enable_graph and self.graph_planner is not None:
            self.graph_planner.warmup(num_warmup_iterations=num_warmup_iterations)

        self.ik_solver.config.exit_early = og_exit_early
        return True

    def _make_warmup_goalset(
        self, current_state: JointState, num_goalset: int,
        joint_index: int, joint_delta: float,
    ) -> Dict[str, Pose]:
        positions_per_frame: Dict[str, list] = {}
        quaternions_per_frame: Dict[str, list] = {}
        for g in range(num_goalset):
            goal_state = current_state.clone()
            delta = joint_delta * (g + 1) / num_goalset
            goal_state.position[..., joint_index] += delta
            kin = self.compute_kinematics(goal_state)
            for frame, pose in kin.tool_poses.to_dict().items():
                positions_per_frame.setdefault(frame, []).append(pose.position)
                quaternions_per_frame.setdefault(frame, []).append(pose.quaternion)

        return {
            frame: Pose(
                position=torch.cat(positions_per_frame[frame], dim=0),
                quaternion=torch.cat(quaternions_per_frame[frame], dim=0),
            )
            for frame in positions_per_frame
        }

    # -- Planning --

    def plan_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
    ) -> Optional[TrajOptSolverResult]:
        """Plan a trajectory to reach target tool poses.

        Goalset is auto-detected from ``goal_tool_poses.num_goalset``.  When
        ``num_goalset > 1`` the planner uses a simpler IK+TrajOpt loop without
        graph seeding.
        """
        if current_state.ndim > 2:
            log_and_raise(f"current_state must be a 2D tensor, got shape: {current_state.shape}")

        if goal_tool_poses.num_goalset > 1:
            return self._plan_pose_goalset(
                goal_tool_poses, current_state, use_implicit_goal, max_attempts,
            )

        return self._plan_pose_single(
            goal_tool_poses, current_state, max_attempts, enable_graph_attempt,
        )

    def plan_cartesian_waypoints(
        self,
        goal_tool_poses: List[GoalToolPose],
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
    ) -> WaypointPlanResult:
        """Plan a collision-free path that flies through Cartesian via poses.

        Each via is still planned with ``plan_pose`` so the geometric path
        stays collision-aware. Those rest-to-rest segments are then retimed
        with a single clamped cubic spline: the trajectory passes through
        each via joint configuration with nonzero speed, and only the global
        start and final goal come to rest. The spline is collision-checked;
        if it hits the scene or self-collides, the concatenated rest-to-rest
        segments are returned instead.

        Args:
            goal_tool_poses: Ordered via poses, last entry is the final goal.
            current_state: Start joint state.
            use_implicit_goal: Forwarded to each ``plan_pose`` call.
            max_attempts: Retries per segment.
            enable_graph_attempt: Graph-seed attempt index forwarded to
                ``plan_pose``.

        Returns:
            :class:`WaypointPlanResult`. ``success`` is false if any segment
            fails; ``failed_index`` is that segment's index.
        """
        if not goal_tool_poses:
            log_and_raise("plan_cartesian_waypoints requires at least one goal pose")

        q = current_state.clone()
        if q.ndim > 2:
            log_and_raise(f"current_state must be a 2D tensor, got shape: {q.shape}")

        segments: List[JointState] = []
        total_time = 0.0
        solve_time = 0.0
        device = q.position.device

        for i, goal in enumerate(goal_tool_poses):
            result = self.plan_pose(
                goal,
                q,
                use_implicit_goal=use_implicit_goal,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
            )
            if result is not None:
                total_time += float(result.total_time)
                solve_time += float(result.solve_time)

            if result is None or not result.success.any():
                return WaypointPlanResult(
                    success=torch.zeros(1, dtype=torch.bool, device=device),
                    interpolated_trajectory=(
                        _concat_interpolated_segments(segments) if segments else None
                    ),
                    segment_trajectories=segments or None,
                    failed_index=i,
                    status=f"segment {i} failed: {format_plan_failure(result, self.last_ik_result)}",
                    failed_result=result,
                    total_time=total_time,
                    solve_time=solve_time,
                    planning_time=total_time,
                )

            traj = result.get_interpolated_plan()
            segments.append(traj)
            end_state = get_joint_state_at_horizon_index(
                traj, traj.position.shape[-2] - 1
            )
            q = self.kinematics.get_active_js(_squeeze_joint_state_to_2d(end_state))

        interpolated, status = self._select_waypoint_interpolated_trajectory(segments)

        return WaypointPlanResult(
            success=torch.ones(1, dtype=torch.bool, device=device),
            interpolated_trajectory=interpolated,
            segment_trajectories=segments,
            failed_index=None,
            status=status,
            total_time=total_time,
            solve_time=solve_time,
            planning_time=total_time,
        )

    def plan_joint_waypoints(
        self,
        goal_states: List[JointState],
        current_state: JointState,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
    ) -> WaypointPlanResult:
        """Plan a collision-free path that flies through joint-space via configs.

        Each via is planned with ``plan_cspace``. Those rest-to-rest segments
        are then retimed with a clamped cubic spline so the trajectory passes
        through each via with nonzero speed; only the global start and final
        goal come to rest. The spline is collision-checked; if it collides,
        the concatenated rest-to-rest segments are returned instead.

        Args:
            goal_states: Ordered via joint states, last entry is the final goal.
            current_state: Start joint state.
            max_attempts: Retries per segment.
            enable_graph_attempt: Graph-seed attempt index forwarded to
                ``plan_cspace``.

        Returns:
            :class:`WaypointPlanResult`. ``success`` is false if any segment
            fails; ``failed_index`` is that segment's index.
        """
        if not goal_states:
            log_and_raise("plan_joint_waypoints requires at least one goal state")

        q = current_state.clone()
        if q.ndim > 2:
            log_and_raise(f"current_state must be a 2D tensor, got shape: {q.shape}")
        q = self.kinematics.get_active_js(_squeeze_joint_state_to_2d(q))

        segments: List[JointState] = []
        total_time = 0.0
        solve_time = 0.0
        device = q.position.device

        for i, goal_state in enumerate(goal_states):
            if goal_state.ndim > 2:
                log_and_raise(
                    f"goal_states[{i}] must be a 2D tensor, got shape: {goal_state.shape}"
                )
            goal = self.kinematics.get_active_js(_squeeze_joint_state_to_2d(goal_state))
            result = self.plan_cspace(
                goal,
                q,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
            )
            if result is not None:
                total_time += float(result.total_time)
                solve_time += float(result.solve_time)

            if result is None or not result.success.any():
                return WaypointPlanResult(
                    success=torch.zeros(1, dtype=torch.bool, device=device),
                    interpolated_trajectory=(
                        _concat_interpolated_segments(segments) if segments else None
                    ),
                    segment_trajectories=segments or None,
                    failed_index=i,
                    status=f"segment {i} failed: {format_plan_failure(result, self.last_ik_result)}",
                    failed_result=result,
                    total_time=total_time,
                    solve_time=solve_time,
                    planning_time=total_time,
                )

            traj = result.get_interpolated_plan()
            segments.append(traj)
            end_state = get_joint_state_at_horizon_index(
                traj, traj.position.shape[-2] - 1
            )
            q = self.kinematics.get_active_js(_squeeze_joint_state_to_2d(end_state))

        interpolated, status = self._select_waypoint_interpolated_trajectory(segments)

        return WaypointPlanResult(
            success=torch.ones(1, dtype=torch.bool, device=device),
            interpolated_trajectory=interpolated,
            segment_trajectories=segments,
            failed_index=None,
            status=status,
            total_time=total_time,
            solve_time=solve_time,
            planning_time=total_time,
        )

    def _select_waypoint_interpolated_trajectory(
        self, segments: List[JointState]
    ) -> Tuple[JointState, Optional[str]]:
        """Retime concatenated via segments with a flyby spline if it stays collision-free."""
        if len(segments) < 2:
            return _concat_interpolated_segments(segments), None

        interpolated = _flyby_spline_through_segments(
            segments,
            interpolation_dt=float(self.trajopt_solver.config.interpolation_dt),
            max_velocity=self.trajopt_solver.transition_model.max_velocity,
            max_acceleration=self.trajopt_solver.transition_model.max_acceleration,
        )
        if self._joint_traj_is_collision_free(interpolated):
            return interpolated, None

        log_warn(
            "Flyby spline collided after smoothing; falling back to concatenated "
            "rest-to-rest segments"
        )
        return (
            _concat_interpolated_segments(segments),
            "flyby spline in collision; using rest-to-rest concatenation",
        )

    def _joint_traj_is_collision_free(self, traj: JointState) -> bool:
        """Return True if every sample on ``traj`` is feasible (scene, self, bounds)."""
        if self.graph_planner is None or traj is None or traj.position is None:
            return True
        pos = _traj_position_2d(traj)
        if pos.shape[0] < 1:
            return True
        active = self.kinematics.get_active_js(
            JointState.from_position(pos, joint_names=traj.joint_names)
        )
        q = _traj_position_2d(active)
        if q.shape[-1] != self.graph_planner.action_dim:
            log_and_raise(
                "Active trajectory dof "
                f"{q.shape[-1]} != graph planner action_dim {self.graph_planner.action_dim}"
            )
        feasible = self.graph_planner.check_samples_feasibility(q)
        return bool(torch.as_tensor(feasible).reshape(-1).all().item())

    def _plan_pose_single(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        max_attempts: int,
        enable_graph_attempt: int,
    ) -> Optional[TrajOptSolverResult]:
        """Single-goal planning with retry, seed repair, and graph seeding."""
        self.last_ik_result = None
        trajopt_result = None
        total_time = 0.0
        solve_time = 0.0
        og_current_state = current_state.clone()
        num_seeds = self.trajopt_solver.config.num_seeds

        for current_attempt in range(max_attempts):
            current_state = og_current_state.clone()
            ik_result = self.ik_solver.solve_pose(
                goal_tool_poses,
                return_seeds=num_seeds,
                current_state=current_state,
            )
            self.last_ik_result = ik_result
            total_time += ik_result.total_time
            solve_time += ik_result.solve_time

            success_count = torch.count_nonzero(ik_result.success)
            if success_count == 0:
                continue

            seed_config = ik_result.solution
            if success_count < num_seeds:
                good_solution = seed_config[ik_result.success][0:1, :].clone()
                seed_config[~ik_result.success][:, :] = good_solution

            seed_traj = None
            finetune_attempts = 1
            finetune_dt_scale = 0.55
            if current_attempt >= enable_graph_attempt and self.graph_planner is not None:
                graph_seed = self._get_graph_seed_trajectories(
                    current_state, seed_config,
                )
                if graph_seed is None:
                    continue
                seed_traj = graph_seed
                total_time += 0.0  # graph time already in graph_seed call
                finetune_attempts = 3
                finetune_dt_scale = 0.75

            trajopt_result = self.trajopt_solver.solve_pose(
                goal_tool_poses, current_state,
                seed_config=seed_config,
                seed_traj=seed_traj,
                use_implicit_goal=True,
                finetune_attempts=finetune_attempts,
                finetune_dt_scale=finetune_dt_scale,
            )
            total_time += trajopt_result.total_time
            solve_time += trajopt_result.solve_time
            if torch.count_nonzero(trajopt_result.success) > 0:
                break

        if trajopt_result is not None:
            trajopt_result.total_time = total_time
            trajopt_result.solve_time = solve_time
        return trajopt_result

    def _plan_pose_goalset(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 10,
    ) -> Optional[TrajOptSolverResult]:
        """Goalset planning: IK + TrajOpt, no graph seeding."""
        self.last_ik_result = None
        for _ in range(max_attempts):
            ik_result = self.ik_solver.solve_pose(
                goal_tool_poses,
                return_seeds=self.trajopt_solver.config.num_seeds,
                current_state=current_state,
            )
            self.last_ik_result = ik_result
            if torch.count_nonzero(ik_result.success) == 0:
                return None

            trajopt_result = self.trajopt_solver.solve_pose(
                goal_tool_poses, current_state,
                seed_config=ik_result.solution,
                use_implicit_goal=use_implicit_goal,
            )
            if torch.count_nonzero(trajopt_result.success) > 0:
                break

        return trajopt_result

    def plan_cspace(
        self,
        goal_state: JointState,
        current_state: JointState,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
    ) -> Optional[TrajOptSolverResult]:
        """Plan a collision-free trajectory to a joint configuration.

        Args:
            goal_state: Target joint configuration.
            current_state: Initial joint state.
            max_attempts: Maximum planning attempts.
            enable_graph_attempt: Attempt at which to start graph seeding.

        Returns:
            TrajOptSolverResult, or None if planning failed.
        """
        if current_state.ndim > 2:
            log_and_raise(f"current_state must be a 2D tensor, got shape: {current_state.shape}")
        if goal_state.ndim > 2:
            log_and_raise(f"goal_state must be a 2D tensor, got shape: {goal_state.shape}")

        trajopt_result = None
        total_time = 0.0
        solve_time = 0.0
        og_current_state = current_state.clone()
        num_seeds = self.trajopt_solver.config.num_seeds

        for current_attempt in range(max_attempts):
            current_state = og_current_state.clone()
            seed_traj = None
            finetune_attempts = 3
            finetune_dt_scale = 0.75

            if current_attempt >= enable_graph_attempt and self.graph_planner is not None:
                goal_configs = goal_state.position.view(1, 1, -1).repeat(1, num_seeds, 1)
                graph_seed = self._get_graph_seed_trajectories(
                    current_state, goal_configs,
                )
                if graph_seed is None:
                    continue
                seed_traj = graph_seed
                finetune_attempts = 3
                finetune_dt_scale = 0.75

            trajopt_result = self.trajopt_solver.solve_cspace(
                goal_state, current_state, seed_traj=seed_traj,
                finetune_attempts=finetune_attempts,
                finetune_dt_scale=finetune_dt_scale,
            )
            total_time += trajopt_result.total_time
            solve_time += trajopt_result.solve_time
            if torch.count_nonzero(trajopt_result.success) > 0:
                break

        if trajopt_result is not None:
            trajopt_result.total_time = total_time
            trajopt_result.solve_time = solve_time
        return trajopt_result

    def _get_graph_seed_trajectories(
        self,
        current_state: JointState,
        seed_config: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Query graph planner for trajectory seeds.

        Args:
            current_state: Start state ``(1, dof)``.
            seed_config: Goal configs ``(1, num_seeds, dof)``.

        Returns:
            Seed trajectories ``(1, n_success, horizon, dof)`` or None.
        """
        dof = self.trajopt_solver.action_dim
        num_seeds = seed_config.shape[1] if seed_config.ndim == 3 else seed_config.shape[0]

        graph_starts = current_state.position.view(1, dof).repeat(num_seeds, 1)
        graph_goals = seed_config.view(num_seeds, dof)

        result = self.graph_planner.find_path(
            graph_starts.clone(), graph_goals.clone(),
            interpolate_waypoints=True,
            interpolation_steps=self.trajopt_solver.action_horizon,
            interpolation_type=TrajInterpolationType.LINEAR,
            validate_interpolated_trajectory=False,
        )
        if torch.count_nonzero(result.success) == 0:
            return None
        return result.interpolated_waypoints[result.success, :, :].unsqueeze(0)

    # -- Grasp planning --

    def plan_grasp(
        self,
        grasp_poses: GoalToolPose,
        current_state: JointState,
        grasp_approach_axis: str = "z",
        grasp_approach_offset: float = -0.15,
        grasp_approach_in_tool_frame: bool = True,
        grasp_lift_axis: str = "z",
        grasp_lift_offset: float = -0.15,
        grasp_lift_in_tool_frame: bool = True,
        plan_approach_to_grasp: bool = True,
        plan_grasp_to_lift: bool = True,
        disable_collision_links: List[str] = None,
    ) -> GraspPlanResult:
        """Plan a grasp motion: goalset -> approach -> grasp -> lift.

        Args:
            grasp_poses: Candidate grasp poses with ``num_goalset`` entries per link.
                Construct via ``GoalToolPose`` with the desired goalset size.
        """
        if disable_collision_links is None:
            disable_collision_links = self.kinematics.config.kinematics_config.grasp_contact_link_names
        if disable_collision_links is None:
            disable_collision_links = []

        device = current_state.device
        result = GraspPlanResult()
        result.success = torch.tensor([False], device=device)
        result.approach_success = torch.tensor([False], device=device)
        result.grasp_success = torch.tensor([False], device=device)
        result.lift_success = torch.tensor([False], device=device)

        # Step 1: Plan to one of the grasp poses (goalset)
        self.disable_link_collision(disable_collision_links)
        goalset_result = self.plan_pose(grasp_poses, current_state)
        self.enable_link_collision(disable_collision_links)

        if goalset_result is None:
            result.status = "Goalset planning returned None."
            return result

        result.success = goalset_result.success.clone()
        result.success[:] = False
        result.goalset_result = goalset_result

        if not goalset_result.success.any():
            result.status = "No grasp in goal set was reachable."
            return result

        result.goalset_index = goalset_result.goalset_index.clone()
        goal_index = int(goalset_result.goalset_index.view(-1)[0].item())

        grasp_poses_dict = {}
        for frame in grasp_poses.tool_frames:
            li = grasp_poses.tool_frames.index(frame)
            pos = grasp_poses.position[:, 0, li, goal_index, :]
            quat = grasp_poses.quaternion[:, 0, li, goal_index, :]
            grasp_poses_dict[frame] = Pose(position=pos, quaternion=quat)

        # Step 2: Plan to approach pose (offset from grasp)
        approach_offset = Pose.from_list([
            grasp_approach_offset * v
            for v in _axis_string_to_vector(grasp_approach_axis)
        ] + [1, 0, 0, 0])

        approach_poses_dict = {}
        for frame, goal_pose in grasp_poses_dict.items():
            if grasp_approach_in_tool_frame:
                approach_poses_dict[frame] = goal_pose.multiply(approach_offset)
            else:
                approach_poses_dict[frame] = approach_offset.multiply(goal_pose)

        approach_tool_pose = GoalToolPose.from_poses(
            approach_poses_dict, ordered_tool_frames=grasp_poses.tool_frames, num_goalset=1,
        )
        approach_result = self.plan_pose(approach_tool_pose, current_state)
        result.approach_result = approach_result

        if approach_result is None or not approach_result.success.any():
            result.status = "Planning to approach pose failed."
            result.approach_success[:] = False
            result.success[:] = False
            return result

        result.approach_success = approach_result.success.clone()
        result.approach_trajectory = approach_result.js_solution
        result.approach_trajectory_dt = approach_result.js_solution.dt
        result.approach_interpolated_trajectory = approach_result.interpolated_trajectory
        result.approach_interpolated_last_tstep = approach_result.interpolated_last_tstep
        result.status = "Planning to approach pose succeeded."

        if not plan_approach_to_grasp:
            result.success[:] = True
            return result

        # Step 3: Linear motion from approach to grasp
        approach_end = get_joint_state_at_horizon_index(approach_result.js_solution, -1).squeeze(0)
        approach_end = self.kinematics.get_active_js(approach_end)

        linear_motion = ToolPoseCriteria.linear_motion(
            axis=grasp_approach_axis, non_terminal_scale=1.0,
            project_distance_to_goal=grasp_approach_in_tool_frame,
        )
        self.update_tool_pose_criteria({k: linear_motion for k in grasp_poses.tool_frames})

        grasp_tool_pose = GoalToolPose.from_poses(
            grasp_poses_dict, ordered_tool_frames=grasp_poses.tool_frames, num_goalset=1,
        )
        self.disable_link_collision(disable_collision_links)
        grasp_result = self.plan_pose(grasp_tool_pose, approach_end)
        self.enable_link_collision(disable_collision_links)

        standard_criteria = {k: ToolPoseCriteria() for k in grasp_poses.tool_frames}
        self.update_tool_pose_criteria(standard_criteria)

        if grasp_result is None or not grasp_result.success.any():
            result.status = "Planning to grasp pose failed."
            result.grasp_success[:] = False
            result.success[:] = False
            return result

        result.grasp_trajectory = grasp_result.js_solution
        result.grasp_trajectory_dt = grasp_result.js_solution.dt
        result.grasp_interpolated_trajectory = grasp_result.interpolated_trajectory
        result.grasp_interpolated_last_tstep = grasp_result.interpolated_last_tstep
        result.success = grasp_result.success.clone()
        result.grasp_success = grasp_result.success.clone()

        if not plan_grasp_to_lift:
            return result

        # Step 4: Lift motion
        lift_start = get_joint_state_at_horizon_index(grasp_result.js_solution, -1).squeeze(0)
        lift_start = self.kinematics.get_active_js(lift_start)

        lift_offset = Pose.from_list([
            grasp_lift_offset * v
            for v in _axis_string_to_vector(grasp_lift_axis)
        ] + [1, 0, 0, 0])

        lift_poses_dict = {}
        for frame, goal_pose in grasp_poses_dict.items():
            if grasp_lift_in_tool_frame:
                lift_poses_dict[frame] = goal_pose.multiply(lift_offset)
            else:
                lift_poses_dict[frame] = lift_offset.multiply(goal_pose)

        lift_tool_pose = GoalToolPose.from_poses(
            lift_poses_dict, ordered_tool_frames=grasp_poses.tool_frames, num_goalset=1,
        )
        lift_linear = ToolPoseCriteria.linear_motion(
            axis=grasp_lift_axis, non_terminal_scale=1.0,
            project_distance_to_goal=grasp_lift_in_tool_frame,
        )
        self.update_tool_pose_criteria({k: lift_linear for k in grasp_poses.tool_frames})

        self.disable_link_collision(disable_collision_links)
        lift_result = self.plan_pose(lift_tool_pose, lift_start)
        self.enable_link_collision(disable_collision_links)
        self.update_tool_pose_criteria(standard_criteria)

        if lift_result is None or not lift_result.success.any():
            result.status = "Planning to lift pose failed."
            result.success[:] = False
            result.lift_success[:] = False
            return result

        result.lift_trajectory = lift_result.js_solution
        result.lift_trajectory_dt = lift_result.js_solution.dt
        result.lift_interpolated_trajectory = lift_result.interpolated_trajectory
        result.lift_interpolated_last_tstep = lift_result.interpolated_last_tstep
        result.success = lift_result.success.clone()
        result.lift_success = lift_result.success.clone()
        result.status = "Planning to lift pose succeeded."
        return result

    # -- Scene / collision management --

    def enable_link_collision(self, enable_collision_links: List[str]):
        for link_name in enable_collision_links:
            self.kinematics.config.kinematics_config.enable_link_spheres(link_name)

    def disable_link_collision(self, disable_collision_links: List[str]):
        for link_name in disable_collision_links:
            self.kinematics.config.kinematics_config.disable_link_spheres(link_name)

    def update_world(self, scene_cfg: SceneCfg):
        self.scene_collision_checker.load_collision_model(scene_cfg)
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()

    def clear_scene_cache(self):
        self.scene_collision_checker.clear_cache()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()

    def reset_seed(self):
        self.ik_solver.reset_seed()
        self.trajopt_solver.reset_seed()
        if self.graph_planner is not None:
            self.graph_planner.reset_buffer()
            self.graph_planner.reset_seed()

    def update_link_inertial(
        self, link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ) -> None:
        self.ik_solver.update_link_inertial(link_name, mass, com, inertia)
        self.trajopt_solver.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(
        self, link_properties: dict[str, dict[str, Union[float, torch.Tensor]]],
    ) -> None:
        self.ik_solver.update_links_inertial(link_properties)
        self.trajopt_solver.update_links_inertial(link_properties)

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        self.ik_solver.update_tool_pose_criteria(tool_pose_criteria)
        self.trajopt_solver.update_tool_pose_criteria(tool_pose_criteria)
