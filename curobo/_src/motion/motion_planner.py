# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-problem motion planner with retry logic and graph-planner seeding."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, Union

import torch

from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.cost.cost_cspace_state import StateCSpaceCost
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.collision.collision_scene import create_scene_collision
from curobo._src.geom.quaternion import angular_distance_axis_angle
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
from curobo._src.util.logging import log_and_raise, log_info, log_warn


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


def waypoint_time_fractions(
    points: torch.Tensor,
    max_velocity: torch.Tensor,
    max_acceleration: torch.Tensor,
) -> Optional[List[float]]:
    """Time fractions for ordered joint waypoints, including the start and goal.

    Each segment lasts as long as its largest joint needs at the velocity and
    acceleration limits. A short move therefore takes a small share of the
    horizon, and the goal fraction is always 1.
    """
    if points.ndim != 2 or points.shape[0] < 2:
        return None
    if not bool(torch.isfinite(points).all()):
        return None
    vmax = max_velocity.detach().reshape(-1).to(device=points.device, dtype=points.dtype)
    amax = max_acceleration.detach().reshape(-1).to(device=points.device, dtype=points.dtype)
    n_lim = min(int(points.shape[-1]), int(vmax.numel()), int(amax.numel()))
    if n_lim < 1:
        return None
    vmax = vmax[:n_lim].abs().clamp_min(1e-6)
    amax = amax[:n_lim].abs().clamp_min(1e-6)
    durations: List[float] = []
    for i in range(int(points.shape[0]) - 1):
        dq = (points[i + 1, :n_lim] - points[i, :n_lim]).abs()
        dt_v = float((dq / vmax).amax().item())
        dt_a = float((2.0 * dq / amax).clamp_min(0.0).sqrt().amax().item())
        durations.append(max(dt_v, dt_a, 1e-4))
    total = float(sum(durations))
    if not math.isfinite(total) or total <= 0.0:
        return None
    fractions = [0.0]
    elapsed = 0.0
    for duration in durations[:-1]:
        elapsed += duration
        fractions.append(elapsed / total)
    fractions.append(1.0)
    return fractions


def horizon_steps_for_fractions(
    fractions: List[float], horizon: int
) -> Optional[List[int]]:
    """Place intermediate fractions on distinct rollout steps, goal excluded."""
    if horizon < 3 or len(fractions) < 3:
        return None
    if fractions[0] != 0.0 or fractions[-1] != 1.0:
        return None
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in fractions):
        return None
    if any(b < a for a, b in zip(fractions, fractions[1:])):
        return None
    last = horizon - 1
    steps: List[int] = []
    used = set()
    for fraction in fractions[1:-1]:
        step = int(round(float(fraction) * last))
        step = min(max(step, 1), horizon - 2)
        while step in used and step < horizon - 2:
            step += 1
        if step in used:
            return None
        used.add(step)
        steps.append(step)
    return steps


_IK_BRANCH_SEPARATION = 0.25


def _nearest_and_alternate(
    tip: JointState,
    candidates: List[JointState],
    min_separation: float,
) -> Tuple[JointState, Optional[JointState]]:
    """Closest candidate to ``tip``, plus one that differs by at least ``min_separation`` rad."""
    tip_q = tip.position.reshape(-1)

    def distance(state: JointState) -> float:
        return float(torch.linalg.vector_norm(state.position.reshape(-1) - tip_q).item())

    ordered = sorted(candidates, key=distance)
    nearest = ordered[0]
    nearest_q = nearest.position.reshape(-1)
    alternate = None
    for other in ordered[1:]:
        gap = float(torch.max(torch.abs(other.position.reshape(-1) - nearest_q)).item())
        if gap >= min_separation:
            alternate = other
            break
    return nearest, alternate


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


# Collision samples stay on this clock even when the published trajectory is coarser.
_FLYBY_COLLISION_DT = 0.02
_FLYBY_COLLISION_MAX_STEP = 0.05
_FLYBY_MAX_COLLISION_SAMPLES = 4096


def _as_limit_vector(limit: Optional[torch.Tensor]) -> Optional["np.ndarray"]:
    import numpy as np

    if limit is None:
        return None
    arr = np.asarray(limit.detach().reshape(-1).cpu().numpy(), dtype=np.float64)
    if arr.size < 1:
        return None
    return arr


def _cpu_spline_knots(
    segments: List[JointState], min_knot_delta: float
) -> Optional["np.ndarray"]:
    """Copy every segment to CPU once, then keep vias and moved samples."""
    import numpy as np

    chunks: List[torch.Tensor] = []
    via_rows: List[int] = []
    cursor = 0
    for segment in segments:
        pos = _traj_position_2d(segment)
        n_steps = int(pos.shape[0])
        if n_steps < 1:
            continue
        start_t = 0 if not chunks else 1
        if start_t >= n_steps:
            continue
        chunk = pos[start_t:]
        n_chunk = int(chunk.shape[0])
        via_rows.append(cursor + n_chunk - 1)
        chunks.append(chunk)
        cursor += n_chunk
    if not chunks:
        return None

    q_all = torch.cat(chunks, dim=0).detach().cpu().numpy()
    keep = [0]
    last = q_all[0]
    via = set(via_rows)
    for row in range(1, int(q_all.shape[0])):
        delta = float(np.max(np.abs(q_all[row] - last)))
        is_via = row in via
        if not is_via and delta < min_knot_delta:
            continue
        if is_via and delta < 1e-8:
            continue
        keep.append(row)
        last = q_all[row]
    return np.ascontiguousarray(q_all[keep])


def _cubic_spline_time_scale(spline, t_knots, vmax, amax, jmax) -> float:
    """Scale time from interval extrema of a clamped cubic, including jerk.

    Velocity is quadratic inside each piece, so the peak can sit between
    knots. Acceleration is linear and jerk is constant on a piece.
    """
    import numpy as np

    c = np.asarray(spline.c, dtype=np.float64)
    h = np.diff(np.asarray(t_knots, dtype=np.float64))
    n_lim = min(int(c.shape[-1]), int(vmax.shape[0]), int(amax.shape[0]))
    if jmax is not None:
        n_lim = min(n_lim, int(jmax.shape[0]))
    if n_lim < 1 or h.size < 1:
        return 1.0
    c0 = c[0, :, :n_lim]
    c1 = c[1, :, :n_lim]
    c2 = c[2, :, :n_lim]
    width = h[:, None]

    vel0 = np.abs(c2)
    vel1 = np.abs(c2 + 2.0 * c1 * width + 3.0 * c0 * width * width)
    vel = np.maximum(vel0, vel1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(np.abs(c0) > 1e-12, -c1 / (3.0 * c0), np.nan)
    inside = np.isfinite(dx) & (dx > 0.0) & (dx < width)
    vel_star = np.abs(c2 + 2.0 * c1 * dx + 3.0 * c0 * dx * dx)
    vel = np.maximum(vel, np.where(inside, vel_star, 0.0))

    acc = np.maximum(np.abs(2.0 * c1), np.abs(2.0 * c1 + 6.0 * c0 * width))
    jerk = np.abs(6.0 * c0)
    v_lim = np.maximum(vmax[:n_lim], 1e-6)
    a_lim = np.maximum(amax[:n_lim], 1e-6)
    s_v = float(np.max(vel / v_lim))
    s_a = float(np.max(np.sqrt(acc / a_lim)))
    s_j = 1.0
    if jmax is not None:
        j_lim = np.maximum(jmax[:n_lim], 1e-6)
        s_j = float(np.max(np.cbrt(jerk / j_lim)))
    scale = float(np.nanmax([1.0, s_v, s_a, s_j]))
    if not math.isfinite(scale) or scale < 1.0:
        return 1.0
    return scale


def _flyby_spline_through_segments(
    segments: List[JointState],
    interpolation_dt: float,
    max_velocity: torch.Tensor,
    max_acceleration: torch.Tensor,
    max_jerk: Optional[torch.Tensor] = None,
    min_knot_delta: float = 0.04,
    collision_dt: float = _FLYBY_COLLISION_DT,
) -> Tuple[JointState, Optional[torch.Tensor]]:
    """Retiming spline that passes through via configs without resting.

    TrajOpt segments are still rest-to-rest. This keeps a sparse subset of
    those collision-aware samples (including each via) and fits a clamped
    cubic spline so only the global start/goal have zero velocity.

    The returned trajectory is sampled at ``interpolation_dt``. The second
    value is a finer position sequence for collision acceptance, on a clock
    that does not follow the publish step.
    """
    from scipy.interpolate import CubicSpline
    import numpy as np

    q_np = _cpu_spline_knots(segments, min_knot_delta)
    if q_np is None or int(q_np.shape[0]) < 3:
        return _concat_interpolated_segments(segments), None

    n_knots, dof = q_np.shape
    vmax = _as_limit_vector(max_velocity)
    amax = _as_limit_vector(max_acceleration)
    jmax = _as_limit_vector(max_jerk)
    if vmax is None or amax is None:
        return _concat_interpolated_segments(segments), None
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
    time_scale = _cubic_spline_time_scale(spline, t_knots, vmax, amax, jmax)
    duration = float(t_knots[-1]) * time_scale
    n_eval = max(int(np.ceil(duration / max(interpolation_dt, 1e-4))) + 1, 2)
    t_out = np.linspace(0.0, duration, n_eval)
    tau = t_out / time_scale
    q_eval = spline(tau)
    v_eval = spline(tau, 1) / time_scale
    a_eval = spline(tau, 2) / (time_scale * time_scale)
    j_eval = spline(tau, 3) / (time_scale ** 3)

    peak_v = float(np.max(vmax)) if vmax.size else 1.0
    step_dt = _FLYBY_COLLISION_MAX_STEP / max(peak_v, 1e-6)
    sample_dt = min(collision_dt, step_dt)
    if not math.isfinite(sample_dt) or sample_dt <= 0.0:
        sample_dt = _FLYBY_COLLISION_DT
    n_col = int(np.ceil(duration / sample_dt)) + 1
    n_col = min(max(n_col, 2), _FLYBY_MAX_COLLISION_SAMPLES)
    tau_check = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, float(t_knots[-1]), n_col),
                t_knots,
                tau,
            )
        )
    )
    q_check = spline(tau_check)

    ref = segments[0]
    device = ref.position.device
    dtype = ref.position.dtype

    def _to_ref(values: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(values)).to(device=device, dtype=dtype)
        return _unsqueeze_like_horizon(tensor, ref.position)

    position = _to_ref(q_eval)
    if ref.dt is not None:
        dt = torch.full_like(ref.dt, interpolation_dt)
    else:
        dt = torch.tensor([interpolation_dt], device=device, dtype=dtype)

    trajectory = JointState(
        position=position,
        velocity=_to_ref(v_eval),
        acceleration=_to_ref(a_eval),
        jerk=_to_ref(j_eval),
        joint_names=ref.joint_names,
        device_cfg=ref.device_cfg,
        dt=dt,
        control_space=ref.control_space,
    )
    check_q = torch.from_numpy(np.ascontiguousarray(q_check)).to(device=device, dtype=dtype)
    return trajectory, check_q


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
        self.waypoint_joint_tolerance = 0.1
        self._timing_depth = 0
        self._timing: Dict[str, float] = {}
        self._timing_t0 = 0.0
        self.last_timing: Dict[str, float] = {}
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

    def _timing_enter(self) -> None:
        if self._timing_depth == 0:
            self._timing = {}
            self._timing_t0 = time.perf_counter()
        self._timing_depth += 1

    def _timing_add(self, name: str, seconds: float) -> None:
        if self._timing_depth <= 0:
            return
        self._timing[name] = self._timing.get(name, 0.0) + float(seconds)

    def _timing_leave(self) -> None:
        self._timing_depth = max(0, self._timing_depth - 1)
        if self._timing_depth == 0:
            self.last_timing = dict(self._timing)
            self.last_timing["wall"] = time.perf_counter() - self._timing_t0

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
        finetune_attempts: Optional[int] = None,
    ) -> Optional[TrajOptSolverResult]:
        """Plan a trajectory to reach target tool poses.

        Goalset is auto-detected from ``goal_tool_poses.num_goalset``.  When
        ``num_goalset > 1`` the planner uses a simpler IK+TrajOpt loop without
        graph seeding.
        """
        if current_state.ndim > 2:
            log_and_raise(f"current_state must be a 2D tensor, got shape: {current_state.shape}")

        self._timing_enter()
        try:
            if goal_tool_poses.num_goalset > 1:
                return self._plan_pose_goalset(
                    goal_tool_poses, current_state, use_implicit_goal, max_attempts,
                    finetune_attempts=finetune_attempts,
                )
            return self._plan_pose_single(
                goal_tool_poses, current_state, max_attempts, enable_graph_attempt,
                finetune_attempts=finetune_attempts,
            )
        finally:
            self._timing_leave()

    def plan_cartesian_waypoints(
        self,
        goal_tool_poses: List[GoalToolPose],
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
        finetune_attempts: Optional[int] = None,
    ) -> WaypointPlanResult:
        """Plan one trajectory that passes through ordered Cartesian waypoints.

        Each pose is solved with IK, then one configuration-space optimization
        has to pass through those joint configurations in order. If that solve
        cannot be set up or misses a waypoint, each pose is planned on its own
        and the flyby spline is used instead.
        """
        if not goal_tool_poses:
            log_and_raise("plan_cartesian_waypoints requires at least one goal pose")
        self._timing_enter()
        try:
            t_wall = time.perf_counter()
            if len(goal_tool_poses) >= 2 and all(
                goal.num_goalset <= 1 for goal in goal_tool_poses
            ):
                t_ik = time.perf_counter()
                chains = self._ik_waypoint_joint_states(goal_tool_poses, current_state)
                self._timing_add("ik", time.perf_counter() - t_ik)
                if chains:
                    solved, joint_s, check_s = self._solve_joint_waypoints_once(
                        chains[0],
                        current_state,
                        finetune_attempts,
                        poses=goal_tool_poses,
                        seed_chains=chains,
                    )
                    self._timing_add("trajopt", joint_s)
                    self._timing_add("check", check_s)
                    if solved is not None:
                        return self._finish_waypoint_timing(solved, t_wall)
                log_warn(
                    "Single-optimization Cartesian waypoint solve failed; "
                    "falling back to segmented planning"
                )
            result = self._plan_cartesian_waypoints_segmented(
                goal_tool_poses,
                current_state,
                use_implicit_goal=use_implicit_goal,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
                finetune_attempts=finetune_attempts,
            )
            return self._finish_waypoint_timing(result, t_wall)
        finally:
            self._timing_leave()

    def plan_joint_waypoints(
        self,
        goal_states: List[JointState],
        current_state: JointState,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
        finetune_attempts: int = 3,
    ) -> WaypointPlanResult:
        """Plan one trajectory that passes through ordered joint waypoints.

        Intermediate waypoints are costs inside a single optimization whose
        terminal state is the last waypoint. The arm is not planned to stop at
        the intermediate waypoints. If that solve misses a waypoint, each
        waypoint is planned as its own rest-to-rest segment.
        """
        if not goal_states:
            log_and_raise("plan_joint_waypoints requires at least one goal state")
        self._timing_enter()
        try:
            t_wall = time.perf_counter()
            if len(goal_states) >= 2:
                solved, joint_s, check_s = self._solve_joint_waypoints_once(
                    goal_states, current_state, finetune_attempts
                )
                self._timing_add("trajopt", joint_s)
                self._timing_add("check", check_s)
                if solved is not None:
                    return self._finish_waypoint_timing(solved, t_wall)
                log_warn(
                    "Single-optimization joint waypoint solve failed; "
                    "falling back to segmented planning"
                )
            result = self._plan_joint_waypoints_segmented(
                goal_states,
                current_state,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
                finetune_attempts=finetune_attempts,
            )
            return self._finish_waypoint_timing(result, t_wall)
        finally:
            self._timing_leave()

    def _ik_waypoint_joint_states(
        self,
        goal_tool_poses: List[GoalToolPose],
        current_state: JointState,
    ) -> Optional[List[List[JointState]]]:
        """Return up to two joint chains, each passing the poses in order.

        The first chain picks, at every pose, the successful IK solution
        closest to the previous configuration. A second chain is forked once
        from a clearly different IK solution so the optimizer gets two
        different initial paths.
        """
        q = self.kinematics.get_active_js(
            _squeeze_joint_state_to_2d(current_state.clone())
        )
        num_seeds = int(self.trajopt_solver.config.num_seeds)
        max_branches = min(max(num_seeds, 1), 2)
        tips: List[JointState] = [q]
        chains: List[List[JointState]] = [[]]
        forked = False
        for goal in goal_tool_poses:
            next_tips: List[JointState] = []
            next_chains: List[List[JointState]] = []
            for tip, chain in zip(tips, chains):
                candidates = self._ik_candidate_states(goal, tip, num_seeds)
                if not candidates:
                    continue
                nearest, alternate = _nearest_and_alternate(
                    tip, candidates, _IK_BRANCH_SEPARATION
                )
                next_tips.append(nearest)
                next_chains.append(chain + [nearest])
                if (
                    not forked
                    and alternate is not None
                    and max_branches > 1
                    and len(tips) == 1
                ):
                    next_tips.append(alternate)
                    next_chains.append(chain + [alternate])
                    forked = True
            if not next_tips:
                return None
            tips, chains = next_tips, next_chains
        return chains

    def _ik_candidate_states(
        self,
        goal: GoalToolPose,
        tip: JointState,
        num_seeds: int,
    ) -> List[JointState]:
        """Successful IK solutions for one pose, in the active joint order."""
        ik_result = self.ik_solver.solve_pose(
            goal, return_seeds=num_seeds, current_state=tip
        )
        self.last_ik_result = ik_result
        if ik_result is None or ik_result.solution is None or not ik_result.success.any():
            return []
        solution = ik_result.solution.reshape(-1, ik_result.solution.shape[-1])
        success = ik_result.success.reshape(-1)
        n = min(int(solution.shape[0]), int(success.shape[0]))
        hits = success[:n].nonzero(as_tuple=False)
        states: List[JointState] = []
        dof = int(tip.position.reshape(-1).shape[0])
        for hit in hits.flatten().tolist():
            state = self.kinematics.get_active_js(
                JointState.from_position(
                    solution[int(hit)].view(1, -1),
                    joint_names=list(self.ik_solver.joint_names),
                )
            )
            if int(state.position.reshape(-1).shape[0]) != dof:
                continue
            states.append(state)
        return states

    def _solve_joint_waypoints_once(
        self,
        goal_states: List[JointState],
        current_state: JointState,
        finetune_attempts: Optional[int],
        poses: Optional[List[GoalToolPose]] = None,
        seed_chains: Optional[List[List[JointState]]] = None,
    ) -> Tuple[Optional[WaypointPlanResult], float, float]:
        """One ``solve_cspace`` whose intermediate states pass through the waypoints.

        Returns the plan, the joint-solve wall time, and the acceptance-check
        wall time. Setup that fails before the solve reports that time as the
        joint stage and zero check time.
        """
        t0 = time.perf_counter()

        def _bail() -> Tuple[Optional[WaypointPlanResult], float, float]:
            return None, time.perf_counter() - t0, 0.0

        if len(goal_states) < 2:
            return _bail()
        q = self.kinematics.get_active_js(
            _squeeze_joint_state_to_2d(current_state.clone())
        )
        active_goals: List[JointState] = []
        for goal_state in goal_states:
            if goal_state.ndim > 2:
                return _bail()
            active_goals.append(
                self.kinematics.get_active_js(_squeeze_joint_state_to_2d(goal_state))
            )
        dof = int(q.position.shape[-1])
        if any(int(goal.position.shape[-1]) != dof for goal in active_goals):
            return _bail()
        points = torch.stack(
            [q.position.reshape(-1)[:dof]]
            + [goal.position.reshape(-1)[:dof] for goal in active_goals],
            dim=0,
        )
        seed_points = points
        if seed_chains:
            branch_rows = []
            for chain in seed_chains:
                if len(chain) != len(active_goals):
                    continue
                rows = [q.position.reshape(-1)[:dof]]
                ok = True
                for goal_state in chain:
                    active = self.kinematics.get_active_js(
                        _squeeze_joint_state_to_2d(goal_state)
                    )
                    if int(active.position.reshape(-1).shape[0]) != dof:
                        ok = False
                        break
                    rows.append(active.position.reshape(-1)[:dof])
                if ok:
                    branch_rows.append(torch.stack(rows, dim=0))
            if branch_rows:
                seed_points = torch.stack(branch_rows, dim=0)
        intermediate = points[1:-1]
        if intermediate.shape[0] < 1 or intermediate.shape[0] > 16:
            return _bail()
        action_horizon = int(self.trajopt_solver.action_horizon)
        if action_horizon < 2:
            return _bail()
        fractions = waypoint_time_fractions(
            points,
            self.trajopt_solver.transition_model.max_velocity,
            self.trajopt_solver.transition_model.max_acceleration,
        )
        steps = (
            None
            if fractions is None
            else horizon_steps_for_fractions(fractions, int(self.trajopt_solver.horizon))
        )
        if fractions is None or steps is None:
            return _bail()
        seed = self._polyline_seed(
            seed_points,
            action_horizon,
            int(self.trajopt_solver.config.num_seeds),
            fractions,
        )
        steps = self._arm_waypoint_costs(intermediate, steps)
        if steps is None:
            return _bail()
        try:
            result = self.trajopt_solver.solve_cspace(
                active_goals[-1],
                q,
                seed_traj=seed,
                finetune_attempts=1 if finetune_attempts is None else int(finetune_attempts),
                finetune_dt_scale=0.75,
            )
        except Exception as exc:
            log_warn(f"Single-optimization waypoint solve raised {exc!r}")
            return _bail()
        finally:
            self._clear_waypoint_costs()
        joint_s = time.perf_counter() - t0
        t_check = time.perf_counter()
        if result is None or not result.success.any():
            return None, joint_s, time.perf_counter() - t_check
        traj = result.get_interpolated_plan()
        state_dt = self._solution_state_dt(result)
        if traj is None:
            log_warn("Single-optimization trajectory is empty")
            accepted = False
        elif poses is None:
            accepted = self._trajectory_hits_vias(traj, intermediate, steps, state_dt)
        else:
            accepted = self._trajectory_hits_poses(traj, poses, steps, state_dt)
        check_s = time.perf_counter() - t_check
        if not accepted or traj is None:
            return None, joint_s, check_s
        return (
            WaypointPlanResult(
                success=torch.ones(1, dtype=torch.bool, device=q.position.device),
                interpolated_trajectory=traj,
                segment_trajectories=[traj],
                failed_index=None,
                status="一次优化经过全部路点",
                total_time=float(result.total_time),
                solve_time=joint_s,
                planning_time=joint_s + check_s,
            ),
            joint_s,
            check_s,
        )

    def _finish_waypoint_timing(
        self,
        result: WaypointPlanResult,
        t_wall: float,
    ) -> WaypointPlanResult:
        """Record the whole waypoint call, including a failed single-solve attempt."""
        wall_s = time.perf_counter() - t_wall
        result.planning_time = wall_s
        result.total_time = wall_s
        return result

    def _polyline_seed(
        self,
        points: torch.Tensor,
        action_horizon: int,
        num_seeds: int,
        fractions: List[float],
    ) -> torch.Tensor:
        """Knot polylines timed by segment travel, shape (1, seeds, H, dof).

        ``points`` is ``(n, dof)`` or ``(branches, n, dof)``. One branch is
        repeated across seeds. Several branches are cycled so the seeds are
        not copies of a single line.
        """
        if points.ndim == 2:
            points = points.unsqueeze(0)
        n = int(points.shape[1])
        u = torch.linspace(0, 1, action_horizon, device=points.device, dtype=points.dtype)
        marks = torch.tensor(fractions, device=points.device, dtype=points.dtype)
        i0 = torch.searchsorted(marks, u, right=True) - 1
        i0 = i0.clamp(0, n - 2)
        span = (marks[i0 + 1] - marks[i0]).clamp_min(1e-8)
        alpha = ((u - marks[i0]) / span).unsqueeze(-1)
        knots = points[:, i0] * (1.0 - alpha) + points[:, i0 + 1] * alpha
        if int(knots.shape[0]) == 1:
            knots = knots.repeat(num_seeds, 1, 1)
        elif int(knots.shape[0]) < num_seeds:
            index = torch.arange(num_seeds, device=points.device) % int(knots.shape[0])
            knots = knots[index]
        elif int(knots.shape[0]) > num_seeds:
            knots = knots[:num_seeds]
        return knots.unsqueeze(0)

    def _cspace_costs(self) -> List[StateCSpaceCost]:
        """Cspace terms in the optimization objective.

        Rollouts also keep a same-named cspace term under the constraint
        manager. That term is summed into the feasibility check and must stay
        at zero unless a joint limit is violated, so waypoint error is not
        written there.
        """
        costs = []
        for rollout in self.trajopt_solver.core.get_all_rollout_instances(
            include_optimizer_rollouts=True
        ):
            manager = getattr(rollout, "cost_manager", None)
            if manager is None or not manager.has_cost("cspace"):
                continue
            cost = manager.get_cost("cspace")
            if cost is not None and hasattr(cost, "set_via_targets"):
                costs.append(cost)
        return costs

    def _arm_waypoint_costs(
        self, positions: torch.Tensor, steps: List[int]
    ) -> Optional[List[int]]:
        costs = self._cspace_costs()
        if not costs:
            return None
        horizon = int(self.trajopt_solver.horizon)
        armed: Optional[List[int]] = None
        ready = False
        try:
            for cost in costs:
                cost_horizon = getattr(cost, "_horizon", None)
                if cost_horizon is None or int(cost_horizon) != horizon:
                    return None
                got = cost.set_via_targets(positions, steps)
                if not got:
                    return None
                got_steps = list(got)
                if armed is None:
                    armed = got_steps
                elif got_steps != armed:
                    return None
            ready = True
            return armed
        except Exception as exc:
            log_warn(f"Waypoint objective setup failed: {exc!r}")
            return None
        finally:
            if not ready:
                # The failing component may have written buffers before failing.
                self._clear_waypoint_costs(costs)

    def _clear_waypoint_costs(self, costs: Optional[List[StateCSpaceCost]] = None) -> None:
        if costs is None:
            costs = self._cspace_costs()
        for cost in costs:
            cost.clear_via_targets()

    def _solution_state_dt(self, result: TrajOptSolverResult) -> Optional[float]:
        """State-step duration of the best seed, in seconds."""
        js = getattr(result, "js_solution", None)
        if js is None or js.dt is None:
            return None
        flat = js.dt.detach().reshape(-1)
        if flat.numel() == 0:
            return None
        dt = float(flat[0].item())
        if not math.isfinite(dt) or dt <= 0.0:
            return None
        return dt

    def _sample_waypoint_positions(
        self,
        traj: JointState,
        steps: List[int],
        state_dt: Optional[float],
    ) -> Optional[torch.Tensor]:
        """Evaluate output positions at exact via times, returning ``[vias, dof]``.

        Use cubic Hermite interpolation for position/velocity commands and
        linear interpolation for position-only commands. Never clamp a waypoint
        outside the output trajectory to an unrelated endpoint.
        """
        if state_dt is None or not math.isfinite(state_dt) or state_dt <= 0.0:
            return None
        if not steps or steps[0] <= 0 or any(b <= a for a, b in zip(steps, steps[1:])):
            return None
        pos = _traj_position_2d(traj)
        if pos.shape[0] < 2 or not bool(torch.isfinite(pos).all()):
            return None
        output_dt = float(self.trajopt_solver.config.interpolation_dt)
        if traj.dt is not None:
            dt = traj.dt.detach().reshape(-1)
            if dt.numel() != 1:
                return None
            output_dt = float(dt[0].item())
        if not math.isfinite(output_dt) or output_dt <= 0.0:
            return None
        last = pos.shape[0] - 1
        horizon = int(self.trajopt_solver.horizon)
        if horizon < 3 or steps[-1] >= horizon - 1:
            return None
        if (
            self.trajopt_solver.config.interpolation_type
            == TrajInterpolationType.BSPLINE_KNOTS_CUDA
        ):
            # B-spline output rounds samples per knot and retimes the entire
            # curve. Preserve spline phase using the actual output duration;
            # the optimized state_dt is no longer the output state spacing.
            sample_times = [step * last / (horizon - 1) for step in steps]
        else:
            sample_times = [step * state_dt / output_dt for step in steps]
        if any(not math.isfinite(t) or t > last + 1e-5 for t in sample_times):
            return None
        times = torch.tensor(sample_times, device=pos.device, dtype=pos.dtype).clamp(max=last)
        left = times.floor().long().clamp(max=last - 1)
        u = (times - left).unsqueeze(-1)
        p0, p1 = pos[left], pos[left + 1]
        if traj.velocity is None:
            sampled = p0 + u * (p1 - p0)
        else:
            if traj.velocity.shape != traj.position.shape:
                return None
            vel = traj.velocity.reshape_as(pos)
            if not bool(torch.isfinite(vel).all()):
                return None
            u2, u3 = u * u, u * u * u
            sampled = (
                (2 * u3 - 3 * u2 + 1) * p0
                + (u3 - 2 * u2 + u) * output_dt * vel[left]
                + (-2 * u3 + 3 * u2) * p1
                + (u3 - u2) * output_dt * vel[left + 1]
            )
        return sampled if bool(torch.isfinite(sampled).all()) else None

    def _trajectory_hits_vias(
        self,
        traj: JointState,
        intermediate: torch.Tensor,
        steps: List[int],
        state_dt: Optional[float],
    ) -> bool:
        """True when each joint waypoint is within ``waypoint_joint_tolerance``."""
        if intermediate.ndim != 2 or len(steps) != intermediate.shape[0]:
            return False
        if not bool(torch.isfinite(intermediate).all()):
            return False
        sampled = self._sample_waypoint_positions(traj, steps, state_dt)
        if sampled is None:
            return False
        active = self.kinematics.get_active_js(
            JointState.from_position(sampled, joint_names=traj.joint_names)
        )
        q = _traj_position_2d(active)
        if q.shape != intermediate.shape:
            return False
        tolerance = float(self.waypoint_joint_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            return False
        err = float((q - intermediate.to(device=q.device, dtype=q.dtype)).abs().amax().item())
        if not math.isfinite(err) or err > tolerance:
            log_warn(f"Joint waypoint missed by {err:.4f} rad, limit {tolerance:.4f} rad")
            return False
        return True

    def _trajectory_hits_poses(
        self,
        traj: JointState,
        poses: List[GoalToolPose],
        steps: List[int],
        state_dt: Optional[float],
    ) -> bool:
        """FK the output and compare each original pose at its trajectory time."""
        pos = _traj_position_2d(traj)
        n = int(pos.shape[0])
        if n < 1 or len(poses) != len(steps) + 1:
            return False
        sampled = self._sample_waypoint_positions(traj, steps, state_dt)
        if sampled is None:
            return False
        positions = torch.cat((sampled, pos[-1:]), dim=0)
        try:
            kin = self.compute_kinematics(
                JointState.from_position(positions, joint_names=traj.joint_names)
            )
            tool = kin.tool_poses
            if tool is None:
                return False
            fk_p = tool.position
            fk_q = tool.quaternion
            if fk_p.ndim == 4 and fk_p.shape[1] == 1:
                fk_p = fk_p.squeeze(1)
            if fk_q.ndim == 4 and fk_q.shape[1] == 1:
                fk_q = fk_q.squeeze(1)
            if fk_p.ndim != 3 or fk_q.ndim != 3:
                return False
            if not bool(torch.isfinite(fk_p).all() & torch.isfinite(fk_q).all()):
                return False
            pos_tol = float(self.trajopt_solver.config.position_tolerance)
            ori_tol = float(self.trajopt_solver.config.orientation_tolerance)
            if not all(math.isfinite(t) and t >= 0.0 for t in (pos_tol, ori_tol)):
                return False
            frames = list(tool.tool_frames)
            for i, goal in enumerate(poses):
                if goal.position.ndim != 5 or goal.quaternion.ndim != 5:
                    return False
                goal_p = goal.position[0, 0, :, 0, :]
                goal_q = goal.quaternion[0, 0, :, 0, :]
                if not bool(torch.isfinite(goal_p).all() & torch.isfinite(goal_q).all()):
                    return False
                for li, name in enumerate(goal.tool_frames):
                    if name not in frames:
                        log_warn(f"Waypoint pose check has no FK frame {name}")
                        return False
                    fi = frames.index(name)
                    pos_err = float(torch.linalg.norm(fk_p[i, fi] - goal_p[li]).item())
                    ori_err = float(
                        angular_distance_axis_angle(
                            goal_q[li], fk_q[i, fi]
                        ).reshape(-1)[0].item()
                    )
                    if (
                        not math.isfinite(pos_err)
                        or not math.isfinite(ori_err)
                        or pos_err > pos_tol
                        or ori_err > ori_tol
                    ):
                        log_warn(
                            f"Cartesian waypoint {i} frame {name} missed pose: "
                            f"position {pos_err:.4f} m (limit {pos_tol:.4f}), "
                            f"orientation {ori_err:.4f} rad (limit {ori_tol:.4f})"
                        )
                        return False
        except Exception as exc:
            log_warn(f"Waypoint pose check failed: {exc!r}")
            return False
        return True

    def _plan_cartesian_waypoints_segmented(
        self,
        goal_tool_poses: List[GoalToolPose],
        current_state: JointState,
        use_implicit_goal: bool = True,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
        finetune_attempts: Optional[int] = None,
    ) -> WaypointPlanResult:
        """Plan each Cartesian via as its own segment, then try a flyby spline.

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
                finetune_attempts=finetune_attempts,
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

    def _plan_joint_waypoints_segmented(
        self,
        goal_states: List[JointState],
        current_state: JointState,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
        finetune_attempts: int = 3,
    ) -> WaypointPlanResult:
        """Plan each joint via as its own segment, then try a flyby spline.

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
                finetune_attempts=finetune_attempts,
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

        if self.graph_planner is None:
            log_warn(
                "Flyby spline has no feasibility checker; "
                "using rest-to-rest concatenation"
            )
            return (
                _concat_interpolated_segments(segments),
                "flyby spline skipped; no feasibility checker",
            )

        t_check = time.perf_counter()
        interpolated, check_q = _flyby_spline_through_segments(
            segments,
            interpolation_dt=float(self.trajopt_solver.config.interpolation_dt),
            max_velocity=self.trajopt_solver.transition_model.max_velocity,
            max_acceleration=self.trajopt_solver.transition_model.max_acceleration,
            max_jerk=self.trajopt_solver.transition_model.max_jerk,
        )
        check_traj = interpolated
        if check_q is not None:
            check_traj = JointState.from_position(
                check_q, joint_names=interpolated.joint_names
            )
        free = self._joint_traj_is_collision_free(check_traj)
        self._timing_add("check", time.perf_counter() - t_check)
        if free:
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
        if traj is None or traj.position is None or self.graph_planner is None:
            return False
        pos = _traj_position_2d(traj)
        if pos.shape[0] < 1:
            return False
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
        finetune_attempts: Optional[int] = None,
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
            t_ik = time.perf_counter()
            ik_result = self.ik_solver.solve_pose(
                goal_tool_poses,
                return_seeds=num_seeds,
                current_state=current_state,
            )
            self._timing_add("ik", time.perf_counter() - t_ik)
            self.last_ik_result = ik_result
            total_time += ik_result.total_time
            solve_time += ik_result.solve_time

            success_count = torch.count_nonzero(ik_result.success)
            if success_count == 0:
                continue

            seed_config = ik_result.solution
            if success_count < num_seeds:
                good_solution = seed_config[ik_result.success][0:1, :].clone()
                # Boolean indexing returns a copy. Assigning into ``[:, :]``
                # writes that copy and leaves the failed seeds unchanged.
                seed_config[~ik_result.success] = good_solution

            seed_traj = None
            attempt_finetune = 1 if finetune_attempts is None else finetune_attempts
            finetune_dt_scale = 0.55
            if current_attempt >= enable_graph_attempt and self.graph_planner is not None:
                t_prm = time.perf_counter()
                graph_seed = self._get_graph_seed_trajectories(
                    current_state, seed_config,
                )
                prm_dt = time.perf_counter() - t_prm
                self._timing_add("prm", prm_dt)
                total_time += prm_dt
                if graph_seed is None:
                    continue
                seed_traj = graph_seed
                if finetune_attempts is None:
                    attempt_finetune = 3
                finetune_dt_scale = 0.75

            t_opt = time.perf_counter()
            trajopt_result = self.trajopt_solver.solve_pose(
                goal_tool_poses, current_state,
                seed_config=seed_config,
                seed_traj=seed_traj,
                use_implicit_goal=True,
                finetune_attempts=attempt_finetune,
                finetune_dt_scale=finetune_dt_scale,
            )
            self._timing_add("trajopt", time.perf_counter() - t_opt)
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
        finetune_attempts: Optional[int] = None,
    ) -> Optional[TrajOptSolverResult]:
        """Goalset planning: IK + TrajOpt, no graph seeding."""
        self.last_ik_result = None
        trajopt_result = None
        total_time = 0.0
        for _ in range(max_attempts):
            t_ik = time.perf_counter()
            ik_result = self.ik_solver.solve_pose(
                goal_tool_poses,
                return_seeds=self.trajopt_solver.config.num_seeds,
                current_state=current_state,
            )
            self._timing_add("ik", time.perf_counter() - t_ik)
            self.last_ik_result = ik_result
            total_time += float(ik_result.total_time)
            if torch.count_nonzero(ik_result.success) == 0:
                return None

            t_opt = time.perf_counter()
            trajopt_result = self.trajopt_solver.solve_pose(
                goal_tool_poses, current_state,
                seed_config=ik_result.solution,
                use_implicit_goal=use_implicit_goal,
                finetune_attempts=1 if finetune_attempts is None else finetune_attempts,
            )
            self._timing_add("trajopt", time.perf_counter() - t_opt)
            total_time += float(trajopt_result.total_time)
            if torch.count_nonzero(trajopt_result.success) > 0:
                break

        if trajopt_result is not None:
            trajopt_result.total_time = total_time
        return trajopt_result

    def plan_cspace(
        self,
        goal_state: JointState,
        current_state: JointState,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
        finetune_attempts: int = 3,
    ) -> Optional[TrajOptSolverResult]:
        """Plan a collision-free trajectory to a joint configuration.

        Args:
            goal_state: Target joint configuration.
            current_state: Initial joint state.
            max_attempts: Maximum planning attempts.
            enable_graph_attempt: Attempt at which to start graph seeding.
            finetune_attempts: Extra time-optimal passes after the first solve.
                ``3`` runs four L-BFGS solves per attempt. ``1`` runs two.
                ``0`` keeps the first feasible trajectory.

        Returns:
            TrajOptSolverResult, or None if planning failed.
        """
        if current_state.ndim > 2:
            log_and_raise(f"current_state must be a 2D tensor, got shape: {current_state.shape}")
        if goal_state.ndim > 2:
            log_and_raise(f"goal_state must be a 2D tensor, got shape: {goal_state.shape}")

        self._timing_enter()
        try:
            return self._plan_cspace_timed(
                goal_state,
                current_state,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
                finetune_attempts=finetune_attempts,
            )
        finally:
            self._timing_leave()

    def _plan_cspace_timed(
        self,
        goal_state: JointState,
        current_state: JointState,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
        finetune_attempts: int = 3,
    ) -> Optional[TrajOptSolverResult]:
        trajopt_result = None
        total_time = 0.0
        solve_time = 0.0
        og_current_state = current_state.clone()
        num_seeds = self.trajopt_solver.config.num_seeds

        for current_attempt in range(max_attempts):
            current_state = og_current_state.clone()
            seed_traj = None
            finetune_dt_scale = 0.75

            if current_attempt >= enable_graph_attempt and self.graph_planner is not None:
                goal_configs = goal_state.position.view(1, 1, -1).repeat(1, num_seeds, 1)
                t_prm = time.perf_counter()
                graph_seed = self._get_graph_seed_trajectories(
                    current_state, goal_configs,
                )
                prm_dt = time.perf_counter() - t_prm
                self._timing_add("prm", prm_dt)
                total_time += prm_dt
                if graph_seed is None:
                    continue
                seed_traj = graph_seed

            t_opt = time.perf_counter()
            trajopt_result = self.trajopt_solver.solve_cspace(
                goal_state, current_state, seed_traj=seed_traj,
                finetune_attempts=finetune_attempts,
                finetune_dt_scale=finetune_dt_scale,
            )
            self._timing_add("trajopt", time.perf_counter() - t_opt)
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
