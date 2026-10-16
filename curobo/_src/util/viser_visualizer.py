# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Third Party
import numpy as np
import torch
import trimesh
import yourdfpy

from curobo._src.geom.types import SceneCfg, Sphere

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.robot.loader.util import load_robot_yaml
from curobo._src.state.state_joint import JointState
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise
from curobo._src.util_file import join_path

try:
    import viser
    import viser.transforms as vtf
    from viser.extras import ViserUrdf
except ImportError:
    raise ImportError("Viser not installed. Install with: pip install viser")


def _fmt_yaml_number(value: float) -> str:
    number = round(float(value), 4)
    if abs(number) < 5e-5:
        number = 0.0
    text = f"{number:.4f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0.0"
    if "." not in text:
        text += ".0"
    return text


def _collision_spheres_link_span(lines: List[str], link: str) -> Optional[tuple]:
    """Return (link_indent, link_line, section_end) for a collision_spheres entry."""
    spheres_i = None
    spheres_indent = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if stripped.startswith("collision_spheres:") and indent < 8:
            spheres_i = i
            spheres_indent = indent
            break
    if spheres_i is None:
        return None
    link_indent = spheres_indent + 2
    link_i = None
    section_end = len(lines)
    for i in range(spheres_i + 1, len(lines)):
        stripped = lines[i].lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[i]) - len(stripped)
        if indent <= spheres_indent:
            section_end = i
            break
        if lines[i] == f"{' ' * link_indent}{link}:":
            link_i = i
            break
    if link_i is None:
        return None
    for i in range(link_i + 1, section_end):
        stripped = lines[i].lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[i]) - len(stripped)
        if indent <= link_indent and not stripped.startswith("-"):
            section_end = i
            break
    return link_indent, link_i, section_end


def _sphere_block(lines: List[str], link: str, local: int) -> Optional[tuple]:
    """Return (start, end) line indices of the uncommented ``local``-th sphere."""
    span = _collision_spheres_link_span(lines, link)
    if span is None:
        return None
    link_indent, link_i, section_end = span
    seen = 0
    for i in range(link_i + 1, section_end):
        stripped = lines[i].lstrip(" ")
        if not stripped.startswith("- center:"):
            continue
        if seen != local:
            seen += 1
            continue
        end = i + 1
        for j in range(i + 1, section_end):
            nxt = lines[j].lstrip(" ")
            if not nxt:
                break
            if nxt.startswith("#") or nxt.startswith("- center:"):
                break
            indent = len(lines[j]) - len(nxt)
            if indent <= link_indent and not nxt.startswith("-"):
                break
            end = j + 1
        return i, end
    return None


def collision_sphere_yaml_line(path: str, link: str, local: int) -> Optional[int]:
    """1-based line of the ``local``-th ``center`` under ``link`` in collision_spheres."""
    lines = Path(path).read_text().splitlines()
    block = _sphere_block(lines, link, local)
    if block is None:
        return None
    return block[0] + 1


def replace_collision_sphere(
    path: str, link: str, local: int, center: Sequence[float], radius: float
) -> Optional[int]:
    """Rewrite one sphere's center and radius. Returns the 1-based center line."""
    file_path = Path(path)
    lines = file_path.read_text().splitlines()
    block = _sphere_block(lines, link, local)
    if block is None:
        return None
    start, end = block
    components = [
        i
        for i in range(start + 1, end)
        if lines[i].lstrip().startswith("- ")
    ]
    if len(components) < 3:
        return None
    for index, value in zip(components[:3], center):
        indent = len(lines[index]) - len(lines[index].lstrip())
        lines[index] = f"{' ' * indent}- {_fmt_yaml_number(value)}"
    for index in range(start + 1, end):
        stripped = lines[index].lstrip()
        if stripped.startswith("radius:"):
            indent = len(lines[index]) - len(stripped)
            lines[index] = f"{' ' * indent}radius: {_fmt_yaml_number(radius)}"
            break
    file_path.write_text("\n".join(lines) + "\n")
    return start + 1


def duplicate_collision_sphere(path: str, link: str, local: int) -> Optional[int]:
    """Insert a copy of one sphere directly under it. Returns the new local index."""
    file_path = Path(path)
    lines = file_path.read_text().splitlines()
    block = _sphere_block(lines, link, local)
    if block is None:
        return None
    start, end = block
    lines[end:end] = lines[start:end]
    file_path.write_text("\n".join(lines) + "\n")
    return local + 1


def delete_collision_sphere(path: str, link: str, local: int) -> Optional[List[str]]:
    """Remove one sphere block. Returns the removed lines for undo."""
    file_path = Path(path)
    lines = file_path.read_text().splitlines()
    block = _sphere_block(lines, link, local)
    if block is None:
        return None
    start, end = block
    removed = lines[start:end]
    del lines[start:end]
    file_path.write_text("\n".join(lines) + "\n")
    return removed


def restore_collision_sphere(
    path: str, link: str, local: int, block_lines: Sequence[str]
) -> Optional[int]:
    """Put a previously deleted sphere block back at ``local``. Returns that index."""
    file_path = Path(path)
    lines = file_path.read_text().splitlines()
    span = _collision_spheres_link_span(lines, link)
    if span is None:
        return None
    _link_indent, link_i, section_end = span
    insert_at = section_end
    seen = 0
    for i in range(link_i + 1, section_end):
        if not lines[i].lstrip().startswith("- center:"):
            continue
        if seen == local:
            insert_at = i
            break
        seen += 1
    lines[insert_at:insert_at] = list(block_lines)
    file_path.write_text("\n".join(lines) + "\n")
    return local


class ViserVisualizer:
    def __init__(
        self,
        content_path: Optional[ContentPath] = None,
        device_cfg: DeviceCfg = DeviceCfg(
            device=torch.device("cuda"), dtype=torch.float32
        ),
        add_robot_to_scene: bool = False,
        connect_ip: str = "0.0.0.0",
        connect_port: int = 8080,
        initialize_viser: bool = True,
        add_control_frames: bool = True,
        visualize_robot_spheres: bool = False,
        visualize_collision_meshes: bool = False,
    ):
        self._visualize_robot_spheres = visualize_robot_spheres

        self._server = viser.ViserServer(host=connect_ip, port=connect_port)

        self._server.scene.add_grid("/ground_plane", width=20, height=20)
        robot_data = None
        visualize_frames = []
        self._robot_model = None
        self._mesh_root = None
        self._sphere_info = None
        self._selected_sphere_idx: Optional[int] = None
        self._sphere_world_xyz = np.zeros((0, 3), dtype=np.float64)
        self._sphere_radius = np.zeros((0,), dtype=np.float64)
        self._sphere_link = np.array([], dtype=object)
        self._sphere_local_idx = np.zeros((0,), dtype=np.int32)
        self._sphere_yaml_xyz = np.zeros((0, 3), dtype=np.float64)
        self._sphere_yaml_radius = np.zeros((0,), dtype=np.float64)
        self._sphere_jump = None
        self._sphere_gizmo = None
        self._sphere_edit_x = None
        self._sphere_edit_y = None
        self._sphere_edit_z = None
        self._sphere_edit_r = None
        self._sphere_copy = None
        self._sphere_delete = None
        self._sphere_undo = None
        self._deleted_spheres: List[dict] = []
        self._sphere_gui_lock = False
        self._sphere_dragging = False
        self._selected_link: Optional[str] = None
        self._selected_local: Optional[int] = None
        self._selected_yaml_line: Optional[int] = None
        self._robot_yaml_path: Optional[str] = None
        self._last_joint_state: Optional[JointState] = None
        # load robot into viser and set it at default joint configuration.
        if add_robot_to_scene and content_path is None:
            log_and_raise("Content path is required to add robot to scene")
        if content_path is not None:
            if not isinstance(content_path, Dict):
                self._robot_yaml_path = content_path.get_robot_configuration_path()
                robot_data = load_robot_yaml(content_path)
            else:
                robot_data = content_path
            if "robot_cfg" not in robot_data:
                robot_data = {"robot_cfg": robot_data}
            robot_data["robot_cfg"]["kinematics"]["load_tool_frames_with_mesh"] = True
            visualize_frames = robot_data["robot_cfg"]["kinematics"]["tool_frames"].copy()
            self._robot_model = KinematicsCfg.from_data_dict(
                robot_data["robot_cfg"]["kinematics"],
                device_cfg=device_cfg,
            )

            self._kinematics = Kinematics(self._robot_model)

            self._mesh_root = self._robot_model.generator_config.asset_root_path
            extra_links = self._robot_model.generator_config.extra_links
            if extra_links:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".urdf", delete=False, mode="w", prefix="curobo_viz_"
                )
                kin_config = self._kinematics.config.kinematics_config
                kin_config.export_to_urdf(
                    output_path=tmp.name,
                    kinematics_parser=self._robot_model.kinematics_parser,
                )
                urdf_path = tmp.name
            else:
                urdf_path = self._robot_model.generator_config.urdf_path
            self._urdf = yourdfpy.URDF.load(
                urdf_path,
                load_meshes=True,
                build_scene_graph=True,
                filename_handler=self._file_name_handler,
                build_collision_scene_graph=visualize_collision_meshes,
                load_collision_meshes=visualize_collision_meshes,
            )
            self._viser_urdf = ViserUrdf(
                self._server,
                self._urdf,
                root_node_name="/" + self._robot_model.generator_config.base_link,
                load_collision_meshes=visualize_collision_meshes,
            )
            self._viser_joint_names = list(self._viser_urdf.get_actuated_joint_names())
            self._vis_frames = visualize_frames.copy()

            self.reset_robot()
            self._robot_spheres = []
            if visualize_robot_spheres:
                self._ensure_sphere_inspect_gui()
            # add robot spheres:
            self._control_frames = {}
            if add_control_frames:
                # get pose of frames at current joint state:
                joint_state = self._kinematics.default_joint_state
                kin_state = self._kinematics.compute_kinematics(joint_state)
                if kin_state.tool_poses is None:
                    log_and_raise("KinematicsState.tool_poses is None; check tool_frames / kinematics config.")
                for frame_name in self._vis_frames:
                    frame_pose = kin_state.tool_poses[frame_name]
                    self._control_frames[frame_name] = self.add_control_frame(
                        "/target_" + frame_name, frame_pose, scale=0.1
                    )

    def _file_name_handler(self, fname: str) -> str:
        # This will also remove package:// from the file name:
        fname = fname.replace("package://", "")
        return join_path(self._mesh_root, fname)

    @property
    def joint_names(self) -> List[str]:
        return self._viser_joint_names

    def _ensure_sphere_inspect_gui(self) -> None:
        if self._sphere_info is not None:
            return
        with self._server.gui.add_folder("Clicked sphere"):
            self._sphere_info = self._server.gui.add_markdown(
                "_点一颗碰撞球。拖动坐标轴改位置，滑条改半径。删除可以撤回，改动会写回 YAML。_"
            )
            self._sphere_edit_x = self._server.gui.add_number(
                "x (连杆)", 0.0, step=0.001, disabled=True
            )
            self._sphere_edit_y = self._server.gui.add_number(
                "y (连杆)", 0.0, step=0.001, disabled=True
            )
            self._sphere_edit_z = self._server.gui.add_number(
                "z (连杆)", 0.0, step=0.001, disabled=True
            )
            self._sphere_edit_r = self._server.gui.add_slider(
                "半径 (m)",
                min=0.005,
                max=0.5,
                step=0.001,
                initial_value=0.08,
                disabled=True,
            )
            self._sphere_radius_down = self._server.gui.add_button(
                "半径 −5 mm", disabled=True
            )
            self._sphere_radius_up = self._server.gui.add_button(
                "半径 +5 mm", disabled=True
            )
            self._sphere_copy = self._server.gui.add_button("原地复制", disabled=True)
            self._sphere_delete = self._server.gui.add_button("删除", disabled=True)
            self._sphere_undo = self._server.gui.add_button("撤回删除", disabled=True)
            self._sphere_jump = self._server.gui.add_button(
                "跳转到 YAML",
                disabled=True,
                hint="在 Cursor 中打开这颗球对应的 collision_spheres 条目",
            )
            self._sphere_edit_x.on_update(lambda _event: self._on_sphere_widget_edit())
            self._sphere_edit_y.on_update(lambda _event: self._on_sphere_widget_edit())
            self._sphere_edit_z.on_update(lambda _event: self._on_sphere_widget_edit())
            self._sphere_edit_r.on_update(lambda _event: self._on_sphere_widget_edit())
            self._sphere_radius_down.on_click(lambda _event: self._bump_selected_radius(-0.005))
            self._sphere_radius_up.on_click(lambda _event: self._bump_selected_radius(0.005))
            self._sphere_copy.on_click(lambda _event: self._duplicate_selected_sphere())
            self._sphere_delete.on_click(lambda _event: self._delete_selected_sphere())
            self._sphere_undo.on_click(lambda _event: self._undo_sphere_delete())
            self._sphere_jump.on_click(lambda _event: self._jump_to_selected_sphere())
        self._sphere_gizmo = self._server.scene.add_transform_controls(
            "/sphere_edit_gizmo",
            scale=0.15,
            disable_rotations=True,
            depth_test=False,
            visible=False,
        )
        self._sphere_gizmo.on_update(self._on_sphere_gizmo)

        @self._server.scene.on_click()
        def _on_scene_click(event) -> None:
            idx = self._pick_sphere_from_ray(event.ray_origin, event.ray_direction)
            if idx is None:
                return
            self._show_clicked_sphere(idx)

    def _sphere_inspect_markdown(self, idx: int) -> str:
        xyz = self._sphere_world_xyz[idx]
        r = float(self._sphere_radius[idx])
        link = str(self._sphere_link[idx])
        local = int(self._sphere_local_idx[idx])
        yxyz = self._sphere_yaml_xyz[idx]
        yr = float(self._sphere_yaml_radius[idx])
        buf = r - yr
        line = self._selected_yaml_line
        where = ""
        if self._robot_yaml_path and line is not None:
            where = f"\n\n`{Path(self._robot_yaml_path).name}:{line}`"
        return (
            f"**`{link}`** · sphere `{local}`{where}\n\n"
            f"world (what you clicked, after FK)\n"
            f"- center: `[{xyz[0]:.5f}, {xyz[1]:.5f}, {xyz[2]:.5f}]`\n"
            f"- radius: `{r:.5f}` m ({r*1000:.1f} mm)\n\n"
            f"yaml (link frame, as in the robot config)\n"
            f"- center: `[{yxyz[0]:.5f}, {yxyz[1]:.5f}, {yxyz[2]:.5f}]`\n"
            f"- radius: `{yr:.5f}` m ({yr*1000:.1f} mm)\n"
            + (f"- buffer: `{buf:.5f}` m ({buf*1000:.1f} mm)\n" if abs(buf) > 1e-6 else "")
        )

    def _pick_sphere_from_ray(
        self, origin: Sequence[float], direction: Sequence[float]
    ) -> Optional[int]:
        """Nearest sphere the click ray intersects, or None."""
        if self._sphere_world_xyz.shape[0] == 0:
            return None
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        n = np.linalg.norm(d)
        if n < 1e-12:
            return None
        d = d / n
        oc = o - self._sphere_world_xyz
        b = 2.0 * np.einsum("ij,j->i", oc, d)
        c = np.einsum("ij,ij->i", oc, oc) - self._sphere_radius**2
        disc = b * b - 4.0 * c
        hit = disc >= 0.0
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t_near = 0.5 * (-b - sqrt_disc)
        t_far = 0.5 * (-b + sqrt_disc)
        t = np.where(t_near > 1e-4, t_near, t_far)
        valid = hit & (t > 1e-4)
        if not np.any(valid):
            return None
        return int(np.argmin(np.where(valid, t, np.inf)))

    def _show_clicked_sphere(self, idx: int) -> None:
        self._selected_sphere_idx = idx
        link = str(self._sphere_link[idx])
        local = int(self._sphere_local_idx[idx])
        line = None
        if self._robot_yaml_path:
            line = collision_sphere_yaml_line(self._robot_yaml_path, link, local)
        self._selected_yaml_line = line
        self._selected_link = link
        self._selected_local = local
        yxyz = self._sphere_yaml_xyz[idx]
        yr = float(self._sphere_yaml_radius[idx])
        editable = line is not None or self._robot_yaml_path is None
        self._set_sphere_editor_enabled(True)
        self._write_sphere_editor_values(yxyz, yr)
        if self._sphere_copy is not None:
            self._sphere_copy.disabled = not editable
        if self._sphere_delete is not None:
            self._sphere_delete.disabled = not editable
        if self._sphere_jump is not None:
            if line is None or not self._robot_yaml_path:
                self._sphere_jump.disabled = True
                self._sphere_jump.label = "跳转到 YAML"
            else:
                self._sphere_jump.disabled = False
                name = Path(self._robot_yaml_path).name
                self._sphere_jump.label = f"跳转到 {name}:{line}"
        self._highlight_selected_sphere()
        text = self._sphere_inspect_markdown(idx)
        if self._sphere_info is not None:
            self._sphere_info.content = text
        xyz = self._sphere_world_xyz[idx]
        r = float(self._sphere_radius[idx])
        print(
            f"[sphere] {self._sphere_link[idx]}[{int(self._sphere_local_idx[idx])}]  "
            f"world center=[{xyz[0]:.5f}, {xyz[1]:.5f}, {xyz[2]:.5f}]  r={r:.5f} m  |  "
            f"yaml center=[{yxyz[0]:.5f}, {yxyz[1]:.5f}, {yxyz[2]:.5f}]  r={yr:.5f} m"
            + (
                f"  |  {self._robot_yaml_path}:{self._selected_yaml_line}"
                if self._robot_yaml_path and self._selected_yaml_line is not None
                else ""
            )
        )
        self._place_sphere_gizmo()

    def _set_sphere_editor_enabled(self, enabled: bool) -> None:
        for widget in (
            self._sphere_edit_x,
            self._sphere_edit_y,
            self._sphere_edit_z,
            self._sphere_edit_r,
            self._sphere_radius_down,
            self._sphere_radius_up,
            self._sphere_copy,
            self._sphere_delete,
        ):
            if widget is not None:
                widget.disabled = not enabled

    def _write_sphere_editor_values(self, center: Sequence[float], radius: float) -> None:
        if self._sphere_edit_x is None:
            return
        radius = float(np.clip(radius, 0.005, 0.5))
        self._sphere_gui_lock = True
        self._sphere_edit_x.value = float(center[0])
        self._sphere_edit_y.value = float(center[1])
        self._sphere_edit_z.value = float(center[2])
        self._sphere_edit_r.value = radius
        self._sphere_gui_lock = False

    def _sphere_buffer(self, link: str) -> float:
        gen = self._kinematics.config.generator_config
        buffer = gen.collision_sphere_buffer if gen is not None else 0.0
        if isinstance(buffer, dict):
            return float(buffer.get(link, 0.0))
        return float(buffer or 0.0)

    def _link_rotation(self, link: str) -> np.ndarray:
        transform = np.asarray(self._urdf.get_transform(link), dtype=np.float64)
        return transform[:3, :3]

    def _world_to_link(self, link: str, world_xyz: Sequence[float]) -> np.ndarray:
        transform = np.asarray(self._urdf.get_transform(link), dtype=np.float64)
        rotation = transform[:3, :3]
        origin = transform[:3, 3]
        return rotation.T @ (np.asarray(world_xyz, dtype=np.float64) - origin)

    def _place_sphere_gizmo(self) -> None:
        gizmo = self._sphere_gizmo
        idx = self._selected_sphere_idx
        if gizmo is None or idx is None or idx >= len(self._sphere_world_xyz):
            if gizmo is not None:
                gizmo.visible = False
            return
        link = str(self._sphere_link[idx])
        world = self._sphere_world_xyz[idx]
        rotation = self._link_rotation(link)
        self._sphere_gui_lock = True
        gizmo.position = (float(world[0]), float(world[1]), float(world[2]))
        gizmo.wxyz = tuple(float(v) for v in vtf.SO3.from_matrix(rotation).wxyz)
        gizmo.visible = True
        self._sphere_gui_lock = False

    def _on_sphere_widget_edit(self) -> None:
        if (
            self._sphere_gui_lock
            or self._sphere_dragging
            or self._selected_link is None
            or self._selected_local is None
        ):
            return
        center = [
            float(self._sphere_edit_x.value),
            float(self._sphere_edit_y.value),
            float(self._sphere_edit_z.value),
        ]
        self._commit_selected_sphere(center, float(self._sphere_edit_r.value), write_yaml=True)

    def _bump_selected_radius(self, delta: float) -> None:
        if self._sphere_edit_r is None or self._selected_link is None:
            return
        radius = float(np.clip(float(self._sphere_edit_r.value) + delta, 0.005, 0.5))
        self._write_sphere_editor_values(
            [
                float(self._sphere_edit_x.value),
                float(self._sphere_edit_y.value),
                float(self._sphere_edit_z.value),
            ],
            radius,
        )
        self._on_sphere_widget_edit()

    def _on_sphere_gizmo(self, event) -> None:
        if self._sphere_gui_lock or self._selected_link is None or self._selected_local is None:
            return
        if event.phase == "start":
            self._sphere_dragging = True
        world = np.asarray(event.target.position, dtype=np.float64)
        center = self._world_to_link(self._selected_link, world)
        radius = float(self._sphere_edit_r.value) if self._sphere_edit_r is not None else 0.08
        write = event.phase == "end"
        self._commit_selected_sphere(center, radius, write_yaml=write, move_gizmo=False)
        self._write_sphere_editor_values(center, radius)
        if event.phase == "end":
            self._sphere_dragging = False

    def _commit_selected_sphere(
        self,
        center: Sequence[float],
        radius: float,
        write_yaml: bool,
        move_gizmo: bool = True,
    ) -> None:
        link = self._selected_link
        local = self._selected_local
        if link is None or local is None:
            return
        center_l = [round(float(v), 4) for v in center]
        radius = round(float(radius), 4)
        gen = self._kinematics.config.generator_config
        spheres = None if gen is None else gen.collision_spheres
        if not spheres or link not in spheres or local >= len(spheres[link]):
            print(f"[sphere] {link}[{local}] is not an editable collision sphere")
            return
        spheres[link][local]["center"] = center_l
        spheres[link][local]["radius"] = radius
        kc = self._kinematics.config.kinematics_config
        padded = radius + self._sphere_buffer(link)
        kc.update_link_spheres(
            link,
            torch.tensor(
                [[*center_l, padded]],
                device=kc.link_spheres.device,
                dtype=kc.link_spheres.dtype,
            ),
            start_sph_idx=local,
        )
        if write_yaml and self._robot_yaml_path:
            line = replace_collision_sphere(
                self._robot_yaml_path, link, local, center_l, radius
            )
            self._selected_yaml_line = line
            if self._sphere_jump is not None and line is not None:
                name = Path(self._robot_yaml_path).name
                self._sphere_jump.disabled = False
                self._sphere_jump.label = f"跳转到 {name}:{line}"
            print(
                f"[sphere] wrote {link}[{local}] center={center_l} r={radius} "
                f"-> {self._robot_yaml_path}:{line}"
            )
        self._redraw_selected_sphere(move_gizmo=move_gizmo)

    def _redraw_selected_sphere(self, move_gizmo: bool) -> None:
        if self._last_joint_state is None:
            return
        link = self._selected_link
        local = self._selected_local
        self.update_robot_spheres(self._last_joint_state)
        idx = self._display_index(link, local)
        if idx is None:
            return
        self._selected_sphere_idx = idx
        self._highlight_selected_sphere()
        if self._sphere_info is not None:
            self._sphere_info.content = self._sphere_inspect_markdown(idx)
        if move_gizmo and not self._sphere_dragging:
            self._place_sphere_gizmo()

    def _display_index(self, link: Optional[str], local: Optional[int]) -> Optional[int]:
        if link is None or local is None:
            return None
        for i, (name, loc) in enumerate(zip(self._sphere_link, self._sphere_local_idx)):
            if str(name) == link and int(loc) == int(local):
                return i
        return None

    def _duplicate_selected_sphere(self) -> None:
        link = self._selected_link
        local = self._selected_local
        if link is None or local is None:
            return
        gen = self._kinematics.config.generator_config
        spheres = None if gen is None else gen.collision_spheres
        if not spheres or link not in spheres or local >= len(spheres[link]):
            return
        source = spheres[link][local]
        copy = {"center": list(source["center"]), "radius": source["radius"]}
        if self._robot_yaml_path:
            new_local = duplicate_collision_sphere(self._robot_yaml_path, link, local)
            if new_local is None:
                print(f"[sphere] could not duplicate {link}[{local}] in YAML")
                return
        else:
            new_local = local + 1
        spheres[link].insert(new_local, copy)
        self._insert_kinematics_sphere(link, local, copy)
        self._selected_link = link
        self._selected_local = new_local
        if self._last_joint_state is not None:
            self.update_robot_spheres(self._last_joint_state)
        idx = self._display_index(link, new_local)
        if idx is not None:
            self._show_clicked_sphere(idx)
        print(f"[sphere] duplicated {link}[{local}] -> [{new_local}]")

    def _insert_kinematics_sphere(self, link: str, local: int, spec: dict) -> None:
        kc = self._kinematics.config.kinematics_config
        indices = kc.get_sphere_index_from_link_name(link)
        insert_at = int(indices[local].item()) + 1
        padded = float(spec["radius"]) + self._sphere_buffer(link)
        row = torch.tensor(
            [[*spec["center"], padded]],
            device=kc.link_spheres.device,
            dtype=kc.link_spheres.dtype,
        ).view(1, 1, 4)
        row = row.expand(kc.link_spheres.shape[0], 1, 4)
        kc.link_spheres = torch.cat(
            (kc.link_spheres[:, :insert_at, :], row, kc.link_spheres[:, insert_at:, :]),
            dim=1,
        ).contiguous()
        link_idx = kc.link_sphere_idx_map[insert_at - 1].view(1)
        kc.link_sphere_idx_map = torch.cat(
            (
                kc.link_sphere_idx_map[:insert_at],
                link_idx,
                kc.link_sphere_idx_map[insert_at:],
            )
        ).contiguous()
        self._kinematics.update_batch_size(
            self._kinematics._batch, self._kinematics._horizon, reset_buffers=True
        )

    def _kinematics_insert_index(self, link: str, local: int) -> int:
        """Global tensor row where ``link``'s ``local`` sphere should be inserted."""
        kc = self._kinematics.config.kinematics_config
        gen = self._kinematics.config.generator_config
        order = list(gen.collision_link_names or []) if gen is not None else []
        before = 0
        found = False
        for name in order:
            if name == link:
                found = True
                break
            before += int(kc.get_sphere_index_from_link_name(name).numel())
        if not found:
            return int(kc.link_spheres.shape[1])
        return before + int(local)

    def _insert_kinematics_sphere_at(self, link: str, local: int, spec: dict) -> None:
        kc = self._kinematics.config.kinematics_config
        insert_at = self._kinematics_insert_index(link, local)
        padded = float(spec["radius"]) + self._sphere_buffer(link)
        row = torch.tensor(
            [[float(spec["center"][0]), float(spec["center"][1]), float(spec["center"][2]), padded]],
            device=kc.link_spheres.device,
            dtype=kc.link_spheres.dtype,
        ).view(1, 1, 4)
        row = row.expand(kc.link_spheres.shape[0], 1, 4)
        link_id = kc.link_name_to_idx_map[link]
        link_idx = torch.tensor(
            [link_id], device=kc.link_sphere_idx_map.device, dtype=kc.link_sphere_idx_map.dtype
        )
        kc.link_spheres = torch.cat(
            (kc.link_spheres[:, :insert_at, :], row, kc.link_spheres[:, insert_at:, :]),
            dim=1,
        ).contiguous()
        kc.link_sphere_idx_map = torch.cat(
            (
                kc.link_sphere_idx_map[:insert_at],
                link_idx,
                kc.link_sphere_idx_map[insert_at:],
            )
        ).contiguous()
        self._kinematics.update_batch_size(
            self._kinematics._batch, self._kinematics._horizon, reset_buffers=True
        )

    def _remove_kinematics_sphere(self, link: str, local: int) -> None:
        kc = self._kinematics.config.kinematics_config
        remove_at = int(kc.get_sphere_index_from_link_name(link)[local].item())
        kc.link_spheres = torch.cat(
            (kc.link_spheres[:, :remove_at, :], kc.link_spheres[:, remove_at + 1 :, :]),
            dim=1,
        ).contiguous()
        kc.link_sphere_idx_map = torch.cat(
            (kc.link_sphere_idx_map[:remove_at], kc.link_sphere_idx_map[remove_at + 1 :])
        ).contiguous()
        self._kinematics.update_batch_size(
            self._kinematics._batch, self._kinematics._horizon, reset_buffers=True
        )

    def _refresh_undo_button(self) -> None:
        if self._sphere_undo is None:
            return
        count = len(self._deleted_spheres)
        self._sphere_undo.disabled = count == 0
        self._sphere_undo.label = "撤回删除" if count == 0 else f"撤回删除 ({count})"

    def _clear_sphere_selection(self) -> None:
        self._selected_sphere_idx = None
        self._selected_link = None
        self._selected_local = None
        self._selected_yaml_line = None
        self._set_sphere_editor_enabled(False)
        if self._sphere_jump is not None:
            self._sphere_jump.disabled = True
            self._sphere_jump.label = "跳转到 YAML"
        if self._sphere_gizmo is not None:
            self._sphere_gizmo.visible = False
        if self._sphere_info is not None:
            self._sphere_info.content = "_这颗球已删除。用「撤回删除」可以放回去。_"
        self._highlight_selected_sphere()

    def _delete_selected_sphere(self) -> None:
        link = self._selected_link
        local = self._selected_local
        if link is None or local is None:
            return
        gen = self._kinematics.config.generator_config
        spheres = None if gen is None else gen.collision_spheres
        if not spheres or link not in spheres or local >= len(spheres[link]):
            return
        spec = spheres[link][local]
        removed_lines = None
        if self._robot_yaml_path:
            removed_lines = delete_collision_sphere(self._robot_yaml_path, link, local)
            if removed_lines is None:
                print(f"[sphere] could not delete {link}[{local}] in YAML")
                return
        self._deleted_spheres.append(
            {
                "link": link,
                "local": int(local),
                "spec": {"center": list(spec["center"]), "radius": spec["radius"]},
                "yaml_lines": removed_lines,
            }
        )
        del spheres[link][local]
        self._remove_kinematics_sphere(link, local)
        self._refresh_undo_button()
        if self._last_joint_state is not None:
            self.update_robot_spheres(self._last_joint_state)
        remaining = len(spheres[link])
        if remaining == 0:
            self._clear_sphere_selection()
        else:
            next_local = local if local < remaining else local - 1
            idx = self._display_index(link, next_local)
            if idx is None:
                self._clear_sphere_selection()
            else:
                self._show_clicked_sphere(idx)
        print(f"[sphere] deleted {link}[{local}]")

    def _undo_sphere_delete(self) -> None:
        if not self._deleted_spheres:
            return
        item = self._deleted_spheres.pop()
        link = item["link"]
        local = int(item["local"])
        spec = item["spec"]
        if self._robot_yaml_path:
            if item["yaml_lines"] is None:
                self._deleted_spheres.append(item)
                return
            restored = restore_collision_sphere(
                self._robot_yaml_path, link, local, item["yaml_lines"]
            )
            if restored is None:
                self._deleted_spheres.append(item)
                print(f"[sphere] could not restore {link}[{local}] in YAML")
                return
        gen = self._kinematics.config.generator_config
        spheres = None if gen is None else gen.collision_spheres
        if spheres is None or link not in spheres:
            self._deleted_spheres.append(item)
            print(f"[sphere] could not restore {link}[{local}]")
            return
        spheres[link].insert(local, spec)
        self._insert_kinematics_sphere_at(link, local, spec)
        self._refresh_undo_button()
        if self._last_joint_state is not None:
            self.update_robot_spheres(self._last_joint_state)
        idx = self._display_index(link, local)
        if idx is not None:
            self._show_clicked_sphere(idx)
        print(f"[sphere] restored {link}[{local}]")

    def _jump_to_selected_sphere(self) -> None:
        path = self._robot_yaml_path
        line = self._selected_yaml_line
        if not path or line is None:
            return
        cursor = shutil.which("cursor")
        if cursor is None:
            print("[sphere] cursor CLI not found; cannot open the YAML")
            return
        subprocess.Popen(
            [cursor, "-r", "-g", f"{path}:{line}:1"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[sphere] opened {path}:{line}")

    def _highlight_selected_sphere(self) -> None:
        handle = getattr(self, "_batched_spheres_handle", None)
        n = self._sphere_world_xyz.shape[0]
        if handle is None or n == 0:
            return
        colors = np.repeat(np.array([[90, 200, 255]], dtype=np.uint8), n, axis=0)
        idx = self._selected_sphere_idx
        if idx is not None and 0 <= idx < n:
            colors[idx] = (255, 70, 40)
        handle.batched_colors = colors

    def update_robot_spheres(self, joint_state: JointState):
        self._robot_spheres = []
        # reindex:
        joint_state = joint_state.clone()
        joint_state = self._kinematics.get_active_js(joint_state)
        spheres = self._kinematics.get_robot_as_spheres(
            joint_state.position.contiguous().view(1, -1), filter_valid=True
        )[0]
        if not spheres:
            self._sphere_world_xyz = np.zeros((0, 3), dtype=np.float64)
            self._sphere_radius = np.zeros((0,), dtype=np.float64)
            self._sphere_link = np.array([], dtype=object)
            self._sphere_local_idx = np.zeros((0,), dtype=np.int32)
            self._sphere_yaml_xyz = np.zeros((0, 3), dtype=np.float64)
            self._sphere_yaml_radius = np.zeros((0,), dtype=np.float64)
            self._selected_sphere_idx = None
            return
        self._cache_sphere_inspect_data(spheres)
        self.add_batched_spheres(spheres)

    def _cache_sphere_inspect_data(self, spheres: List[Sphere]) -> None:
        """Keep world/yaml params aligned with the spheres the viewer draws."""
        self._sphere_world_xyz = np.asarray(
            [s.pose[:3] for s in spheres], dtype=np.float64
        ).reshape(-1, 3)
        self._sphere_radius = np.asarray([s.radius for s in spheres], dtype=np.float64)
        kc = self._kinematics.config.kinematics_config
        local = kc.link_spheres[0].cpu().numpy()
        link_idx = kc.link_sphere_idx_map.cpu().numpy()
        idx_to_name = {v: k for k, v in kc.link_name_to_idx_map.items()}
        yaml_spheres = {}
        gen = self._kinematics.config.generator_config
        if gen is not None and gen.collision_spheres is not None:
            yaml_spheres = gen.collision_spheres
        counts: dict = {}
        local_idx = np.zeros(len(link_idx), dtype=np.int32)
        for i, li in enumerate(link_idx):
            local_idx[i] = counts.get(int(li), 0)
            counts[int(li)] = int(local_idx[i]) + 1
        names = np.array(
            [idx_to_name.get(int(li), "?") for li in link_idx], dtype=object
        )
        yaml_xyz = np.zeros((len(link_idx), 3), dtype=np.float64)
        yaml_r = np.zeros(len(link_idx), dtype=np.float64)
        for i, name in enumerate(names):
            spec = yaml_spheres.get(str(name))
            j = int(local_idx[i])
            if spec is not None and j < len(spec):
                yaml_xyz[i] = spec[j]["center"]
                yaml_r[i] = spec[j]["radius"]
            else:
                yaml_xyz[i] = local[i, :3]
                yaml_r[i] = max(float(local[i, 3]), 0.0)
        valid = local[:, 3] > 0.0
        self._sphere_link = names[valid]
        self._sphere_local_idx = local_idx[valid]
        self._sphere_yaml_xyz = yaml_xyz[valid]
        self._sphere_yaml_radius = yaml_r[valid]
        n = self._sphere_world_xyz.shape[0]
        if self._sphere_link.shape[0] != n:
            self._sphere_link = np.array(["?"] * n, dtype=object)
            self._sphere_local_idx = np.arange(n, dtype=np.int32)
            self._sphere_yaml_xyz = np.zeros((n, 3), dtype=np.float64)
            self._sphere_yaml_radius = self._sphere_radius.copy()
        if self._selected_sphere_idx is not None and self._selected_sphere_idx >= n:
            self._selected_sphere_idx = None

    def add_frame(self, frame_name: str, frame_pose: Pose, scale: float = 0.2) -> viser.FrameHandle:
        frame_position = frame_pose.position.cpu().squeeze().numpy()
        frame_quat = frame_pose.quaternion.cpu().squeeze().numpy()
        frame = self._server.scene.add_frame(
            frame_name,
            position=frame_position,
            wxyz=frame_quat,
            axes_length=scale,
            axes_radius=max(scale * 0.05, 0.002),
        )
        return frame

    def add_batched_frames(self, frame_name: str, frame_poses: Pose):
        frame_positions = frame_poses.position.cpu().squeeze().numpy()
        frame_quats = frame_poses.quaternion.cpu().squeeze().numpy()
        self._batched_frames_handle = self._server.scene.add_batched_axes(
            name=frame_name,
            batched_positions=frame_positions,
            batched_wxyzs=frame_quats,
            axes_length=0.02,
            axes_radius=0.005,
        )

    def add_control_frame(
        self, frame_name: str, frame_pose: Pose, scale: float = 0.2
    ) -> viser.TransformControlsHandle:
        frame_position = frame_pose.position.cpu().squeeze().numpy()
        frame_quat = frame_pose.quaternion.cpu().squeeze().numpy()
        transform_control = self._server.scene.add_transform_controls(
            frame_name, scale=scale, position=frame_position, wxyz=frame_quat
        )
        return transform_control

    def reset_robot(self):
        joint_state = self._kinematics.get_full_js(self._kinematics.default_joint_state)
        self.set_joint_state(joint_state)

    def set_joint_positions(
        self, joint_positions: Sequence[float], joint_names: List[str], **kwargs
    ) -> None:
        joint_state = JointState.from_position(joint_positions, joint_names=joint_names)
        self.set_joint_state(joint_state)

    def get_control_frame_pose(self) -> Dict[str, Pose]:
        """Get pose of interactive frames.

        Note that these interactive frames are named with a "target_" prefix in the visualizer
        but this function returns the actual frame names (e.g., "target_ee_link" -> "ee_link").

        Returns:
            Dictionary of frame names and their poses.
        """
        poses = {}
        for frame_name in self._vis_frames:
            pose = Pose.from_numpy(
                self._control_frames[frame_name].position, self._control_frames[frame_name].wxyz
            )
            poses[frame_name] = pose
        return poses

    def set_control_frame_pose(self, poses: Dict[str, Pose]) -> None:
        """Move interactive target gizmos to the given world poses."""
        for frame_name, pose in poses.items():
            handle = self._control_frames.get(frame_name)
            if handle is None:
                continue
            p = pose.position.detach().cpu().numpy().reshape(-1)
            q = pose.quaternion.detach().cpu().numpy().reshape(-1)
            handle.position = (float(p[0]), float(p[1]), float(p[2]))
            handle.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def set_joint_state(self, joint_state: JointState) -> None:
        joint_state = joint_state.clone()
        viser_set = set(self._viser_joint_names)
        js_set = set(joint_state.joint_names)
        if viser_set != js_set:
            missing_from_js = viser_set - js_set
            if missing_from_js:
                joint_state = self._kinematics.get_full_js(joint_state)

        self._last_joint_state = joint_state.clone()
        if self._visualize_robot_spheres:
            self.update_robot_spheres(joint_state)

        joint_state.reindex(self._viser_joint_names)
        self._viser_urdf.update_cfg(joint_state.position.cpu().squeeze().numpy())
        if (
            not self._sphere_dragging
            and self._selected_link is not None
            and self._sphere_gizmo is not None
        ):
            idx = self._display_index(self._selected_link, self._selected_local)
            if idx is not None:
                self._selected_sphere_idx = idx
                self._highlight_selected_sphere()
                self._place_sphere_gizmo()

    def add_batched_spheres(self, spheres: List[Sphere]):
        # create a trimesh for the first sphere:
        sphere_mesh = spheres[0].get_trimesh_mesh()
        scale = []
        position = []
        quaternion = []
        mesh_radius = spheres[0].radius
        for i in range(0, len(spheres)):
            scale.append(spheres[i].radius / mesh_radius)
            position.append(spheres[i].pose[0:3])
            quaternion.append(spheres[i].pose[3:7])

        self._batched_spheres_handle = self._server.scene.add_batched_meshes_simple(
            name="curobo_spheres",
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            batched_scales=scale,
            batched_positions=position,
            batched_wxyzs=quaternion,
            opacity=0.55,
            cast_shadow=False,
        )
        self._ensure_sphere_inspect_gui()
        self._highlight_selected_sphere()

    def add_batched_spheres_from_position(
        self, position: np.ndarray, radius: np.ndarray, color=None, name: str = "curobo_spheres"
    ):
        # create a trimesh for the first sphere:
        radius[radius < 0.001] = 0.001
        sphere_mesh = Sphere(
            name="curobo_spheres", pose=[0, 0, 0, 1, 0, 0, 0], radius=radius[0]
        ).get_trimesh_mesh()
        quaternion = np.zeros((len(position), 4))
        quaternion[:, 0] = 1.0
        scale = radius / radius[0]
        colors = np.zeros((len(position), 3))
        if color is not None:
            colors[:, :] = color

        self._batched_spheres_handle = self._server.scene.add_batched_meshes_simple(
            name=name,
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            batched_scales=scale,
            batched_positions=position,
            batched_wxyzs=quaternion,
            batched_colors=colors,
        )

    def add_sphere(self, sphere: Sphere):
        if not isinstance(sphere, Sphere):
            log_and_raise("Sphere is not a valid Sphere object")
        sphere_handle = self._server.scene.add_icosphere(
            name=sphere.name,
            position=np.ravel(sphere.position),
            radius=sphere.radius,
            color=sphere.color if sphere.color is not None else (0, 200, 0),
        )
        return sphere_handle

    def add_line_segments(self, line_segments: np.ndarray, color: np.ndarray):
        points = line_segments
        colors = color

        self._server.scene.add_line_segments(
            "/line_segments",
            points=points,
            colors=colors,
            line_width=3.0,
        )

    def add_mesh(self, mesh_trimesh: trimesh.Trimesh, name: str = "mesh"):
        if not isinstance(mesh_trimesh, trimesh.Trimesh):
            log_and_raise("Mesh is not a valid Mesh object")
        mesh_handle = self._server.scene.add_mesh_trimesh(
            name=name,
            mesh=mesh_trimesh,
        )
        return mesh_handle

    def add_scene(self, scene_cfg: SceneCfg, add_control_frames: bool = False):
        obstacle_frames = {}
        mesh_scene = SceneCfg.create_mesh_scene(scene_cfg)
        for mesh in mesh_scene.mesh:
            self.add_mesh(
                mesh.get_trimesh_mesh(transform_with_pose=not add_control_frames),
                name="/obstacles/" + mesh.name + "/mesh",
            )
            if add_control_frames:
                obstacle_frames[mesh.name] = self.add_control_frame(
                    "/obstacles/" + mesh.name, Pose.from_list(mesh.pose), scale=0.2
                )
        return obstacle_frames

    def add_point_cloud(
        self,
        pointcloud: np.ndarray,
        colors=[200, 200, 200],
        point_size: float = 0.005,
        name: str = "pointcloud",
    ):
        handle = self._server.scene.add_point_cloud(
            name=name,
            points=pointcloud,
            colors=colors,
            point_size=point_size,
        )
        return handle

    def add_image(
        self,
        image: np.ndarray,
        render_width: float,
        render_height: float,
        pose: Pose,
        name: str = "image",
    ):
        """Add an image to the 3D scene at a specific pose.

        Args:
            image: RGB image array (H, W, 3) as uint8.
            render_width: Width of the image in world units (meters).
            render_height: Height of the image in world units (meters).
            pose: Pose of the image in world coordinates.
            name: Name for the image in the scene.

        Returns:
            Handle to the image object.
        """
        handle = self._server.scene.add_image(
            name=name,
            image=image,
            render_width=render_width,
            render_height=render_height,
            position=pose.position.cpu().squeeze().numpy(),
            wxyz=pose.quaternion.cpu().squeeze().numpy(),
        )
        return handle

