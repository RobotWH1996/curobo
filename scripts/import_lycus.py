#!/usr/bin/env python3
"""Import Lycus into cuRobo.

Copies URDF + meshes from Lycus_ros. The robot is PD2616-2.1 with
AR5-5_08 W4C1C5 arms. Running this script rebuilds collision spheres.
The checked-in lycus.yml keeps the previous hand-tuned spheres and cspace
values, with link and joint names remapped onto this URDF.

AR5 STLs are millimetre CAD with URDF scale 0.001; SolidWorks body meshes
are open shells, so MorphIt/VOXEL fits are unreliable.

Active DoF is the 14 arm joints. Torso, head and wheels are locked at 0.
Default arm pose folds the arms down (joint_2 / joint_4 = 1.57) so the CAD
T-pose does not start in self-collision.
"""

from __future__ import annotations

import argparse
import math
import shutil
import struct
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import yaml

LYCUS_ROOT = Path("/home/victor/wh/Lycus_ros/src/lycus_description")
SRC_URDF = LYCUS_ROOT / "urdf" / "lycus.urdf"

CUROBO_ROOT = Path("/home/victor/wh/curobo/curobo")
DST_ASSET = CUROBO_ROOT / "content" / "assets" / "robot" / "lycus"
DST_MESH = DST_ASSET / "meshes"
DST_URDF = DST_ASSET / "lycus.urdf"
DST_YML = CUROBO_ROOT / "content" / "configs" / "robot" / "lycus.yml"

LEFT_ARM_JOINTS = [
    "AR5-5_08L-W4C1C5-ZY2_joint_1",
    "AR5-5_08L-W4C1C5-ZY2_joint_2",
    "AR5-5_08L-W4C1C5-ZY2_joint_3",
    "AR5-5_08L-W4C1C5-ZY2_joint_4",
    "AR5-5_08L-W4C1C5-ZY2_joint_5",
    "AR5-5_08L-W4C1C5-ZY2_joint_6",
    "AR5-5_08L-W4C1C5-ZY2_joint_7",
]
RIGHT_ARM_JOINTS = [
    "AR5-5_08R-W4C1C5-ZY2_joint_1",
    "AR5-5_08R-W4C1C5-ZY2_joint_2",
    "AR5-5_08R-W4C1C5-ZY2_joint_3",
    "AR5-5_08R-W4C1C5-ZY2_joint_4",
    "AR5-5_08R-W4C1C5-ZY2_joint_5",
    "AR5-5_08R-W4C1C5-ZY2_joint_6",
    "AR5-5_08R-W4C1C5-ZY2_joint_7",
]
ACTIVE_ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
# Gripper positions are feedback-only; ROS updates their locked transforms.
GRIPPER_STATE_JOINTS = ("left_gripper", "right_gripper")
TOOL_FRAMES = ["left_arm_ee_link", "right_arm_ee_link"]

# CAD T-pose. Capsule spheres are collision-free here; folding joint_2/4
# to 1.57 drives the forearms through the waist.
INITIAL_POSITIONS = {j: 0.0 for j in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS}

PACKAGE_PREFIX = "package://lycus_description/"
SKIP_COLLISION = {
    "world",
    "base_link",
    "Wheel1_link",
    "Wheel2_link",
    "Wheel3_link",
    "Wheel4_link",
    "HeadRGBD_link",
    "RGB1_link",
    "RGB2_link",
    "RGB3_link",
    "ChestRGBD_link",
    "camera_link",
}

PLACEHOLDER = """v -0.01 -0.01 -0.01
v  0.01 -0.01 -0.01
v  0.01  0.01 -0.01
v -0.01  0.01 -0.01
v -0.01 -0.01  0.01
v  0.01 -0.01  0.01
v  0.01  0.01  0.01
v -0.01  0.01  0.01
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 4 8 5 1
"""


def _xyz(el) -> tuple:
    origin = el.find("origin")
    if origin is None:
        return (0.0, 0.0, 0.0)
    return tuple(float(x) for x in origin.get("xyz", "0 0 0").split())


def copy_meshes() -> int:
    src_dir = LYCUS_ROOT / "meshes"
    DST_MESH.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in src_dir.iterdir():
        if src.suffix.lower() != ".stl":
            continue
        shutil.copy2(src, DST_MESH / src.name)
        n += 1
    return n


def rewrite_urdf(src: Path, dst: Path) -> list[str]:
    tree = ET.parse(src)
    root = tree.getroot()
    visual_links: list[str] = []
    for link in root.findall("link"):
        name = link.get("name")
        has_visual_mesh = False
        for mesh in link.iter("mesh"):
            fn = mesh.get("filename", "")
            if fn.startswith(PACKAGE_PREFIX):
                rel = fn[len(PACKAGE_PREFIX) :]
            else:
                rel = fn
            local = DST_ASSET / rel
            if local.is_file() and local.stat().st_size > 512:
                mesh.set("filename", rel)
                has_visual_mesh = True
            else:
                mesh.set("filename", "meshes/placeholder.obj")
        if has_visual_mesh and name:
            visual_links.append(name)
    for joint in root.findall("joint"):
        jtype = (joint.get("type") or "").lower()
        if jtype == "continuous":
            joint.set("type", "revolute")
            if joint.find("limit") is None:
                limit = ET.SubElement(joint, "limit")
                limit.set("lower", "-6.28")
                limit.set("upper", "6.28")
                limit.set("velocity", "6.28")
                limit.set("effort", "50")
        elif jtype == "revolute" and joint.find("limit") is None:
            limit = ET.SubElement(joint, "limit")
            limit.set("lower", "-3.14")
            limit.set("upper", "3.14")
            limit.set("velocity", "1.5")
            limit.set("effort", "10")
    for limit in root.iter("limit"):
        try:
            vel = float(limit.get("velocity", "1.0"))
        except ValueError:
            vel = 0.0
        if vel <= 0.0:
            limit.set("velocity", "1.5")
        try:
            effort = float(limit.get("effort", "1.0"))
        except ValueError:
            effort = 0.0
        if effort <= 0.0:
            limit.set("effort", "10")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return visual_links


def _as_spheres(layout: list[tuple]) -> list[dict]:
    return [
        {
            "center": [round(float(x), 4), round(float(y), 4), round(float(z), 4)],
            "radius": round(float(r), 4),
        }
        for x, y, z, r in layout
    ]


def _link_radius(parent: str, child: str) -> float:
    names = f"{parent} {child}".lower()
    if "wheel" in names:
        return 0.06
    if "chest" in names or "waist" in names or "thigh" in names or "calf" in names:
        return 0.07
    if "chassis" in names:
        return 0.08
    if "neck" in names or "head" in names:
        return 0.05
    if "link6" in names or "link7" in names or "flan" in names or "ee_link" in names:
        return 0.028
    if "link4" in names or "link5" in names:
        return 0.038
    if "link2" in names or "link3" in names:
        return 0.045
    if "link1" in names or "_base" in names:
        return 0.05
    return 0.04


def _load_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:5].lower() == b"solid" and b"facet" in data[:400].lower():
        verts = []
        for line in data.decode("utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if parts and parts[0] == "vertex" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.asarray(verts, dtype=np.float64)
    count = struct.unpack_from("<I", data, 80)[0]
    verts = np.empty((count * 3, 3), dtype=np.float64)
    for i in range(count):
        base = 84 + i * 50
        for k in range(3):
            verts[i * 3 + k] = struct.unpack_from("<fff", data, base + 12 + 12 * k)
    return verts


def _link_mesh_vertices(root: ET.Element, link_name: str) -> np.ndarray | None:
    link = root.find(f"link[@name='{link_name}']")
    if link is None:
        return None
    chunks = []
    for geom_parent in list(link.findall("collision")) or list(link.findall("visual")):
        mesh = geom_parent.find("geometry/mesh")
        if mesh is None or not mesh.get("filename"):
            continue
        rel = mesh.get("filename").replace(PACKAGE_PREFIX, "")
        path = LYCUS_ROOT / rel
        if not path.exists():
            hits = list((LYCUS_ROOT / "meshes").glob(Path(rel).name))
            if not hits:
                continue
            path = hits[0]
        verts = _load_stl_vertices(path)
        scale = mesh.get("scale")
        if scale:
            verts = verts * np.array([float(v) for v in scale.split()], dtype=np.float64)
        chunks.append(verts)
    if not chunks:
        return None
    verts = np.concatenate(chunks, axis=0)
    if len(verts) > 8000:
        idx = np.linspace(0, len(verts) - 1, 8000).astype(np.int64)
        verts = verts[idx]
    return verts


def _sphere(center, radius: float) -> dict:
    return {
        "center": [round(float(c), 4) for c in center],
        "radius": round(float(radius), 4),
    }


# Oversized on purpose. Shrink these in lycus.yml after the line looks right.
ARM_LINE_RADIUS = 0.08


def _outgoing_xyz(root: ET.Element, link_name: str) -> np.ndarray:
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if parent is not None and parent.get("link") == link_name:
            return np.asarray(_xyz(joint), dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _fit_arm_line(verts: np.ndarray, end: np.ndarray) -> list[tuple]:
    """Collision spheres in one straight line along the arm segment.

    Direction follows the child joint when that segment is long enough.
    Short links use the mesh's longest axis, and stay on Z when that extent
    is close, so the shoulder and wrist do not line up sideways. The line
    passes through the mesh center.
    """
    length = float(np.linalg.norm(end))
    extent = verts.max(axis=0) - verts.min(axis=0)
    if length >= 0.05:
        direction = end / length
    else:
        axis = int(np.argmax(extent))
        if extent[2] >= 0.8 * float(extent[axis]):
            axis = 2
        direction = np.zeros(3, dtype=np.float64)
        direction[axis] = 1.0
    proj = verts @ direction
    lateral = verts - np.outer(proj, direction)
    offset = lateral.mean(axis=0)
    p0, p1 = float(proj.min()), float(proj.max())
    span = max(p1 - p0, 1e-3)
    if span < 0.03:
        return [(verts.mean(axis=0), ARM_LINE_RADIUS)]
    n_spheres = int(np.clip(round(span / 0.09) + 1, 2, 6))
    inset = min(0.25 * span, 0.03)
    samples = np.linspace(p0 + inset, p1 - inset, n_spheres)
    return [(offset + direction * float(s), ARM_LINE_RADIUS) for s in samples]


def _fit_sparse_volume(verts: np.ndarray, radius: float, max_axis: int = 3) -> list[tuple]:
    """A few spheres inside a bulky shell (head / chest), Franka density."""
    low, high = verts.min(axis=0), verts.max(axis=0)
    extent = high - low
    inset = max(radius * 0.55, 0.02)
    spheres: list[tuple] = []
    axes: list[np.ndarray] = []
    for dim in range(3):
        lo = float(low[dim] + inset)
        hi = float(high[dim] - inset)
        if hi - lo < 0.03:
            axes.append(np.array([(low[dim] + high[dim]) / 2.0]))
            continue
        count = min(max_axis, max(2, int(round((hi - lo) / (radius * 1.4)))))
        axes.append(np.linspace(lo, hi, count))
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                center = np.array([x, y, z], dtype=np.float64)
                if np.count_nonzero(np.linalg.norm(verts - center, axis=1) <= radius) >= 20:
                    spheres.append((center, radius))
    return spheres or [(verts.mean(axis=0), radius)]


def _as_fitted(spheres: list[tuple]) -> list[dict]:
    return [_sphere(center, radius) for center, radius in spheres]


def _replace_with_mesh_fit(root: ET.Element, spheres: dict) -> None:
    """Fit head and chest from the mesh. Arms are a straight line of spheres."""
    arm_names = [
        name
        for name in spheres
        if "AR5-5_08" in name or name in ("left_arm_ee_link", "right_arm_ee_link")
    ]
    for link_name in arm_names:
        if "flan" in link_name or link_name.endswith("ee_link"):
            spheres[link_name] = [_sphere([0.0, 0.0, 0.0], ARM_LINE_RADIUS)]
            print(f"  arm line {link_name}: 1 sphere, r={ARM_LINE_RADIUS:.3f} m")
            continue
        verts = _link_mesh_vertices(root, link_name)
        if verts is None:
            continue
        fitted = _as_fitted(_fit_arm_line(verts, _outgoing_xyz(root, link_name)))
        if not fitted:
            continue
        spheres[link_name] = fitted
        print(f"  arm line {link_name}: {len(fitted)} spheres, r={ARM_LINE_RADIUS:.3f} m")

    for link_name, fitted_fn in (
        ("Head_link", lambda verts: _fit_sparse_volume(verts, radius=0.05, max_axis=2)),
        ("Chest_link", lambda verts: _fit_sparse_volume(verts, radius=0.06, max_axis=3)),
    ):
        if link_name not in spheres:
            continue
        verts = _link_mesh_vertices(root, link_name)
        if verts is None:
            continue
        fitted = _as_fitted(fitted_fn(verts))
        if not fitted:
            continue
        spheres[link_name] = fitted
        radii = [item["radius"] for item in fitted]
        print(
            f"  mesh fit {link_name}: {len(fitted)} spheres, "
            f"r={min(radii):.3f}-{max(radii):.3f} m"
        )


def capsules_from_urdf(src: Path) -> dict:
    root = ET.parse(src).getroot()
    spheres: dict[str, list] = {}

    def add(link: str, center, radius: float):
        if link in SKIP_COLLISION:
            return
        spheres.setdefault(link, []).append(
            {"center": [round(float(c), 4) for c in center], "radius": float(radius)}
        )

    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link")
        child = child_el.get("link")
        ox, oy, oz = _xyz(joint)
        length = math.sqrt(ox * ox + oy * oy + oz * oz)
        if parent == "world":
            continue
        radius = _link_radius(parent, child)
        if parent == "Chest_link" and length > 0.05:
            add(child, [0.0, 0.0, 0.0], 0.05)
            continue
        n = max(1, int(length / 0.05) + 1)
        for i in range(n):
            t = 0.0 if n == 1 else i / (n - 1)
            add(parent, [ox * t, oy * t, oz * t], radius)
        add(child, [0.0, 0.0, 0.0], radius)

    # Chest volume: stay on the torso, not out into the armpits.
    for z in (0.04, 0.12, 0.20):
        for x in (-0.06, 0.02):
            add("Chest_link", [x, 0.0, z], 0.055)
    for z in (0.06, 0.16):
        add("Chassis_link", [-0.12, 0.0, z], 0.09)
        add("Chassis_link", [0.02, 0.0, z], 0.08)

    cleaned = {}
    for link, items in spheres.items():
        if link in SKIP_COLLISION:
            continue
        uniq = []
        for s in items:
            if any(
                math.dist(s["center"], u["center"]) < 0.025
                and abs(s["radius"] - u["radius"]) < 0.012
                for u in uniq
            ):
                continue
            uniq.append(s)
        if uniq:
            cleaned[link] = uniq
    _replace_with_mesh_fit(root, cleaned)
    return cleaned


def always_colliding_pairs(
    yml: Path, num_samples: int = 300, frac: float = 0.9, seed: int = 0
) -> dict[str, list[str]]:
    import numpy as np
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState

    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(str(yml)))
    data = yaml.safe_load(yml.read_text())["robot_cfg"]["kinematics"]
    spheres = data["collision_spheres"]
    ignore = data["self_collision_ignore"]
    owner = [link for link in data["collision_link_names"] for _ in spheres[link]]

    limits = kin.config.get_joint_limits()
    lo = limits.position_lower_limits.cpu().numpy()
    hi = limits.position_upper_limits.cpu().numpy()
    rng = np.random.default_rng(seed)
    q0 = kin.default_joint_state.position
    names = kin.joint_names
    hits: dict[tuple[str, str], int] = {}
    for _ in range(num_samples):
        q = q0.new_tensor(rng.uniform(lo, hi)).view(1, -1)
        state = kin.compute_kinematics(JointState.from_position(q, joint_names=names))
        s = state.robot_spheres.reshape(-1, 4).cpu().numpy()
        c, r = s[:, :3], s[:, 3]
        live = r > 0
        gap = np.linalg.norm(c[:, None] - c[None], axis=2) - (r[:, None] + r[None])
        seen = set()
        for i, j in np.argwhere((gap < 0) & live[:, None] & live[None]):
            if i >= j or j >= len(owner):
                continue
            a, b = owner[i], owner[j]
            if a == b or b in ignore.get(a, []):
                continue
            seen.add((a, b) if a < b else (b, a))
        for key in seen:
            hits[key] = hits.get(key, 0) + 1

    out: dict[str, list[str]] = {}
    for (a, b), n in sorted(hits.items(), key=lambda kv: -kv[1]):
        if n >= frac * num_samples:
            out.setdefault(a, []).append(b)
            print(f"  always colliding: {a} <-> {b} ({n}/{num_samples})")
    return out


def postprocess_yaml(
    path: Path,
    lock_joints: dict,
    spheres: dict,
    mesh_link_names: list[str],
    sphere_buffer: float,
    extra_ignore: dict[str, list[str]] | None = None,
) -> None:
    with path.open() as f:
        data = yaml.safe_load(f)
    kin = data["kinematics"] if "kinematics" in data else data["robot_cfg"]["kinematics"]
    cspace = kin.get("cspace") or {}
    names = cspace.get("joint_names") or []
    names = [j for j in ACTIVE_ARM_JOINTS if j in names] + [
        j for j in names if j not in ACTIVE_ARM_JOINTS
    ]
    cspace["joint_names"] = names
    kin["lock_joints"] = {j: float(v) for j, v in lock_joints.items() if j in names}
    kin["tool_frames"] = list(TOOL_FRAMES)
    kin["base_link"] = "base_link"
    kin["urdf_path"] = "robot/lycus/lycus.urdf"
    kin["asset_root_path"] = "robot/lycus"
    kin["collision_spheres"] = spheres
    kin["collision_link_names"] = list(spheres.keys())
    kin["mesh_link_names"] = [n for n in mesh_link_names if n not in SKIP_COLLISION] or list(
        spheres.keys()
    )
    kin["load_meshes"] = True
    kin["collision_sphere_buffer"] = sphere_buffer
    kin["extra_collision_spheres"] = {"attached_object": 4}
    kin["extra_links"] = {
        "attached_object": {
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_name": "attach_joint",
            "joint_type": "FIXED",
            "link_name": "attached_object",
            "parent_link_name": TOOL_FRAMES[0],
        }
    }
    kin["self_collision_buffer"] = {name: 0.0 for name in spheres}
    ignore = kin.get("self_collision_ignore") or {}
    ignore = {k: v for k, v in ignore.items() if k in spheres or k == "attached_object"}

    def add_ignore(a: str, b: str) -> None:
        if a not in spheres or b not in spheres:
            return
        ignore.setdefault(a, [])
        ignore.setdefault(b, [])
        if b not in ignore[a]:
            ignore[a].append(b)
        if a not in ignore[b]:
            ignore[b].append(a)

    for a, others in (extra_ignore or {}).items():
        for b in others:
            add_ignore(a, b)

    torso = [
        "Chassis_link",
        "Calf_link",
        "Thigh_link",
        "Waist_link",
        "Chest_link",
        "Neck_link",
        "Head_link",
    ]
    for i, a in enumerate(torso):
        for b in torso[i + 1 :]:
            add_ignore(a, b)

    for side in ("L", "R"):
        prefix = f"AR5-5_08{side}-W4C1C5-ZY2_"
        chain = [f"{prefix}base"] + [f"{prefix}link{i}" for i in range(1, 8)] + [
            f"{prefix}flan_link"
        ]
        for i, a in enumerate(chain):
            for b in chain[i + 1 : min(i + 4, len(chain))]:
                add_ignore(a, b)
        add_ignore("Chest_link", f"{prefix}base")
        add_ignore("Chest_link", f"{prefix}link1")
        add_ignore("Chest_link", f"{prefix}link2")
        add_ignore("Neck_link", f"{prefix}base")
        add_ignore("Waist_link", f"{prefix}base")

    add_ignore("left_arm_ee_link", "AR5-5_08L-W4C1C5-ZY2_link7")
    add_ignore("left_arm_ee_link", "AR5-5_08L-W4C1C5-ZY2_flan_link")
    add_ignore("right_arm_ee_link", "AR5-5_08R-W4C1C5-ZY2_link7")
    add_ignore("right_arm_ee_link", "AR5-5_08R-W4C1C5-ZY2_flan_link")

    # Grippers are locked, with measured transforms updated by ROS. Adjacent parts overlap
    # at the mount, and the two jaws are a mimic pair that is allowed to meet.
    for side, prefix in (("left", "AR5-5_08L-W4C1C5-ZY2_"), ("right", "AR5-5_08R-W4C1C5-ZY2_")):
        base = f"{side}_gripper_base"
        jaws = (f"{side}_gripper_right_jaw", f"{side}_gripper_left_jaw")
        ee = f"{side}_arm_ee_link"
        neighbors = (f"{prefix}flan_link", f"{prefix}link7", ee)
        for neighbor in neighbors:
            add_ignore(base, neighbor)
            for jaw in jaws:
                add_ignore(jaw, neighbor)
        add_ignore(base, jaws[0])
        add_ignore(base, jaws[1])
        add_ignore(jaws[0], jaws[1])

    ignore.setdefault("attached_object", [])
    for link in (
        "left_arm_ee_link",
        "AR5-5_08L-W4C1C5-ZY2_link6",
        "AR5-5_08L-W4C1C5-ZY2_link7",
        "AR5-5_08L-W4C1C5-ZY2_flan_link",
        "left_gripper_base",
        "left_gripper_right_jaw",
        "left_gripper_left_jaw",
    ):
        if link in spheres and "attached_object" not in ignore.get(link, []):
            ignore.setdefault(link, []).append("attached_object")
        if link not in ignore["attached_object"]:
            ignore["attached_object"].append(link)
    kin["self_collision_ignore"] = ignore

    n = len(names)
    if n:
        cspace["cspace_distance_weight"] = [
            0.0 if j in GRIPPER_STATE_JOINTS else 1.0 for j in names
        ]
        cspace["null_space_weight"] = [
            0.0 if j in GRIPPER_STATE_JOINTS else 1.0 for j in names
        ]
        cspace["default_joint_position"] = [
            float(INITIAL_POSITIONS.get(j, 0.0)) for j in names
        ]
        cspace["max_acceleration"] = 3.0
        cspace["max_jerk"] = 500.0
        for scale_key, fill in (("velocity_scale", 1.0), ("acceleration_scale", 1.0), ("jerk_scale", 1.0)):
            values = cspace.get(scale_key)
            if not isinstance(values, list) or len(values) != n:
                values = [fill] * n
            cspace[scale_key] = [
                1.0 if names[i] in GRIPPER_STATE_JOINTS else float(values[i])
                for i in range(n)
            ]
        kin["cspace"] = cspace
    dumped = yaml.safe_dump({"robot_cfg": {"kinematics": kin}}, sort_keys=False)
    path.write_text(
        "# Lycus cuRobo robot config.\n"
        "# Planning uses collision_spheres; visualization uses URDF/STL.\n"
        "# Regenerate with: python scripts/import_lycus.py\n"
        + dumped
    )


def install_into_live_curobo() -> None:
    """Copy generated assets into the curobo the current interpreter imports."""
    from curobo.content import get_assets_path, get_robot_configs_path

    live_asset = Path(get_assets_path()) / "robot" / "lycus"
    live_yml = Path(get_robot_configs_path()) / "lycus.yml"
    if live_asset.resolve() != DST_ASSET.resolve():
        if live_asset.exists():
            shutil.rmtree(live_asset)
        shutil.copytree(DST_ASSET, live_asset)
        print(f"copied assets -> {live_asset}")
    if live_yml.resolve() != DST_YML.resolve():
        shutil.copy2(DST_YML, live_yml)
        print(f"copied yaml -> {live_yml}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Lycus into cuRobo")
    parser.add_argument(
        "--sphere-buffer",
        type=float,
        default=0.002,
        help="Grow every sphere by this radius (m)",
    )
    parser.add_argument(
        "--skip-always-pairs",
        action="store_true",
        help="Skip sampling always-colliding pairs (faster, less complete ACM)",
    )
    args = parser.parse_args()

    DST_MESH.mkdir(parents=True, exist_ok=True)
    (DST_MESH / "placeholder.obj").write_text(PLACEHOLDER)
    n_stl = copy_meshes()
    print(f"copied {n_stl} STL meshes")
    visual_links = rewrite_urdf(SRC_URDF, DST_URDF)
    spheres = capsules_from_urdf(SRC_URDF)
    print(
        f"capsule spheres: {sum(len(v) for v in spheres.values())} "
        f"on {len(spheres)} links"
    )

    from curobo.robot_builder import RobotBuilder

    builder = RobotBuilder(
        urdf_path=str(DST_URDF),
        asset_path=str(DST_ASSET),
        tool_frames=list(TOOL_FRAMES),
    )
    builder._collision_spheres = spheres
    builder.compute_collision_matrix(prune_collisions=False)
    config = builder.build()
    builder.save(config, str(DST_YML))

    lock = {
        j: float(INITIAL_POSITIONS.get(j, 0.0))
        for j in builder._joint_names
        if j not in ACTIVE_ARM_JOINTS
    }
    postprocess_yaml(DST_YML, lock, spheres, visual_links, args.sphere_buffer)
    install_into_live_curobo()
    if not args.skip_always_pairs:
        extra = always_colliding_pairs(DST_YML)
        if extra:
            postprocess_yaml(
                DST_YML, lock, spheres, visual_links, args.sphere_buffer, extra
            )
            install_into_live_curobo()
    print(f"Wrote {DST_YML}")
    print(f"active joints: {ACTIVE_ARM_JOINTS}")
    print(f"visual links: {len(visual_links)}")


if __name__ == "__main__":
    main()
