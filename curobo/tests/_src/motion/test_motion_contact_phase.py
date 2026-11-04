# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contact scheduling preserves geometry and CUDA graph replay semantics."""
import math
import torch
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState
from curobo._src.geom.quaternion import angular_distance_axis_angle


def test_contact_phase_cuda_graph_replay() -> None:
    model = Kinematics(KinematicsCfg.from_robot_yaml_file('lycus.yml'))
    q = model.default_joint_position.reshape(1, 1, -1).repeat(1, 9, 1)
    state = JointState.from_position(q, joint_names=model.joint_names)
    indices = model.config.kinematics_config.get_sphere_index_from_link_name('left_gripper_left_jaw').cpu().tolist()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): model.compute_kinematics(state)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): output = model.compute_kinematics(state).robot_spheres
    graph.replay(); baseline = output.clone()
    pointer = model._collision_phase_masks[(1, 9)].data_ptr()
    model.set_collision_phase(indices, .5)
    graph.replay()
    torch.testing.assert_close(output[:, :5], baseline[:, :5])
    assert bool((output[:, 5:, indices, 3] < 0).all())
    torch.testing.assert_close(output[..., :3], baseline[..., :3])
    others = [i for i in range(model.total_spheres) if i not in indices]
    torch.testing.assert_close(output[:, :, others], baseline[:, :, others])
    assert model._collision_phase_masks[(1, 9)].data_ptr() == pointer
    model.set_collision_phase([], None); graph.replay()
    torch.testing.assert_close(output, baseline)


def test_contact_phase_gradient_and_single_state() -> None:
    model = Kinematics(KinematicsCfg.from_robot_yaml_file('lycus.yml'))
    index = model.config.kinematics_config.get_sphere_index_from_link_name('left_gripper_left_jaw').cpu().tolist()
    model.set_collision_phase(index, .5)
    q = model.default_joint_position.reshape(1, 1, -1).repeat(1, 9, 1).clone().requires_grad_(True)
    spheres = model.compute_kinematics(JointState.from_position(q, joint_names=model.joint_names)).robot_spheres
    spheres[..., :3].square().sum().backward()
    assert q.grad is not None and bool(torch.isfinite(q.grad).all())
    assert float(q.grad.abs().max()) > 0
    single = model.compute_kinematics(model.default_joint_state).robot_spheres
    assert bool((single[:, :, index, 3] > 0).all())


def test_pairwise_quaternion_angles() -> None:
    identity = torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]], dtype=torch.float32)
    current = torch.tensor([[1.,0.,0.,0.],[math.cos(math.pi/6),0.,0.,math.sin(math.pi/6)]], dtype=torch.float32)
    angle = angular_distance_axis_angle(identity, current)
    assert angle.shape == (2,)
    torch.testing.assert_close(angle, torch.tensor([0.,math.pi/3], dtype=torch.float32))
    torch.testing.assert_close(angular_distance_axis_angle(identity, -current), angle)


def test_contact_phase_uses_each_valid_length() -> None:
    model = Kinematics(KinematicsCfg.from_robot_yaml_file('lycus.yml'))
    q = model.default_joint_position.reshape(1, 1, -1).repeat(2, 9, 1)
    indices = model.config.kinematics_config.get_sphere_index_from_link_name('left_gripper_left_jaw').cpu().tolist()
    model.set_collision_phase(indices, .5)
    model.set_collision_horizon(torch.tensor([5, 9], device=q.device))
    result = model.compute_kinematics(JointState.from_position(q, joint_names=model.joint_names)).robot_spheres
    assert bool((result[0, :3, indices, 3] > 0).all())
    assert bool((result[0, 3:, indices, 3] < 0).all())
    assert bool((result[1, :5, indices, 3] > 0).all())
    assert bool((result[1, 5:, indices, 3] < 0).all())
    model.set_collision_horizon(None)
    model.set_collision_phase([], None)
