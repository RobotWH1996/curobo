// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/motion_planner.h"

#include "motion_planner_runtime.h"

#include <utility>

namespace curobo::robot {

TrajectoryOptimizationCandidates MotionPlanner::candidates() const {
  return runtime_->candidates();
}

TimeOptTrajectory MotionPlanner::timeopt_trajectory() const {
  return runtime_->timeopt_trajectory();
}

TimeOptCapacityDiagnostics MotionPlanner::last_timeopt_capacity_diagnostics() const {
  return runtime_->last_timeopt_capacity_diagnostics();
}
void MotionPlanner::set_execution_speed_scale(float scale) {
  runtime_->set_execution_speed_scale(scale);
}

const int32_t *MotionPlanner::validate(const TrajectoryValidationRequest &request) const {
  return runtime_->validate(request);
}

void MotionPlanner::validate_configurations(const float *positions, int count,
                                            uint8_t *violations, cudaStream_t stream) const {
  runtime_->validate_configurations(positions, count, violations, stream);
}

void MotionPlanner::classify_configurations(const float *positions, int count,
                                            uint8_t *feasible, cudaStream_t stream) const {
  runtime_->classify_configurations(positions, count, feasible, stream);
}

MotionPlanner::MotionPlanner(const std::string& artifact_directory,
                             const MotionPlanningConfig& config,
                             const Scene* scene)
    : runtime_(std::make_unique<detail::MotionPlannerRuntime>(
          artifact_directory, config, scene)) {}

void MotionPlanner::set_tool_upright(const std::vector<float> &weights,
                                     const std::vector<float> &signs) {
  runtime_->set_tool_upright(weights, signs);
}

void MotionPlanner::update_fixed_transforms(const std::vector<float> &transforms) {
  runtime_->update_fixed_transforms(transforms);
}

MotionPlanner::~MotionPlanner() = default;

MotionPlanner::MotionPlanner(MotionPlanner&& other) noexcept = default;

MotionPlanner& MotionPlanner::operator=(MotionPlanner&& other) noexcept =
    default;

CuroboRobotInfo MotionPlanner::robot_info() const {
  return runtime_->robot_info();
}

MotionPlanningInfo MotionPlanner::info() const { return runtime_->info(); }

const MotionPlanningConfig& MotionPlanner::config() const {
  return runtime_->config();
}

int MotionPlanner::native_horizon() const { return info().native_horizon(); }

double MotionPlanner::last_ik_time() const { return runtime_->last_ik_time(); }

double MotionPlanner::last_rrt_time() const {
  return runtime_->last_rrt_time();
}

int MotionPlanner::last_rrt_calls() const {
  return runtime_->last_rrt_calls();
}

TrajoptGraphTiming MotionPlanner::last_trajopt_graph_timing() const {
  return runtime_->last_trajopt_graph_timing();
}

bool MotionPlanner::retained_rrt_seed() const {
  return runtime_->retained_rrt_seed();
}

RrtDiagnostics MotionPlanner::last_rrt_diagnostics() const {
  return runtime_->last_rrt_diagnostics();
}

std::vector<std::vector<std::vector<float>>> MotionPlanner::last_rrt_paths() const {
  return runtime_->last_rrt_paths();
}

int MotionPlanner::attachment_sphere_capacity(
    const std::string& link_name) const {
  return runtime_->attachment_sphere_capacity(link_name);
}

void MotionPlanner::attach_object(const SphereAttachment& attachment) {
  runtime_->attach_object(attachment);
}

void MotionPlanner::set_link_collision_mask(const std::string& link_name, uint32_t layers) {
  runtime_->set_link_collision_mask(link_name, layers);
}

void MotionPlanner::detach_object(const std::string& link_name) {
  runtime_->detach_object(link_name);
}

void MotionPlanner::set_collision_mask(uint32_t layers) { runtime_->set_collision_mask(layers); }

void MotionPlanner::update_link_inertial(
    const LinkInertialUpdate& update) {
  runtime_->update_link_inertial(update);
}

void MotionPlanner::update_links_inertial(
    const LinkInertialUpdates& updates) {
  runtime_->update_links_inertial(updates);
}

void MotionPlanner::plan(const MotionPlanningRequest& request) const {
  runtime_->plan(request, /*reset_rrt_samples=*/true);
}

void MotionPlanner::plan_with_next_rrt_samples(
    const MotionPlanningRequest& request) const {
  runtime_->plan(request, /*reset_rrt_samples=*/false);
}

void MotionPlanner::plan_optimized(const MotionPlanningRequest& request,
                                  bool reset_rrt_samples) const {
  runtime_->plan(request, reset_rrt_samples, /*allow_retained_rrt_seed=*/false);
}

void MotionPlanner::plan_pose(
    const PoseMotionPlanningRequest& request) const {
  runtime_->plan_pose(request);
}

void MotionPlanner::plan_pose_waypoints(const WaypointMotionPlanningRequest &request) const {
  runtime_->plan_pose_waypoints(request);
}

std::unique_ptr<InverseKinematics> MotionPlanner::create_inverse_kinematics(
    const InverseKinematicsConfig& config) const {
  return runtime_->create_inverse_kinematics(config);
}

GraspPlanningStatus MotionPlanner::plan_grasp(
    const GraspPlanningRequest& request) const {
  return runtime_->plan_grasp(request);
}

}  // namespace curobo::robot
