// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
namespace curobo::robot::detail {
cudaError_t launch_tool_upright(const float *positions, const float *quaternions,
                                const float *settings, float *cost, float *position_gradient,
                                float *quaternion_gradient, int trajectories, int horizon,
                                int tools, cudaStream_t stream);
}
