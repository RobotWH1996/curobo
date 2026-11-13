// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include "tool_constraint_cuda.h"
namespace curobo::robot::detail {
namespace {
__global__ void tool_upright(const float *positions, const float *quaternions,
                             const float *settings, float *cost, float *position_gradient,
                             float *quaternion_gradient, int count, int horizon, int tools) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int tool = index % tools, time = (index / tools) % horizon;
  const float weight = settings[tool * 2], sign = settings[tool * 2 + 1];
  if (weight == 0 || time == 0 || time == horizon - 1) return;
  const float x = quaternions[index * 4 + 1], y = quaternions[index * 4 + 2];
  const float inclination = fmaxf(0.F, 0.965925826289F - sign * (1.F - 2.F * (x * x + y * y)));
  float extra = inclination * weight;
  if (inclination > 0) {
    quaternion_gradient[index * 4 + 1] += 4.F * weight * sign * x;
    quaternion_gradient[index * 4 + 2] += 4.F * weight * sign * y;
  }
  const int start = index - time * tools, end = start + (horizon - 1) * tools;
  float direction[3], relative[3];
  float squared = 0.F, fraction = 0.F;
  for (int axis = 0; axis < 3; ++axis) {
    direction[axis] = positions[end * 3 + axis] - positions[start * 3 + axis];
    relative[axis] = positions[index * 3 + axis] - positions[start * 3 + axis];
    squared += direction[axis] * direction[axis];
    fraction += relative[axis] * direction[axis];
  }
  fraction = fminf(1.F, fmaxf(0.F, fraction / fmaxf(squared, 1e-9F)));
  float delta[3], distance = 0.F;
  for (int axis = 0; axis < 3; ++axis) {
    delta[axis] = relative[axis] - fraction * direction[axis];
    distance += delta[axis] * delta[axis];
  }
  if (distance > 0.01F) {
    extra += 3.F * weight * (distance - 0.01F);
    for (int axis = 0; axis < 3; ++axis)
      position_gradient[index * 3 + axis] += 6.F * weight * delta[axis];
  }
  cost[index * 2] += extra;
}
}  // namespace
cudaError_t launch_tool_upright(const float *positions, const float *quaternions,
                                const float *settings, float *cost, float *position_gradient,
                                float *quaternion_gradient, int trajectories, int horizon,
                                int tools, cudaStream_t stream) {
  const int count = trajectories * horizon * tools;
  tool_upright<<<(count + 127) / 128, 128, 0, stream>>>(positions, quaternions, settings, cost,
                                                        position_gradient, quaternion_gradient,
                                                        count, horizon, tools);
  return cudaGetLastError();
}
}  // namespace curobo::robot::detail
