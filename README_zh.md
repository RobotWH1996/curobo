<!-- SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# cuRobo

*CUDA 加速机器人库*

**[文档](https://nvlabs.github.io/curobo) | [论文](https://arxiv.org/abs/2603.05493)**

> [!NOTE]
> cuRoboV2 是一次大幅重写，公开 API 已相对 cuRobo v1 发生变化。
> 若仍依赖 v1 API，请固定使用 [`v0.7.8`](https://github.com/NVlabs/curobo/tree/v0.7.8) 标签。

cuRobo 是一个面向机器人运动生成的 CUDA 加速库，基于 PyTorch、CUDA 与 Warp 构建。它提供 GPU 并行算法，覆盖正/逆运动学、碰撞检测、轨迹优化、几何规划、GPU 原生感知以及全身运动生成，可从单臂机械臂扩展到高自由度人形机器人。

核心能力：
- **动力学感知的轨迹优化**：采用 B 样条表示，约束平滑性与力矩限制
- **GPU 原生 ESDF 感知**：由深度图生成稠密有符号距离场，最高可达现有先进方法约 10 倍速度
- **可扩展的全身计算**：包含拓扑感知运动学、可微逆动力学，以及面向高自由度机器人的 map-reduce 自碰撞
- **无碰撞运动生成**：融合 IK、几何规划与轨迹优化

## 引用

如果本工作对你有帮助，请引用 cuRoboV2：

```
@misc{curobo_v2,
      title={cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots},
      author={Balakumar Sundaralingam and Adithyavairavan Murali and Stan Birchfield},
      year={2026},
      eprint={2603.05493},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## 贡献

欢迎贡献。Bug 反馈：[提交 Issue](https://github.com/NVlabs/curobo/issues)。一般使用问题：[GitHub Discussions](https://github.com/NVlabs/curobo/discussions)。提交 Pull Request 前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。所有提交必须包含 DCO 签名（`git commit -s`）。

## 许可证

cuRobo 以 [Apache 2.0 许可证](LICENSE) 发布。

本仓库附带的示例机器人资源按其各自许可证提供，详见 [LICENSE_ASSETS](LICENSE_ASSETS)。

## 第三方软件

本项目会下载并安装额外的第三方开源软件。使用前请查阅这些开源项目的许可条款。
