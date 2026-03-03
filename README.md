# SPH-FSI — 流固耦合模拟合集

基于 **WCSPH（弱可压缩光滑粒子流体动力学）** 与 **虚拟边界粒子法（Akinci 2012）** 的 2D 流固耦合演示项目，使用 [Taichi](https://taichi-lang.org/) 进行 GPU 加速。

---

## Demo

> 圆形刚体漂浮演示 (circle_fsi)

<video src="https://github.com/user-attachments/assets/c81e71ec-b2d6-4c25-afb8-68310163f4d7" controls width="700"></video>

## 子项目总览

| 目录 | 说明 | 刚体形状 | 流体 | 粒子数 | 刚体碰撞 |
|------|------|----------|------|--------|----------|
| [`original/`](./original/README.md) | 初始版本，代码最简 | 矩形 | WCSPH | ~1 000 | SAT+SI |
| [`taichi_parallel/`](./taichi_parallel/README.md) | 高分辨率版，效果最好 | 矩形 | WCSPH | ~20 000 | SAT+SI |
| [`circle_fsi/`](./circle_fsi/README.md) | 圆形刚体漂浮演示 | 圆形 | WCSPH | ~20 000 | 无 |
| [`rigid_body/`](./rigid_body/README.md) | 纯刚体碰撞，无流体 | 矩形 | 无 | — | SAT+SI |

---

## 快速开始

### 安装依赖

```bash
pip install taichi numpy
```

### 运行各版本

```bash
# 高分辨率版（推荐）
cd taichi_parallel
python main.py

# 圆形刚体版
cd circle_fsi
python main.py

# 纯刚体版（无需 GPU）
cd rigid_body
python main.py

# 初始版（最轻量）
cd original
python main.py
```

---

## 算法概要

| 模块 | 方法 |
|------|------|
| 流体求解 | WCSPH — Tait 物态方程显式压力（Monaghan 1992） |
| 时间步进 | CFL 自适应 — 基于声速 / 最大速度 / 加速度动态调整 dt |
| 边界处理 | 虚拟粒子法 — 墙壁 / 刚体表面采样，伪质量 ψ（Akinci 2012） |
| 邻居搜索 | 均匀网格 + Counting Sort + Prefix Sum |
| 刚体动力学 | 牛顿-欧拉积分，流体→虚拟粒子→力矩累加 |
| 刚体碰撞 | SAT 窄相检测 + Sequential Impulse 约束求解（移植自 box2d-lite） |
| 可视化 | Taichi GGUI（canvas.circles + canvas.lines） |

### 耦合流程（每子步）

```
同步虚拟粒子 → 网格邻居搜索 → 密度计算
  → 非压力力（重力 + 粘性）→ 压力力（Tait EOS）
  → CFL 自适应 dt → 刚体受力累加
  → 流体平流（v += dt·a, x += dt·v）
  → 刚体速度积分 → SAT+SI 碰撞求解 → 刚体位置积分 → 边界钳位
```

---

## 参考文献

- Akinci N., et al. *"Versatile rigid-fluid coupling for incompressible SPH."* ACM TOG, 2012.
- Catto E. *"Iterative Dynamics with Temporal Coherence."* GDC, 2005.
- Monaghan J. J. *"Smoothed particle hydrodynamics."* Annual Review of Astronomy and Astrophysics, 1992.
