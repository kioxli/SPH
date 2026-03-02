# FSI Demo — WCSPH + 虚拟粒子法流固耦合

基于 **WCSPH (弱可压缩光滑粒子流体动力学)** 与 **虚拟边界粒子法 (Akinci 2012)** 的 2D 流固耦合演示，使用 [Taichi](https://taichi-lang.org/) 进行 GPU 加速。

---

## 效果

- 溃坝 (dam-break) 冲击可移动刚体
- **左键点击** 在鼠标位置添加新方块（最多 10 个）
- 刚体受流体压力驱动，可自由平移 + 旋转
- 刚体间碰撞响应 (SAT + Sequential Impulse)

## 算法概要

| 模块 | 方法 |
|------|------|
| 流体求解 | **WCSPH** — Tait 物态方程显式压力 (Monaghan 1992) |
| 时间步进 | CFL 自适应 — 基于声速 / 最大速度 / 加速度动态调整 dt |
| 边界处理 | 虚拟粒子法 — 墙壁 / 刚体表面采样, 伪质量 ψ (Akinci 2012) |
| 邻居搜索 | 均匀网格 + Counting Sort + Prefix Sum |
| 刚体动力学 | 牛顿-欧拉积分, 流体→虚拟粒子→力矩累加 |
| 刚体碰撞 | **SAT** 窄相检测 + **Sequential Impulse** 约束求解 (移植自 box2d-lite) |
| 可视化 | Taichi GGUI (canvas.circles + canvas.lines) |

### 耦合流程 (每子步)

```
同步虚拟粒子 → 网格邻居搜索 → 密度计算
  → 非压力力 (重力 + 粘性) → 压力力 (Tait EOS)
  → CFL 自适应 dt → 刚体受力累加
  → 流体平流 (v += dt·a, x += dt·v)
  → 刚体速度积分 → SAT+SI 碰撞求解 → 刚体位置积分 → 边界钳位
```

## 项目结构

```
fsi/
├── main.py                   # 入口: 窗口 + 事件循环 + 渲染
├── config.py                 # 全局参数配置
├── particles.py              # 粒子生成 & 边界 ψ 计算 (纯 NumPy)
├── simulation.py             # Taichi 引擎: WCSPH + 刚体 + 可视化
├── collision.py              # 刚体碰撞: SAT 检测 + Sequential Impulse (CPU)
├── validate_hydrostatic.py   # 正确性验证: 静水压力对比解析解
├── requirements.txt          # 依赖
├── README.md
│
├── original/          # 初始版本 (r=0.012, ~1000 粒子, 低分辨率)
├── taichi_parallel/   # 高分辨率版本 (r=0.005, ~20000 粒子, 矩形刚体)
├── circle_fsi/        # 圆形刚体版本 (无刚体间碰撞)
└── rigid_body/        # 纯刚体模拟 (无流体, box2d-lite 风格)
```

### 版本对比

| 目录 | 刚体形状 | 流体 | 粒子数 | 刚体碰撞 | 说明 |
|------|----------|------|--------|----------|------|
| `fsi/`（本目录）| 矩形 | WCSPH | ~1 000 | SAT+SI | 主演示版本 |
| `original/` | 矩形 | WCSPH | ~1 000 | SAT+SI | 起点版本，代码最简 |
| `taichi_parallel/` | 矩形 | WCSPH | ~20 000 | SAT+SI | 高分辨率，效果最好 |
| `circle_fsi/` | 圆形 | WCSPH | ~20 000 | 无 | 圆球漂浮演示 |
| `rigid_body/` | 矩形 | 无 | — | SAT+SI | 纯碰撞测试，无流体 |

### 模块职责

- **`config.py`** — 所有可调参数集中于此（Tait EOS 刚度、CFL 步长、碰撞参数、刚体几何等）。
- **`particles.py`** — 纯 Python/NumPy 模块。负责流体初始块、墙壁边界、刚体虚拟粒子模板的生成，以及边界伪质量 ψ 的预计算。
- **`simulation.py`** — `@ti.data_oriented` 类 `FSISimulation`，封装全部 Taichi field 与 kernel。包括 Tait EOS 压力求解、CFL 自适应时间步、刚体力传递与积分。
- **`collision.py`** — CPU 侧刚体碰撞模块。SAT 窄相检测 + Arbiter 接触管理 + Sequential Impulse 迭代求解。移植自 box2d-lite，支持 warm-starting。
- **`main.py`** — 薄入口层。初始化 Taichi → 生成粒子 → 创建模拟器 → 进入渲染 / 事件循环。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行

```bash
cd SPH/fsi
python main.py
```

### 交互

| 操作 | 效果 |
|------|------|
| 左键点击 | 在鼠标位置添加一个新方块 |
| 关闭窗口 | 退出模拟 |

## 参数调节

打开 `config.py`，常见可调项：

| 参数 | 说明 | 默认 |
|------|------|------|
| `PARTICLE_RADIUS` | 粒子半径，越小分辨率越高 | 0.012 |
| `STIFFNESS` | Tait EOS 刚度系数 B | 50000 |
| `EXPONENT` | Tait 指数 γ | 7 |
| `VISCOSITY` | 人工粘性系数 | 0.05 |
| `CFL_FACTOR` | CFL 安全系数 | 0.4 |
| `SUBSTEPS` | 每帧子步数 | 4 |
| `BODY_DENSITY` | 刚体密度（< 1000 会浮） | 500 |
| `MAX_BODIES` | 最大刚体数 | 10 |
| `FRICTION` | Coulomb 摩擦系数 | 0.3 |
| `RESTITUTION` | 碰撞恢复系数 | 0.2 |
| `SI_ITERS` | Sequential Impulse 迭代次数 | 10 |

## 正确性验证

运行静水压力测试，对比 SPH 测量压力与解析解 `p(y) = ρ₀ · g · (H − y)`：

```bash
python validate_hydrostatic.py
```

典型输出 (`r = 0.012`, 3160 流体粒子)：

| 高度 (m) | p_SPH | p_theory | 误差 |
|----------|-------|----------|------|
| 0.096 | 7042 | 7178 | 1.9% |
| 0.387 | 3601 | 4321 | 16.7% |
| 0.672 | 1244 | 1532 | 18.8% |

平均误差 ~16%，符合 WCSPH 弱可压缩方法的预期精度水平。底层精度最高 (~2%)，自由面附近误差较大是 SPH 已知特性。

## 参考文献

- Akinci N., et al. *"Versatile rigid-fluid coupling for incompressible SPH."* ACM TOG, 2012.
- Catto E. *"Iterative Dynamics with Temporal Coherence."* GDC, 2005.
- Monaghan J. J. *"Smoothed particle hydrodynamics."* Annual Review of Astronomy and Astrophysics, 1992.
