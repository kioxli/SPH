# rigid_body — 纯刚体模拟 Demo

本目录是**不含流体**的纯刚体物理模拟，移植自 box2d-lite，使用 Sequential Impulse 约束求解器处理碰撞与摩擦。默认场景为 15 个方块组成的金字塔，可通过鼠标投入新刚体进行交互。

## 用途

- 独立验证刚体碰撞算法（SAT + Sequential Impulse）的正确性
- 学习 box2d-lite 风格的约束求解器实现
- 作为 FSI 项目碰撞模块的独立测试平台

## 文件说明

| 文件 | 职责 |
|------|------|
| `main.py` | 入口：场景构建、事件循环、Taichi 渲染 |
| `config.py` | 物理参数（重力、摩擦、恢复系数、时间步） |
| `physics.py` | 刚体物理引擎（`Body`、`StaticBody`、`World`） |

## 运行

```bash
cd SPH/fsi/rigid_body
python main.py
```

> 本模块仅使用 CPU（`ti.init(arch=ti.cpu)`），无需 GPU。

## 交互

| 操作 | 效果 |
|------|------|
| 左键点击 | 在鼠标位置投入新矩形刚体 |
| 空格 | 暂停 / 继续 |
| R | 重置场景（恢复初始金字塔） |
| ESC / 关闭窗口 | 退出 |

## 颜色说明

| 颜色 | 含义 |
|------|------|
| 浅灰 | 静态边界（地面、左墙、右墙） |
| 彩色 | 动态刚体（每个不同颜色，循环使用 8 色调色板） |

## 参数说明（config.py）

| 参数 | 说明 | 默认 |
|------|------|------|
| `DOMAIN_W` / `DOMAIN_H` | 场景宽高 | 14 / 9 m |
| `GRAVITY` | 重力加速度 | -10 m/s² |
| `FRICTION` | Coulomb 摩擦系数 | 0.4 |
| `RESTITUTION` | 碰撞恢复系数（0=完全非弹性） | 0.25 |
| `SI_ITERS` | Sequential Impulse 迭代次数 | 20 |
| `DT` | 物理时间步 | 1/60 s |
| `SUBSTEPS` | 每帧子步数 | 6 |
| `MAX_BODIES` | 动态刚体上限 | 40 |

## 算法说明

| 模块 | 方法 |
|------|------|
| 碰撞检测 | SAT（分离轴定理）窄相检测 |
| 约束求解 | Sequential Impulse（迭代冲量法） |
| Warm-starting | 保留上一帧冲量加速收敛 |
| 积分 | 半隐式欧拉（速度先更新，位置后更新） |
