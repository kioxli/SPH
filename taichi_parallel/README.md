# taichi_parallel — 高分辨率 FSI（WCSPH + 矩形刚体）

本目录是在 `original` 基础上升级的**高分辨率版本**，将粒子半径从 0.012 m 缩小到 0.005 m，约有 20 000 个粒子，视觉效果更接近真实水体。同时开放了流体初始尺寸、粘性、粒子密度等参数的快捷调节入口。

## 与其他版本的区别

| 项目 | taichi_parallel | original |
|------|-----------------|----------|
| 粒子半径 | 0.005 m（默认） | 0.012 m |
| 粒子数量 | ~20 000 | ~1 000 |
| 粘性默认值 | 0.015（接近真实水） | 0.05（糖浆感） |
| 流体块大小 | `FLUID_X_MAX` / `FLUID_Y_MAX` | 固定 |
| 渲染半径 | 按粒子直径自适应 | 固定 0.003 |
| 刚体碰撞 | SAT + Sequential Impulse | SAT + Sequential Impulse |

## 文件说明

| 文件 | 职责 |
|------|------|
| `main.py` | 入口：Taichi 初始化、窗口、事件循环 |
| `config.py` | 全局参数，含粒子数量档位注释 |
| `particles.py` | 粒子生成与边界伪质量 ψ 预计算 |
| `simulation.py` | Taichi FSI 引擎（WCSPH + 刚体积分） |
| `collision.py` | 刚体碰撞（SAT + Sequential Impulse，CPU） |
| `requirements.txt` | Python 依赖 |

## 运行

```bash
cd SPH/fsi/taichi_parallel
pip install -r requirements.txt
python main.py
```

## 交互

| 操作 | 效果 |
|------|------|
| 左键点击 | 在鼠标位置添加方块（最多 10 个） |
| 关闭窗口 | 退出模拟 |

## 常用参数调节（config.py）

| 参数 | 说明 | 默认 |
|------|------|------|
| `PARTICLE_RADIUS` | 粒子半径，越小越精细 | 0.005 |
| `VISCOSITY` | 粘性系数 | 0.015 |
| `FLUID_X_MAX` | 初始水体右边界 | 1.4 m |
| `FLUID_Y_MAX` | 初始水体上边界 | 1.5 m |
| `BODY_W` / `BODY_H` | 刚体宽/高 | 0.30 / 0.18 m |
| `BODY_DENSITY` | 刚体密度（< 1000 会浮） | 500 kg/m³ |
| `SUBSTEPS` | 每帧子步数 | 8 |

## 硬件建议

| 粒子半径 | 粒子数 | 显存需求 |
|----------|--------|---------|
| 0.012 | ~1 000 | 低 |
| 0.008 | ~3 500 | 低 |
| 0.005 | ~20 000 | ~4 GB |
| 0.004 | ~32 000 | ~8 GB |
| 0.003 | ~57 000 | ~16 GB |
