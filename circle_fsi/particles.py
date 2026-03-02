"""
particles.py — 粒子生成与边界伪质量计算 (圆形刚体版)
═══════════════════════════════════════════════════════════════

与矩形版的唯一区别: generate_rigid_template 沿圆周均匀采样虚拟粒子。
"""

import math
import numpy as np

from config import (
    DOMAIN_X, DOMAIN_Y,
    PARTICLE_DIAMETER, SUPPORT_RADIUS, DENSITY_0,
    BODY_RADIUS, WALL_LAYERS,
    FLUID_X_MAX, FLUID_Y_MAX,
    BODY0_X, BODY0_Y,
)


# ═══════════════════════════════════════════════════════════
#  CPU 版 Cubic-Spline 核函数 (仅用于 ψ 初始化)
# ═══════════════════════════════════════════════════════════

def _W_cpu(r_norm: float) -> float:
    h = SUPPORT_RADIUS
    k = 40.0 / 7.0 / math.pi / (h * h)
    q = r_norm / h
    if q > 1.0:
        return 0.0
    if q <= 0.5:
        return k * (6.0 * q ** 3 - 6.0 * q ** 2 + 1.0)
    return k * 2.0 * (1.0 - q) ** 3


# ═══════════════════════════════════════════════════════════
#  边界伪质量 ψ
# ═══════════════════════════════════════════════════════════

def compute_boundary_psi(positions: np.ndarray) -> np.ndarray:
    n = len(positions)
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)

    h = SUPPORT_RADIUS
    k = 40.0 / 7.0 / math.pi / (h * h)
    q = dists / h

    ws = np.zeros_like(q)
    m1 = q <= 0.5
    m2 = (q > 0.5) & (q <= 1.0)
    ws[m1] = k * (6.0 * q[m1] ** 3 - 6.0 * q[m1] ** 2 + 1.0)
    ws[m2] = k * 2.0 * (1.0 - q[m2]) ** 3
    np.fill_diagonal(ws, 0.0)

    W0    = _W_cpu(0.0)
    delta = W0 + np.sum(ws, axis=1)
    return (DENSITY_0 / np.maximum(delta, 1e-6)).astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  流体粒子
# ═══════════════════════════════════════════════════════════

def generate_fluid(x_min: float = None, x_max: float = None,
                   y_min: float = None, y_max: float = None):
    dx     = PARTICLE_DIAMETER
    margin = dx * (WALL_LAYERS + 0.5)
    if x_min is None: x_min = margin
    if y_min is None: y_min = margin
    if x_max is None: x_max = FLUID_X_MAX
    if y_max is None: y_max = FLUID_Y_MAX

    xs = np.arange(x_min, x_max, dx)
    ys = np.arange(y_min, y_max, dx)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    pos = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    # 排除初始圆形刚体内部的流体粒子
    # 剔除半径: 圆半径 + 1 个粒子直径 (留出安全间距)
    excl_r2 = (BODY_RADIUS + PARTICLE_DIAMETER) ** 2
    d2 = (pos[:, 0] - BODY0_X) ** 2 + (pos[:, 1] - BODY0_Y) ** 2
    pos = pos[d2 > excl_r2]

    return pos, int(pos.shape[0])


# ═══════════════════════════════════════════════════════════
#  墙壁边界粒子
# ═══════════════════════════════════════════════════════════

def generate_walls():
    dx  = PARTICLE_DIAMETER
    pts = []

    for layer in range(WALL_LAYERS):
        y_w = dx * (0.5 + layer)
        for xi in np.arange(dx * 0.5, DOMAIN_X, dx):
            pts.append([xi, y_w])

    y_start = dx * (0.5 + WALL_LAYERS)
    for layer in range(WALL_LAYERS):
        x_left  = dx * (0.5 + layer)
        x_right = DOMAIN_X - dx * (0.5 + layer)
        for yi in np.arange(y_start, DOMAIN_Y, dx):
            pts.append([x_left,  yi])
            pts.append([x_right, yi])

    pos = np.array(pts, dtype=np.float32)
    return pos, int(pos.shape[0])


# ═══════════════════════════════════════════════════════════
#  圆形刚体虚拟粒子模板
# ═══════════════════════════════════════════════════════════

def generate_rigid_template():
    """
    沿圆周均匀采样虚拟粒子, 返回局部坐标 (圆心为原点)。

    采样数 = 圆周长 / 粒子直径, 保证粒子间距与流体一致。
    """
    dx    = PARTICLE_DIAMETER
    circ  = 2.0 * math.pi * BODY_RADIUS
    n_pts = max(6, int(circ / dx))          # 至少 6 个粒子

    angles = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    pts = np.stack([
        BODY_RADIUS * np.cos(angles),
        BODY_RADIUS * np.sin(angles),
    ], axis=1).astype(np.float32)

    return pts, n_pts


# ═══════════════════════════════════════════════════════════
#  便捷: 一次性生成全部初始数据
# ═══════════════════════════════════════════════════════════

def generate_all():
    fluid_pos,   n_fluid          = generate_fluid()
    wall_pos,    n_wall           = generate_walls()
    rigid_local, n_rigid_per_body = generate_rigid_template()

    psi_wall  = compute_boundary_psi(wall_pos)
    psi_rigid = compute_boundary_psi(rigid_local)

    print(f"[particles] 流体: {n_fluid}  墙壁: {n_wall}  "
          f"每圆: {n_rigid_per_body}")
    print(f"[particles] ψ_wall ∈ [{psi_wall.min():.1f}, {psi_wall.max():.1f}]  "
          f"ψ_rigid ∈ [{psi_rigid.min():.1f}, {psi_rigid.max():.1f}]")

    return dict(
        fluid_pos=fluid_pos, n_fluid=n_fluid,
        wall_pos=wall_pos,   n_wall=n_wall,   psi_wall=psi_wall,
        rigid_local=rigid_local,
        n_rigid_per_body=n_rigid_per_body,
        psi_rigid=psi_rigid,
    )
