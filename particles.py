"""
particles.py — 粒子生成与边界伪质量计算
═══════════════════════════════════════════════════════════════

职责:
    1. 生成流体粒子初始位置 (溃坝块)
    2. 生成墙壁边界粒子 (底/左/右, 多层)
    3. 生成刚体虚拟粒子模板 (矩形表面采样)
    4. 计算边界粒子伪质量 ψ  (Akinci 2012)

所有函数均为纯 Python / NumPy, 不依赖 Taichi。
"""

import math
import numpy as np

from config import (
    DOMAIN_X, DOMAIN_Y,
    PARTICLE_DIAMETER, SUPPORT_RADIUS, DENSITY_0,
    BODY_W, BODY_H, WALL_LAYERS,
)


# ═══════════════════════════════════════════════════════════
#  CPU 版 Cubic-Spline 核函数 (仅用于 ψ 初始化)
# ═══════════════════════════════════════════════════════════

def _W_cpu(r_norm: float) -> float:
    """Cubic-Spline 核函数 W(r, h), 标量版本"""
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
    """
    计算一组边界粒子的伪质量 ψ (向量化)。

        ψ_j = ρ₀ / [ W(0) + Σ_{k≠j} W(|x_j − x_k|) ]

    Parameters
    ----------
    positions : (N, 2) float32
        边界粒子坐标 (墙壁用世界坐标, 刚体用局部坐标)。

    Returns
    -------
    psi : (N,) float32
    """
    n = len(positions)
    diffs = positions[:, None, :] - positions[None, :, :]   # (N, N, 2)
    dists = np.linalg.norm(diffs, axis=2)                   # (N, N)

    h = SUPPORT_RADIUS
    k = 40.0 / 7.0 / math.pi / (h * h)
    q = dists / h

    ws = np.zeros_like(q)
    m1 = q <= 0.5
    m2 = (q > 0.5) & (q <= 1.0)
    ws[m1] = k * (6.0 * q[m1] ** 3 - 6.0 * q[m1] ** 2 + 1.0)
    ws[m2] = k * 2.0 * (1.0 - q[m2]) ** 3
    np.fill_diagonal(ws, 0.0)

    W0 = _W_cpu(0.0)
    delta = W0 + np.sum(ws, axis=1)
    return (DENSITY_0 / np.maximum(delta, 1e-6)).astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  流体粒子
# ═══════════════════════════════════════════════════════════

def generate_fluid(x_min: float = None, x_max: float = 0.7,
                   y_min: float = None, y_max: float = 1.0):
    """
    在矩形区域内均匀采样流体粒子 (溃坝初始块)。

    Returns
    -------
    positions : (N, 2) float32
    count     : int
    """
    dx = PARTICLE_DIAMETER
    margin = dx * (WALL_LAYERS + 0.5)
    if x_min is None:
        x_min = margin
    if y_min is None:
        y_min = margin

    xs = np.arange(x_min, x_max, dx)
    ys = np.arange(y_min, y_max, dx)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    pos = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    return pos, int(pos.shape[0])


# ═══════════════════════════════════════════════════════════
#  墙壁边界粒子
# ═══════════════════════════════════════════════════════════

def generate_walls():
    """
    生成底部 / 左侧 / 右侧墙壁的多层边界粒子。

    Returns
    -------
    positions : (N, 2) float32
    count     : int
    """
    dx = PARTICLE_DIAMETER
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
#  刚体虚拟粒子模板
# ═══════════════════════════════════════════════════════════

def generate_rigid_template():
    """
    沿矩形刚体表面均匀采样虚拟粒子, 返回 *局部* 坐标。
    所有刚体共享同一模板, 实际世界坐标由刚体姿态变换得到。

    Returns
    -------
    local_positions : (N, 2) float32
    count           : int
    """
    dx = PARTICLE_DIAMETER
    hw, hh = BODY_W / 2.0, BODY_H / 2.0
    pts = []

    for xi in np.arange(-hw, hw + dx * 0.5, dx):
        pts.append([xi, -hh])      # 底边
        pts.append([xi,  hh])      # 顶边

    for yi in np.arange(-hh + dx, hh, dx):
        pts.append([-hw, yi])      # 左边 (不含角点)
        pts.append([ hw, yi])      # 右边

    pos = np.array(pts, dtype=np.float32)
    return pos, int(pos.shape[0])


# ═══════════════════════════════════════════════════════════
#  便捷: 一次性生成全部初始数据
# ═══════════════════════════════════════════════════════════

def generate_all():
    """
    生成流体、墙壁、刚体模板, 并计算边界 ψ。

    Returns
    -------
    data : dict  包含以下键:
        fluid_pos, n_fluid,
        wall_pos,  n_wall,  psi_wall,
        rigid_local, n_rigid_per_body, psi_rigid
    """
    fluid_pos,   n_fluid          = generate_fluid()
    wall_pos,    n_wall           = generate_walls()
    rigid_local, n_rigid_per_body = generate_rigid_template()

    psi_wall  = compute_boundary_psi(wall_pos)
    psi_rigid = compute_boundary_psi(rigid_local)

    print(f"[particles] 流体: {n_fluid}  墙壁: {n_wall}  "
          f"每刚体: {n_rigid_per_body}")
    print(f"[particles] ψ_wall ∈ [{psi_wall.min():.1f}, {psi_wall.max():.1f}]  "
          f"ψ_rigid ∈ [{psi_rigid.min():.1f}, {psi_rigid.max():.1f}]")

    return dict(
        fluid_pos=fluid_pos, n_fluid=n_fluid,
        wall_pos=wall_pos,   n_wall=n_wall,   psi_wall=psi_wall,
        rigid_local=rigid_local,
        n_rigid_per_body=n_rigid_per_body,
        psi_rigid=psi_rigid,
    )
