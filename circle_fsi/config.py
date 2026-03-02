"""
config.py — 圆形刚体 FSI 全局参数配置
═══════════════════════════════════════════════════════════════

本版本特点:
    · 刚体形状: 矩形 → 圆形
    · 无刚体间碰撞 (已移除 CollisionSolver)
    · 其余参数与 taichi_parallel 版本保持一致
"""

import math

# ── 计算域 ────────────────────────────────────────────────
DOMAIN_X = 2.0
DOMAIN_Y = 2.0

# ── SPH 离散化 ────────────────────────────────────────────
#   0.005  ≈  20 000 粒子  (默认, 16 GB 流畅)
#   0.004  ≈  32 000 粒子
#   0.003  ≈  57 000 粒子
PARTICLE_RADIUS   = 0.005
PARTICLE_DIAMETER = 2.0 * PARTICLE_RADIUS
SUPPORT_RADIUS    = 4.0 * PARTICLE_RADIUS
M_V0              = 0.8 * PARTICLE_DIAMETER ** 2

# ── 物态方程 (Tait EOS) ──────────────────────────────────
DENSITY_0  = 1000.0
STIFFNESS  = 50000.0
EXPONENT   = 7.0

# ── 粘性 ──────────────────────────────────────────────────
VISCOSITY = 0.015

# ── 时间步进 (CFL 自适应) ─────────────────────────────────
SOUND_SPEED = float(math.sqrt(STIFFNESS * EXPONENT / DENSITY_0))
DT_MAX     = 2e-4
DT_MIN     = 1e-6
CFL_FACTOR = 0.4
SUBSTEPS   = 8

# ── 初始流体块范围 ────────────────────────────────────────
FLUID_X_MAX = 1.4
FLUID_Y_MAX = 1.5

# ── 圆形刚体 ─────────────────────────────────────────────
# ★ 修改 BODY_RADIUS 即可改变圆的大小
#   0.08  → 小球
#   0.12  → 默认
#   0.18  → 大球
BODY_RADIUS  = 0.12                     # 圆半径 (m)
BODY_DENSITY = 500.0                    # 密度 (kg/m³), < ρ₀ → 会浮

BODY_MASS     = float(BODY_DENSITY * math.pi * BODY_RADIUS ** 2)
BODY_INV_MASS = 1.0 / BODY_MASS
BODY_I        = 0.5 * BODY_MASS * BODY_RADIUS ** 2   # 圆盘: I = ½mr²
BODY_INV_I    = 1.0 / BODY_I

# ── 多刚体 ────────────────────────────────────────────────
MAX_BODIES = 10

# ── 网格 (邻居搜索) ───────────────────────────────────────
GRID_SIZE   = SUPPORT_RADIUS
GRID_NX     = int(math.ceil(DOMAIN_X / GRID_SIZE))
GRID_NY     = int(math.ceil(DOMAIN_Y / GRID_SIZE))
TOTAL_GRIDS = GRID_NX * GRID_NY

# ── 墙壁 ─────────────────────────────────────────────────
WALL_LAYERS = 2

# ── 初始场景 ──────────────────────────────────────────────
BODY0_X = 1.3
BODY0_Y = 0.3

# ── 圆形轮廓渲染 ──────────────────────────────────────────
N_CIRCLE_SEGS = 36                      # 轮廓线段数 (越多越圆滑)
