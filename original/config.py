"""
config.py — 全局参数配置
═══════════════════════════════════════════════════════════════

所有可调参数集中于此，修改本文件即可调整模拟行为，无需改动其他模块。

参数分组:
    · 计算域          — 模拟空间尺寸
    · SPH 离散化      — 粒子半径、支撑域、初始体积
    · 物态方程 (EOS)  — Tait 方程刚度 / 指数
    · 粘性            — 人工粘性系数
    · 时间步进 (CFL)  — 自适应步长范围、CFL 系数
    · 刚体            — 尺寸、密度、质量参数
    · 多刚体          — 最大刚体数
    · 碰撞            — 摩擦 / 恢复系数、SI 迭代
    · 网格            — 邻居搜索网格
    · 墙壁            — 边界粒子层数
"""

import math

# ── 计算域 ────────────────────────────────────────────────
DOMAIN_X = 2.0                          # 域宽 (m)
DOMAIN_Y = 2.0                          # 域高 (m)

# ── SPH 离散化 ────────────────────────────────────────────
PARTICLE_RADIUS   = 0.012               # 粒子半径 r
PARTICLE_DIAMETER = 2.0 * PARTICLE_RADIUS
SUPPORT_RADIUS    = 4.0 * PARTICLE_RADIUS  # 核函数支撑半径 h = 4r
M_V0              = 0.8 * PARTICLE_DIAMETER ** 2  # 流体粒子初始体积

# ── 物态方程 (Tait EOS) ──────────────────────────────────
DENSITY_0  = 1000.0                     # 静止密度 ρ₀ (kg/m³)
STIFFNESS  = 50000.0                    # 刚度系数 B
EXPONENT   = 7.0                        # Tait 指数 γ

# ── 粘性 ──────────────────────────────────────────────────
VISCOSITY = 0.05                        # 人工粘性系数

# ── 时间步进 (CFL 自适应) ─────────────────────────────────
SOUND_SPEED = float(math.sqrt(STIFFNESS * EXPONENT / DENSITY_0))
DT_MAX     = 5e-4                       # CFL 上限 ~1.0ms, 此值为其 49% (安全)
DT_MIN     = 1e-5                       # 最小时间步长 (s)
CFL_FACTOR = 0.4                        # CFL 安全系数
SUBSTEPS   = 6                          # 每帧子步数 (视觉速度 = SUBSTEPS × dt)

# ── 刚体 (所有方块共享相同几何) ───────────────────────────
BODY_W       = 0.3                      # 宽 (m)
BODY_H       = 0.2                      # 高 (m)
BODY_DENSITY = 500.0                    # 密度 (kg/m³), 小于 ρ₀ → 会浮

BODY_MASS     = float(BODY_DENSITY * BODY_W * BODY_H)
BODY_INV_MASS = 1.0 / BODY_MASS
BODY_I        = BODY_MASS * (BODY_W ** 2 + BODY_H ** 2) / 12.0  # 转动惯量
BODY_INV_I    = 1.0 / BODY_I

# ── 多刚体 ────────────────────────────────────────────────
MAX_BODIES = 10                         # 场景最多支持的刚体数量

# ── 网格 (用于邻居搜索) ──────────────────────────────────
GRID_SIZE   = SUPPORT_RADIUS
GRID_NX     = int(math.ceil(DOMAIN_X / GRID_SIZE))
GRID_NY     = int(math.ceil(DOMAIN_Y / GRID_SIZE))
TOTAL_GRIDS = GRID_NX * GRID_NY

# ── 墙壁 ─────────────────────────────────────────────────
WALL_LAYERS = 2                         # 边界粒子层数 (底 / 左 / 右)

# ── 碰撞 ─────────────────────────────────────────────────
FRICTION    = 0.3                       # Coulomb 摩擦系数
RESTITUTION = 0.2                       # 恢复系数
SI_ITERS    = 10                        # Sequential Impulse 迭代次数

# ── 初始场景 ──────────────────────────────────────────────
BODY0_X = 1.2                           # 第 0 号刚体初始 x
BODY0_Y = 0.3                           # 第 0 号刚体初始 y
