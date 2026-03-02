"""
simulation.py — Taichi WCSPH + 虚拟粒子法流固耦合引擎 (性能优化版)
═══════════════════════════════════════════════════════════════

核心优化策略:
    · 合并 density + EOS + forces + CFL + advect 为单个 mega-kernel
    · 在流体力循环中通过牛顿第三定律累加刚体受力 (省去独立 kernel)
    · CFL 完全在 GPU 侧计算, 无 CPU↔GPU 同步
    · 合并 sync_rigid + grid_assign 为一次启动
    · 每个 substep 仅 4 次 kernel 启动 + 1 prefix-sum (原 ~16 次)
    · 碰撞求解全部在 GPU 侧完成 (零 CPU↔GPU 数据传输)
"""

import math
import numpy as np
import taichi as ti

from config import (
    DOMAIN_X, DOMAIN_Y,
    PARTICLE_RADIUS, PARTICLE_DIAMETER, SUPPORT_RADIUS,
    M_V0, DENSITY_0, STIFFNESS, EXPONENT, VISCOSITY,
    DT_MAX, DT_MIN, CFL_FACTOR, SOUND_SPEED,
    BODY_W, BODY_H, BODY_MASS, BODY_INV_MASS, BODY_INV_I,
    MAX_BODIES, GRID_SIZE, GRID_NX, GRID_NY, TOTAL_GRIDS,
    BODY0_X, BODY0_Y,
)
from collision import CollisionSolver

PI = 3.14159265358979


@ti.data_oriented
class FSISimulation:
    """WCSPH + 虚拟粒子法流固耦合模拟器 (性能优化版)。"""

    def __init__(self, data: dict):
        self.n_fluid = data['n_fluid']
        self.n_wall  = data['n_wall']
        self.n_rpb   = data['n_rigid_per_body']
        self.n_base  = self.n_fluid + self.n_wall
        self.n_max   = self.n_base + MAX_BODIES * self.n_rpb

        self.h    = SUPPORT_RADIUS
        self.rho0 = DENSITY_0
        self.gx   = GRID_NX
        self.gy   = GRID_NY

        self._create_fields()
        self._load_initial_data(data)
        # ── Taichi 并行碰撞求解器 (直接引用字段, 零拷贝) ──
        self.collision_solver = CollisionSolver(
            BODY_W / 2.0, BODY_H / 2.0,
            self.b_pos, self.b_vel, self.b_ang, self.b_omg)

    # ─────────────────────────────────────────────────────
    #  Fields
    # ─────────────────────────────────────────────────────

    def _create_fields(self):
        N = self.n_max

        self.x      = ti.Vector.field(2, float, N)
        self.v      = ti.Vector.field(2, float, N)
        self.acc    = ti.Vector.field(2, float, N)
        self.rho    = ti.field(float, N)
        self.prs    = ti.field(float, N)
        self.mV     = ti.field(float, N)
        self.ptype  = ti.field(int, N)
        self.bid    = ti.field(int, N)
        self.lpos   = ti.Vector.field(2, float, N)

        # counting-sort 缓冲
        self.x_buf   = ti.Vector.field(2, float, N)
        self.v_buf   = ti.Vector.field(2, float, N)
        self.mV_buf  = ti.field(float, N)
        self.pt_buf  = ti.field(int, N)
        self.bid_buf = ti.field(int, N)
        self.lp_buf  = ti.Vector.field(2, float, N)

        self.g_count     = ti.field(int, TOTAL_GRIDS)
        self.g_count_tmp = ti.field(int, TOTAL_GRIDS)
        self.g_prefix    = ti.algorithms.PrefixSumExecutor(TOTAL_GRIDS)
        self.g_ids       = ti.field(int, N)
        self.g_ids_buf   = ti.field(int, N)
        self.g_ids_new   = ti.field(int, N)

        self.n_active = ti.field(int, shape=())
        self.n_bodies = ti.field(int, shape=())
        self.dt       = ti.field(float, shape=())

        # CFL 归约临时量
        self._cfl_vm = ti.field(float, shape=())
        self._cfl_am = ti.field(float, shape=())

        # 刚体
        self.b_pos = ti.Vector.field(2, float, MAX_BODIES)
        self.b_vel = ti.Vector.field(2, float, MAX_BODIES)
        self.b_ang = ti.field(float, MAX_BODIES)
        self.b_omg = ti.field(float, MAX_BODIES)
        self.b_frc = ti.Vector.field(2, float, MAX_BODIES)
        self.b_trq = ti.field(float, MAX_BODIES)

        self.tpl_lpos = ti.Vector.field(2, float, self.n_rpb)
        self.tpl_mV   = ti.field(float, self.n_rpb)

        # 可视化
        self.x_vis   = ti.Vector.field(2, float, N)
        self.c_vis   = ti.Vector.field(3, float, N)
        self.corners = ti.Vector.field(2, float, MAX_BODIES * 8)
        self.c_off   = ti.Vector.field(2, float, 4)
        self.palette  = ti.Vector.field(3, float, MAX_BODIES)

    # ─────────────────────────────────────────────────────
    #  数据加载
    # ─────────────────────────────────────────────────────

    def _load_initial_data(self, data: dict):
        nf, nw, nrpb = self.n_fluid, self.n_wall, self.n_rpb
        nb = self.n_base
        init_na = nb + nrpb

        body0_world = self._rigid_world_pos(
            data['rigid_local'], BODY0_X, BODY0_Y, 0.0)

        pos_np = np.zeros((self.n_max, 2), dtype=np.float32)
        pos_np[:nf]        = data['fluid_pos']
        pos_np[nf:nb]      = data['wall_pos']
        pos_np[nb:init_na] = body0_world

        mV_np = np.zeros(self.n_max, dtype=np.float32)
        mV_np[:nf]         = M_V0
        mV_np[nf:nb]       = data['psi_wall'] / DENSITY_0
        mV_np[nb:init_na]  = data['psi_rigid'] / DENSITY_0

        pt_np = np.full(self.n_max, -1, dtype=np.int32)
        pt_np[:nf]         = 0
        pt_np[nf:nb]       = 1
        pt_np[nb:init_na]  = 2

        bid_np = np.full(self.n_max, -1, dtype=np.int32)
        bid_np[nb:init_na] = 0

        lp_np = np.zeros((self.n_max, 2), dtype=np.float32)
        lp_np[nb:init_na]  = data['rigid_local']

        self.x.from_numpy(pos_np)
        self.mV.from_numpy(mV_np)
        self.ptype.from_numpy(pt_np)
        self.bid.from_numpy(bid_np)
        self.lpos.from_numpy(lp_np)
        self.tpl_lpos.from_numpy(data['rigid_local'])
        self.tpl_mV.from_numpy(
            (data['psi_rigid'] / DENSITY_0).astype(np.float32))

        self.n_active[None] = init_na
        self.n_bodies[None] = 1
        self.dt[None] = DT_MAX

        self.b_pos[0] = ti.Vector([BODY0_X, BODY0_Y])
        self.b_vel[0] = ti.Vector([0.0, 0.0])
        self.b_ang[0] = 0.0
        self.b_omg[0] = 0.0

        hw, hh = BODY_W / 2.0, BODY_H / 2.0
        self.c_off[0] = ti.Vector([-hw, -hh])
        self.c_off[1] = ti.Vector([ hw, -hh])
        self.c_off[2] = ti.Vector([ hw,  hh])
        self.c_off[3] = ti.Vector([-hw,  hh])

        pal = np.array([
            [1.0,0.30,0.20],[0.20,1.0,0.35],[1.0,0.70,0.10],
            [0.80,0.25,1.0],[0.10,0.90,0.90],[1.0,0.50,0.70],
            [0.50,1.0,0.50],[0.95,0.95,0.15],[0.35,0.55,1.0],
            [1.0,0.40,0.00],
        ], dtype=np.float32)
        self.palette.from_numpy(pal[:MAX_BODIES])
        self._init_dynamic()

    @staticmethod
    def _rigid_world_pos(local, cx, cy, angle):
        c, s = math.cos(angle), math.sin(angle)
        world = np.empty_like(local)
        world[:, 0] = cx + c * local[:, 0] - s * local[:, 1]
        world[:, 1] = cy + s * local[:, 0] + c * local[:, 1]
        return world

    @ti.kernel
    def _init_dynamic(self):
        for i in range(self.n_max):
            self.v[i]   = ti.Vector([0.0, 0.0])
            self.rho[i] = DENSITY_0
            self.prs[i] = 0.0
            self.acc[i]  = ti.Vector([0.0, 0.0])
            self.x_vis[i] = ti.Vector([-1.0, -1.0])
            self.c_vis[i] = ti.Vector([0.0, 0.0, 0.0])
        for i in range(MAX_BODIES * 8):
            self.corners[i] = ti.Vector([-1.0, -1.0])

    # ─────────────────────────────────────────────────────
    #  添加刚体
    # ─────────────────────────────────────────────────────

    @ti.kernel
    def _spawn(self, bid: int, wx: float, wy: float, base: int):
        for k in range(self.n_rpb):
            idx = base + k
            lp = self.tpl_lpos[k]
            self.x[idx]     = ti.Vector([wx + lp[0], wy + lp[1]])
            self.v[idx]     = ti.Vector([0.0, 0.0])
            self.mV[idx]    = self.tpl_mV[k]
            self.ptype[idx] = 2
            self.bid[idx]   = bid
            self.lpos[idx]  = lp
            self.rho[idx]   = DENSITY_0
            self.prs[idx]   = 0.0
        self.b_pos[bid] = ti.Vector([wx, wy])
        self.b_vel[bid] = ti.Vector([0.0, 0.0])
        self.b_ang[bid] = 0.0
        self.b_omg[bid] = 0.0

    def add_body(self, wx: float, wy: float):
        nb = int(self.n_bodies[None])
        if nb >= MAX_BODIES:
            return
        base = int(self.n_active[None])
        self._spawn(nb, wx, wy, base)
        self.n_active[None] = base + self.n_rpb
        self.n_bodies[None] = nb + 1
        print(f"[FSI] 添加刚体 #{nb} @ ({wx:.2f}, {wy:.2f})")

    # ─────────────────────────────────────────────────────
    #  SPH 核函数 (inline)
    # ─────────────────────────────────────────────────────

    @ti.func
    def _cell(self, pos):
        c = ti.cast(pos / GRID_SIZE, int)
        c[0] = ti.max(0, ti.min(c[0], self.gx - 1))
        c[1] = ti.max(0, ti.min(c[1], self.gy - 1))
        return c

    @ti.func
    def _flat(self, cell):
        return cell[0] * self.gy + cell[1]

    @ti.func
    def _nb_range(self, gid):
        s = 0 if gid == 0 else self.g_count[gid - 1]
        return s, self.g_count[gid]

    @ti.func
    def _W(self, r_norm):
        k = 40.0 / 7.0 / PI / (self.h * self.h)
        q = r_norm / self.h
        res = ti.cast(0.0, ti.f32)
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                res = k * (6.0 * q2 * q - 6.0 * q2 + 1.0)
            else:
                res = k * 2.0 * ti.pow(1.0 - q, 3.0)
        return res

    @ti.func
    def _gradW(self, r):
        k = 6.0 * 40.0 / 7.0 / PI / (self.h * self.h)
        rn = r.norm()
        q = rn / self.h
        res = ti.Vector([0.0, 0.0])
        if rn > 1e-5 and q <= 1.0:
            gq = r / (rn * self.h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * gq
            else:
                res = k * (-(1.0 - q) ** 2) * gq
        return res

    # ─────────────────────────────────────────────────────
    #  Kernel 1: 同步刚体粒子 + 网格分配 (合并)
    # ─────────────────────────────────────────────────────

    @ti.kernel
    def _sync_and_assign(self):
        na = self.n_active[None]
        for i in range(na):
            if self.ptype[i] == 2:
                b = self.bid[i]
                ca = ti.cos(self.b_ang[b])
                sa = ti.sin(self.b_ang[b])
                bp = self.b_pos[b]
                lp = self.lpos[i]
                wx = bp[0] + ca * lp[0] - sa * lp[1]
                wy = bp[1] + sa * lp[0] + ca * lp[1]
                self.x[i] = ti.Vector([wx, wy])
                bw = self.b_omg[b]
                self.v[i] = self.b_vel[b] + ti.Vector(
                    [-bw * (wy - bp[1]), bw * (wx - bp[0])])
        for I in ti.grouped(self.g_count):
            self.g_count[I] = 0
        for i in range(na):
            gid = self._flat(self._cell(self.x[i]))
            self.g_ids[i] = gid
            ti.atomic_add(self.g_count[gid], 1)
        for I in ti.grouped(self.g_count):
            self.g_count_tmp[I] = self.g_count[I]

    # ─────────────────────────────────────────────────────
    #  Kernel 2: 网格排序 (counting sort)
    # ─────────────────────────────────────────────────────

    @ti.kernel
    def _grid_sort(self):
        na = self.n_active[None]
        for i in range(na):
            I = na - 1 - i
            gid = self.g_ids[I]
            base = 0 if gid == 0 else self.g_count[gid - 1]
            self.g_ids_new[I] = (
                ti.atomic_sub(self.g_count_tmp[gid], 1) - 1 + base)
        for i in range(na):
            ni = self.g_ids_new[i]
            self.g_ids_buf[ni] = self.g_ids[i]
            self.x_buf[ni]   = self.x[i]
            self.v_buf[ni]   = self.v[i]
            self.mV_buf[ni]  = self.mV[i]
            self.pt_buf[ni]  = self.ptype[i]
            self.bid_buf[ni] = self.bid[i]
            self.lp_buf[ni]  = self.lpos[i]
        for i in range(na):
            self.g_ids[i] = self.g_ids_buf[i]
            self.x[i]     = self.x_buf[i]
            self.v[i]     = self.v_buf[i]
            self.mV[i]    = self.mV_buf[i]
            self.ptype[i] = self.pt_buf[i]
            self.bid[i]   = self.bid_buf[i]
            self.lpos[i]  = self.lp_buf[i]

    # ─────────────────────────────────────────────────────
    #  Kernel 3: MEGA 物理步
    #  density → EOS → forces(+body reaction) → CFL → advect → body_vel
    #  6 个顺序 for-loop 合并为 1 次 kernel 启动, 0 次 GPU 同步
    # ─────────────────────────────────────────────────────

    @ti.kernel
    def _physics_step(self):
        na = self.n_active[None]
        nb = self.n_bodies[None]
        h  = self.h

        # ── Loop 1: 密度 + Tait EOS ────────────────────
        for i in range(na):
            if self.ptype[i] != 0:
                continue
            d = self.mV[i] * self._W(0.0)
            center = self._cell(self.x[i])
            for off in ti.grouped(ti.ndrange((-1, 2), (-1, 2))):
                nc = center + off
                if 0 <= nc[0] < self.gx and 0 <= nc[1] < self.gy:
                    s, e = self._nb_range(self._flat(nc))
                    for j in range(s, e):
                        if i != j:
                            rn = (self.x[i] - self.x[j]).norm()
                            if rn < h:
                                d += self.mV[j] * self._W(rn)
            rho_i = ti.max(d * self.rho0, self.rho0)
            self.rho[i] = rho_i
            self.prs[i] = STIFFNESS * (ti.pow(rho_i / self.rho0, EXPONENT) - 1.0)

        # ── Loop 2: 重置刚体力累加器 ───────────────────
        for bid_val in range(nb):
            self.b_frc[bid_val] = ti.Vector([0.0, 0.0])
            self.b_trq[bid_val] = 0.0

        # ── Loop 3: 力计算 (重力 + 粘性 + 压力 + 刚体反力) ──
        for i in range(na):
            if self.ptype[i] != 0:
                continue
            dv = ti.Vector([0.0, -9.81])
            rho_i = self.rho[i]
            dpi = self.prs[i] / (rho_i * rho_i)
            center = self._cell(self.x[i])
            for off in ti.grouped(ti.ndrange((-1, 2), (-1, 2))):
                nc = center + off
                if 0 <= nc[0] < self.gx and 0 <= nc[1] < self.gy:
                    s, e = self._nb_range(self._flat(nc))
                    for j in range(s, e):
                        if i == j:
                            continue
                        r = self.x[i] - self.x[j]
                        rn = r.norm()
                        if rn >= h:
                            continue
                        gw = self._gradW(r)
                        mj = self.rho0 * self.mV[j]
                        pt_j = self.ptype[j]

                        if pt_j == 0:
                            vr = (self.v[i] - self.v[j]).dot(r)
                            den = rn * rn + 0.01 * h * h
                            dv += (8.0 * VISCOSITY * self.mV[j]
                                   * self.rho0 / self.rho[j]
                                   * vr / den * gw)
                            dpj = self.prs[j] / (self.rho[j] * self.rho[j])
                            dv += -mj * (dpi + dpj) * gw
                        elif pt_j == 1:
                            dv += -mj * dpi * gw
                        else:
                            dv += -mj * dpi * gw
                            f_on_body = (self.rho0 * self.rho0
                                         * self.mV[i] * self.mV[j]
                                         * dpi * gw)
                            body_id = self.bid[j]
                            ti.atomic_add(self.b_frc[body_id], f_on_body)
                            rj = self.x[j] - self.b_pos[body_id]
                            ti.atomic_add(self.b_trq[body_id],
                                          rj[0] * f_on_body[1]
                                          - rj[1] * f_on_body[0])
            self.acc[i] = dv

        # ── Loop 4: CFL 归约 (全 GPU, 无同步) ──────────
        self._cfl_vm[None] = 0.0
        self._cfl_am[None] = 0.0
        for i in range(na):
            if self.ptype[i] == 0:
                ti.atomic_max(self._cfl_vm[None], self.v[i].norm())
                ti.atomic_max(self._cfl_am[None], self.acc[i].norm())
        for _ in range(1):
            vm = self._cfl_vm[None]
            am = self._cfl_am[None]
            new_dt = ti.cast(DT_MAX, ti.f32)
            dt_v = CFL_FACTOR * SUPPORT_RADIUS / (SOUND_SPEED + vm)
            if dt_v < new_dt:
                new_dt = dt_v
            if am > 1.0e-6:
                dt_a = CFL_FACTOR * ti.sqrt(SUPPORT_RADIUS / am)
                if dt_a < new_dt:
                    new_dt = dt_a
            if new_dt < DT_MIN:
                new_dt = DT_MIN
            self.dt[None] = new_dt

        # ── Loop 5: 流体平流 + 边界钳位 ────────────────
        for i in range(na):
            if self.ptype[i] != 0:
                continue
            dt = self.dt[None]
            self.v[i] += dt * self.acc[i]
            self.x[i] += dt * self.v[i]
            if self.x[i][0] < PARTICLE_DIAMETER:
                self.x[i][0] = PARTICLE_DIAMETER
                self.v[i][0] *= -0.3
            if self.x[i][0] > DOMAIN_X - PARTICLE_DIAMETER:
                self.x[i][0] = DOMAIN_X - PARTICLE_DIAMETER
                self.v[i][0] *= -0.3
            if self.x[i][1] < PARTICLE_DIAMETER:
                self.x[i][1] = PARTICLE_DIAMETER
                self.v[i][1] *= -0.3
            if self.x[i][1] > DOMAIN_Y - PARTICLE_DIAMETER:
                self.x[i][1] = DOMAIN_Y - PARTICLE_DIAMETER
                self.v[i][1] *= -0.3

        # ── Loop 6: 刚体速度积分 ───────────────────────
        for bid_val in range(nb):
            dt = self.dt[None]
            self.b_frc[bid_val] += ti.Vector([0.0, -9.81]) * BODY_MASS
            self.b_vel[bid_val] += dt * BODY_INV_MASS * self.b_frc[bid_val]
            self.b_omg[bid_val] += dt * BODY_INV_I * self.b_trq[bid_val]
            self.b_vel[bid_val] *= 0.998
            self.b_omg[bid_val] *= 0.998

    # ─────────────────────────────────────────────────────
    #  Kernel 4: 刚体位置积分 + 钳位 (碰撞求解后)
    # ─────────────────────────────────────────────────────

    @ti.kernel
    def _finalize_bodies(self):
        dt = self.dt[None]
        nb = self.n_bodies[None]
        margin = 2.0 * PARTICLE_DIAMETER
        ext = ti.max(BODY_W, BODY_H) * 0.5
        lo = margin + ext
        hi_x = DOMAIN_X - margin - ext
        hi_y = DOMAIN_Y - margin - ext
        for bid_val in range(nb):
            self.b_pos[bid_val] += dt * self.b_vel[bid_val]
            self.b_ang[bid_val] += dt * self.b_omg[bid_val]
            if self.b_pos[bid_val][0] < lo:
                self.b_pos[bid_val][0] = lo
                self.b_vel[bid_val][0] *= -0.3
            if self.b_pos[bid_val][0] > hi_x:
                self.b_pos[bid_val][0] = hi_x
                self.b_vel[bid_val][0] *= -0.3
            if self.b_pos[bid_val][1] < lo:
                self.b_pos[bid_val][1] = lo
                self.b_vel[bid_val][1] *= -0.3
            if self.b_pos[bid_val][1] > hi_y:
                self.b_pos[bid_val][1] = hi_y
                self.b_vel[bid_val][1] *= -0.3

    # ─────────────────────────────────────────────────────
    #  刚体间碰撞 (全 Taichi GPU, 零 CPU↔GPU 传输)
    # ─────────────────────────────────────────────────────

    def _resolve_collisions(self):
        nb = int(self.n_bodies[None])
        if nb < 2:
            return
        self.collision_solver.solve(self.n_bodies, self.dt)

    # ─────────────────────────────────────────────────────
    #  完整子步: 4 kernel + 1 prefix-sum + Taichi collision
    # ─────────────────────────────────────────────────────

    def substep(self):
        self._sync_and_assign()                     # K1
        self.g_prefix.run(self.g_count)             # prefix-sum
        self._grid_sort()                           # K2
        self._physics_step()                        # K3 (mega)
        self._resolve_collisions()                  # Taichi GPU
        self._finalize_bodies()                     # K4

    # ─────────────────────────────────────────────────────
    #  可视化 (合并 vis + outlines)
    # ─────────────────────────────────────────────────────

    @ti.kernel
    def update_vis(self):
        na = self.n_active[None]
        for i in range(na):
            self.x_vis[i] = ti.Vector(
                [self.x[i][0] / DOMAIN_X, self.x[i][1] / DOMAIN_Y])
            t = self.ptype[i]
            if t == 0:
                # 速度驱动的水色渐变:
                #   静止 → 深海蓝 (0.02, 0.12, 0.45)
                #   中速 → 海洋蓝 (0.05, 0.50, 0.80)
                #   高速 → 浅青白 (0.40, 0.88, 0.98)
                spd   = self.v[i].norm()
                t_val = ti.min(spd / 3.0, 1.0)   # 以 3 m/s 为满速
                r = 0.02 + 0.38 * t_val
                g = 0.12 + 0.76 * t_val
                b = 0.45 + 0.53 * t_val
                self.c_vis[i] = ti.Vector([r, g, b])
            elif t == 1:
                self.c_vis[i] = ti.Vector([0.45, 0.45, 0.45])
            elif t == 2:
                self.c_vis[i] = self.palette[self.bid[i]]
            else:
                self.c_vis[i] = ti.Vector([0.0, 0.0, 0.0])
        nb = self.n_bodies[None]
        for bid_val in range(MAX_BODIES):
            if bid_val < nb:
                ca = ti.cos(self.b_ang[bid_val])
                sa = ti.sin(self.b_ang[bid_val])
                bp = self.b_pos[bid_val]
                for c in range(4):
                    lp  = self.c_off[c]
                    lp2 = self.c_off[(c + 1) % 4]
                    w1 = ti.Vector([
                        bp[0] + ca * lp[0]  - sa * lp[1],
                        bp[1] + sa * lp[0]  + ca * lp[1]])
                    w2 = ti.Vector([
                        bp[0] + ca * lp2[0] - sa * lp2[1],
                        bp[1] + sa * lp2[0] + ca * lp2[1]])
                    self.corners[bid_val*8+c*2]   = ti.Vector(
                        [w1[0]/DOMAIN_X, w1[1]/DOMAIN_Y])
                    self.corners[bid_val*8+c*2+1] = ti.Vector(
                        [w2[0]/DOMAIN_X, w2[1]/DOMAIN_Y])
            else:
                for c in range(8):
                    self.corners[bid_val * 8 + c] = ti.Vector([-1.0, -1.0])
