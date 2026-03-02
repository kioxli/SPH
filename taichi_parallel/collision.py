"""
collision.py — Taichi 并行 OBB SAT + Sequential Impulse 碰撞求解器
═══════════════════════════════════════════════════════════════

已全面迁移至 Taichi GPU kernel，消除 CPU↔GPU 数据传输瓶颈。

设计:
    · _detect_all  : 并行 SAT 碰撞检测 (各对互不干扰)
    · _pre_step    : 串行 warm-start + 有效质量预计算 (Gauss-Seidel)
    · _si_iter     : 串行 Sequential Impulse 单次迭代 (Gauss-Seidel)

b_pos / b_vel / b_ang / b_omg 直接引用 FSISimulation 的 Taichi 字段，
无需 CPU 读回 / 写入，消除每帧同步开销。

参考: Erin Catto, "Iterative Dynamics with Temporal Coherence", GDC 2005
"""

import numpy as np
import taichi as ti

from config import (
    FRICTION, RESTITUTION, SI_ITERS,
    MAX_BODIES, BODY_INV_MASS, BODY_INV_I,
)

MAX_PAIRS = MAX_BODIES * (MAX_BODIES - 1) // 2   # 45 (MAX_BODIES=10)
MAX_CP    = 2                                      # 每对最多 2 个接触点


@ti.data_oriented
class CollisionSolver:
    """
    全 Taichi 刚体间碰撞求解器。

    b_pos, b_vel, b_ang, b_omg 直接引用 FSISimulation 的字段 (零拷贝)。
    """

    def __init__(self, hw: float, hh: float,
                 b_pos, b_vel, b_ang, b_omg):
        self.hw = float(hw)
        self.hh = float(hh)

        # 直接引用模拟器字段
        self.b_pos = b_pos
        self.b_vel = b_vel
        self.b_ang = b_ang
        self.b_omg = b_omg

        # 预计算对索引 (不变量)
        self.pair_i = ti.field(ti.i32, MAX_PAIRS)
        self.pair_j = ti.field(ti.i32, MAX_PAIRS)
        _pi = np.zeros(MAX_PAIRS, np.int32)
        _pj = np.zeros(MAX_PAIRS, np.int32)
        k = 0
        for a in range(MAX_BODIES):
            for b in range(a + 1, MAX_BODIES):
                _pi[k] = a
                _pj[k] = b
                k += 1
        self.pair_i.from_numpy(_pi)
        self.pair_j.from_numpy(_pj)

        # 碰撞检测输出
        self.n_contacts = ti.field(ti.i32, MAX_PAIRS)
        self.c_pos      = ti.Vector.field(2, ti.f32, (MAX_PAIRS, MAX_CP))
        self.c_normal   = ti.Vector.field(2, ti.f32, (MAX_PAIRS, MAX_CP))
        self.c_sep      = ti.field(ti.f32, (MAX_PAIRS, MAX_CP))

        # SI 求解器中间量
        self.c_r1     = ti.Vector.field(2, ti.f32, (MAX_PAIRS, MAX_CP))
        self.c_r2     = ti.Vector.field(2, ti.f32, (MAX_PAIRS, MAX_CP))
        self.c_mass_n = ti.field(ti.f32, (MAX_PAIRS, MAX_CP))
        self.c_mass_t = ti.field(ti.f32, (MAX_PAIRS, MAX_CP))
        self.c_bias   = ti.field(ti.f32, (MAX_PAIRS, MAX_CP))

        # 累积冲量 (warm-start, 跨帧保留)
        self.c_pn = ti.field(ti.f32, (MAX_PAIRS, MAX_CP))
        self.c_pt = ti.field(ti.f32, (MAX_PAIRS, MAX_CP))

    # ─────────────────────────────────────────────────────
    #  辅助 ti.func
    # ─────────────────────────────────────────────────────

    @ti.func
    def _rot(self, ang: ti.f32):
        c = ti.cos(ang)
        s = ti.sin(ang)
        return ti.Matrix([[c, -s], [s, c]])

    @ti.func
    def _cross2(self, a: ti.template(), b: ti.template()) -> ti.f32:
        return a[0] * b[1] - a[1] * b[0]

    @ti.func
    def _clip_2pts(self, v0: ti.template(), v1: ti.template(),
                   normal: ti.template(), offset: ti.f32):
        """将 2 端点线段裁剪到半平面 n·x ≤ offset。返回 (count, p0, p1)"""
        d0 = normal.dot(v0) - offset
        d1 = normal.dot(v1) - offset
        n  = 0
        p0 = ti.Vector([0.0, 0.0])
        p1 = ti.Vector([0.0, 0.0])
        if d0 <= 0.0:
            if n == 0:
                p0 = v0
            else:
                p1 = v0
            n += 1
        if d1 <= 0.0:
            if n == 0:
                p0 = v1
            else:
                p1 = v1
            n += 1
        if d0 * d1 < 0.0:
            t  = d0 / (d0 - d1)
            pt = v0 + t * (v1 - v0)
            if n == 0:
                p0 = pt
            else:
                p1 = pt
            n += 1
        return n, p0, p1

    @ti.func
    def _incident_edge(self, pos: ti.template(), rot: ti.template(),
                       normal: ti.template()):
        """返回入射边的两个世界坐标端点。"""
        hw = ti.cast(self.hw, ti.f32)
        hh = ti.cast(self.hh, ti.f32)
        n_local = -(rot.transpose() @ normal)
        na = ti.abs(n_local)
        v0 = ti.Vector([0.0, 0.0])
        v1 = ti.Vector([0.0, 0.0])
        if na[0] > na[1]:
            if n_local[0] > 0.0:
                v0 = pos + rot @ ti.Vector([ hw, -hh])
                v1 = pos + rot @ ti.Vector([ hw,  hh])
            else:
                v0 = pos + rot @ ti.Vector([-hw,  hh])
                v1 = pos + rot @ ti.Vector([-hw, -hh])
        else:
            if n_local[1] > 0.0:
                v0 = pos + rot @ ti.Vector([ hw,  hh])
                v1 = pos + rot @ ti.Vector([-hw,  hh])
            else:
                v0 = pos + rot @ ti.Vector([-hw, -hh])
                v1 = pos + rot @ ti.Vector([ hw, -hh])
        return v0, v1

    @ti.func
    def _collide_obb(self, pos_a: ti.template(), ang_a: ti.f32,
                     pos_b: ti.template(), ang_b: ti.f32):
        """
        OBB SAT 碰撞检测 (两个相同尺寸的盒子)。
        返回 (n_contacts, normal, c0, c1, s0, s1)
        """
        hw = ti.cast(self.hw, ti.f32)
        hh = ti.cast(self.hh, ti.f32)
        RA = self._rot(ang_a)
        RB = self._rot(ang_b)
        dP = pos_b - pos_a
        dA = RA.transpose() @ dP
        dB = RB.transpose() @ dP
        C  = RA.transpose() @ RB
        aC = ti.abs(C)

        # SAT 轴测试 (负值 = 穿透)
        fAx = ti.abs(dA[0]) - hw - (aC[0, 0] * hw + aC[0, 1] * hh)
        fAy = ti.abs(dA[1]) - hh - (aC[1, 0] * hw + aC[1, 1] * hh)
        fBx = ti.abs(dB[0]) - (aC[0, 0] * hw + aC[1, 0] * hh) - hw
        fBy = ti.abs(dB[1]) - (aC[0, 1] * hw + aC[1, 1] * hh) - hh

        nc     = 0
        normal = ti.Vector([0.0, 0.0])
        c0     = ti.Vector([0.0, 0.0])
        c1     = ti.Vector([0.0, 0.0])
        s0     = 0.0
        s1     = 0.0

        if fAx <= 0.0 and fAy <= 0.0 and fBx <= 0.0 and fBy <= 0.0:
            # 提取各旋转矩阵列向量
            col0_A = ti.Vector([RA[0, 0], RA[1, 0]])
            col1_A = ti.Vector([RA[0, 1], RA[1, 1]])
            col0_B = ti.Vector([RB[0, 0], RB[1, 0]])
            col1_B = ti.Vector([RB[0, 1], RB[1, 1]])

            # 选最小穿透轴 (最大分离量)
            rel_tol = 0.95
            axis = 0
            sep  = fAx
            normal = col0_A if dA[0] > 0.0 else -col0_A

            if fAy > rel_tol * sep + 0.01 * hh:
                sep = fAy; axis = 1
                normal = col1_A if dA[1] > 0.0 else -col1_A

            if fBx > rel_tol * sep + 0.01 * hw:
                sep = fBx; axis = 2
                normal = col0_B if dB[0] > 0.0 else -col0_B

            if fBy > rel_tol * sep + 0.01 * hh:
                axis = 3
                normal = col1_B if dB[1] > 0.0 else -col1_B

            # 构造参考面 / 入射边
            fn     = ti.Vector([0.0, 0.0])
            front  = 0.0
            side_n = ti.Vector([0.0, 0.0])
            neg_s  = 0.0
            pos_s  = 0.0
            ie_v0  = ti.Vector([0.0, 0.0])
            ie_v1  = ti.Vector([0.0, 0.0])

            if axis == 0:
                fn     = normal
                front  = pos_a.dot(fn) + hw
                side_n = col1_A
                side_d = pos_a.dot(side_n)
                neg_s  = -side_d + hh
                pos_s  =  side_d + hh
                ie_v0, ie_v1 = self._incident_edge(pos_b, RB, fn)
            elif axis == 1:
                fn     = normal
                front  = pos_a.dot(fn) + hh
                side_n = col0_A
                side_d = pos_a.dot(side_n)
                neg_s  = -side_d + hw
                pos_s  =  side_d + hw
                ie_v0, ie_v1 = self._incident_edge(pos_b, RB, fn)
            elif axis == 2:
                fn     = -normal
                front  = pos_b.dot(fn) + hw
                side_n = col1_B
                side_d = pos_b.dot(side_n)
                neg_s  = -side_d + hh
                pos_s  =  side_d + hh
                ie_v0, ie_v1 = self._incident_edge(pos_a, RA, fn)
            else:
                fn     = -normal
                front  = pos_b.dot(fn) + hh
                side_n = col0_B
                side_d = pos_b.dot(side_n)
                neg_s  = -side_d + hw
                pos_s  =  side_d + hw
                ie_v0, ie_v1 = self._incident_edge(pos_a, RA, fn)

            # Sutherland-Hodgman 裁剪
            n1, cp1_0, cp1_1 = self._clip_2pts(ie_v0, ie_v1, -side_n, neg_s)
            if n1 >= 2:
                n2, cp2_0, cp2_1 = self._clip_2pts(cp1_0, cp1_1, side_n, pos_s)
                if n2 >= 2:
                    sv0 = fn.dot(cp2_0) - front
                    if sv0 <= 0.0:
                        c0 = cp2_0 - sv0 * fn
                        s0 = sv0
                        nc = 1
                    sv1 = fn.dot(cp2_1) - front
                    if sv1 <= 0.0:
                        if nc == 0:
                            c0 = cp2_1 - sv1 * fn
                            s0 = sv1
                            nc = 1
                        else:
                            c1 = cp2_1 - sv1 * fn
                            s1 = sv1
                            nc = 2

        return nc, normal, c0, c1, s0, s1

    # ─────────────────────────────────────────────────────
    #  Kernel 1: 碰撞检测 (并行, 各对独立)
    # ─────────────────────────────────────────────────────

    @ti.kernel
    def _detect_all(self, nb: int):
        for p in range(MAX_PAIRS):
            bi = self.pair_i[p]
            bj = self.pair_j[p]
            if bi < nb and bj < nb:
                nc, nm, c0, c1, sv0, sv1 = self._collide_obb(
                    self.b_pos[bi], self.b_ang[bi],
                    self.b_pos[bj], self.b_ang[bj])
                self.n_contacts[p] = nc
                if nc >= 1:
                    self.c_normal[p, 0] = nm
                    self.c_pos[p, 0]    = c0
                    self.c_sep[p, 0]    = sv0
                if nc >= 2:
                    self.c_normal[p, 1] = nm
                    self.c_pos[p, 1]    = c1
                    self.c_sep[p, 1]    = sv1
                # 碰撞消失 → 清除累积冲量
                if nc == 0:
                    for k in ti.static(range(MAX_CP)):
                        self.c_pn[p, k] = 0.0
                        self.c_pt[p, k] = 0.0
            else:
                self.n_contacts[p] = 0
                for k in ti.static(range(MAX_CP)):
                    self.c_pn[p, k] = 0.0
                    self.c_pt[p, k] = 0.0

    # ─────────────────────────────────────────────────────
    #  Kernel 2: 有效质量 + warm-start (串行 Gauss-Seidel)
    # ─────────────────────────────────────────────────────

    @ti.kernel
    def _pre_step(self, nb: int, inv_dt: float):
        ti.loop_config(serialize=True)
        for p in range(MAX_PAIRS):
            nc = self.n_contacts[p]
            if nc <= 0:
                continue
            bi = self.pair_i[p]
            bj = self.pair_j[p]
            if bi >= nb or bj >= nb:
                continue
            for k in range(nc):
                nm   = self.c_normal[p, k]
                cpos = self.c_pos[p, k]
                r1   = cpos - self.b_pos[bi]
                r2   = cpos - self.b_pos[bj]
                self.c_r1[p, k] = r1
                self.c_r2[p, k] = r2

                # 法向有效质量
                rn1 = r1.dot(nm); rn2 = r2.dot(nm)
                kN  = (BODY_INV_MASS * 2.0
                       + BODY_INV_I * (r1.dot(r1) - rn1 * rn1)
                       + BODY_INV_I * (r2.dot(r2) - rn2 * rn2))
                self.c_mass_n[p, k] = 1.0 / kN

                # 切向有效质量
                tg  = ti.Vector([nm[1], -nm[0]])
                rt1 = r1.dot(tg); rt2 = r2.dot(tg)
                kT  = (BODY_INV_MASS * 2.0
                       + BODY_INV_I * (r1.dot(r1) - rt1 * rt1)
                       + BODY_INV_I * (r2.dot(r2) - rt2 * rt2))
                self.c_mass_t[p, k] = 1.0 / kT

                # Baumgarte 位置修正 + 恢复系数 bias
                sep      = self.c_sep[p, k]
                pos_bias = -0.2 * inv_dt * ti.min(0.0, sep + 0.01)
                omg1 = self.b_omg[bi]; omg2 = self.b_omg[bj]
                v1   = self.b_vel[bi] + ti.Vector([-omg1 * r1[1], omg1 * r1[0]])
                v2   = self.b_vel[bj] + ti.Vector([-omg2 * r2[1], omg2 * r2[0]])
                vn   = (v2 - v1).dot(nm)
                rest_bias = -RESTITUTION * vn if vn < -1.0 else 0.0
                self.c_bias[p, k] = pos_bias + rest_bias

                # warm-start: 施加上帧累积冲量
                P = self.c_pn[p, k] * nm + self.c_pt[p, k] * tg
                self.b_vel[bi] -= BODY_INV_MASS * P
                self.b_omg[bi] -= BODY_INV_I * self._cross2(r1, P)
                self.b_vel[bj] += BODY_INV_MASS * P
                self.b_omg[bj] += BODY_INV_I * self._cross2(r2, P)

    # ─────────────────────────────────────────────────────
    #  Kernel 3: 单次 SI 迭代 (串行 Gauss-Seidel)
    # ─────────────────────────────────────────────────────

    @ti.kernel
    def _si_iter(self, nb: int):
        ti.loop_config(serialize=True)
        for p in range(MAX_PAIRS):
            nc = self.n_contacts[p]
            if nc <= 0:
                continue
            bi = self.pair_i[p]
            bj = self.pair_j[p]
            if bi >= nb or bj >= nb:
                continue
            friction = ti.cast(FRICTION, ti.f32)
            for k in range(nc):
                nm  = self.c_normal[p, k]
                tg  = ti.Vector([nm[1], -nm[0]])
                r1  = self.c_r1[p, k]
                r2  = self.c_r2[p, k]

                # ── 法向冲量 ────────────────────────────
                omg1 = self.b_omg[bi]; omg2 = self.b_omg[bj]
                v1   = self.b_vel[bi] + ti.Vector([-omg1 * r1[1], omg1 * r1[0]])
                v2   = self.b_vel[bj] + ti.Vector([-omg2 * r2[1], omg2 * r2[0]])
                vn   = (v2 - v1).dot(nm)
                dPn  = self.c_mass_n[p, k] * (-vn + self.c_bias[p, k])
                Pn0  = self.c_pn[p, k]
                Pn1  = ti.max(Pn0 + dPn, 0.0)
                dPn  = Pn1 - Pn0
                self.c_pn[p, k] = Pn1
                Pn   = dPn * nm
                self.b_vel[bi] -= BODY_INV_MASS * Pn
                self.b_omg[bi] -= BODY_INV_I * self._cross2(r1, Pn)
                self.b_vel[bj] += BODY_INV_MASS * Pn
                self.b_omg[bj] += BODY_INV_I * self._cross2(r2, Pn)

                # ── 切向冲量 ────────────────────────────
                omg1 = self.b_omg[bi]; omg2 = self.b_omg[bj]
                v1   = self.b_vel[bi] + ti.Vector([-omg1 * r1[1], omg1 * r1[0]])
                v2   = self.b_vel[bj] + ti.Vector([-omg2 * r2[1], omg2 * r2[0]])
                vt   = (v2 - v1).dot(tg)
                dPt  = self.c_mass_t[p, k] * (-vt)
                Pt0  = self.c_pt[p, k]
                maxP = friction * self.c_pn[p, k]
                Pt1  = ti.min(ti.max(Pt0 + dPt, -maxP), maxP)
                dPt  = Pt1 - Pt0
                self.c_pt[p, k] = Pt1
                Pt   = dPt * tg
                self.b_vel[bi] -= BODY_INV_MASS * Pt
                self.b_omg[bi] -= BODY_INV_I * self._cross2(r1, Pt)
                self.b_vel[bj] += BODY_INV_MASS * Pt
                self.b_omg[bj] += BODY_INV_I * self._cross2(r2, Pt)

    # ─────────────────────────────────────────────────────
    #  顶层接口
    # ─────────────────────────────────────────────────────

    def solve(self, nb_field, dt_field):
        """
        Parameters
        ----------
        nb_field : ti.field(int, shape=())   n_bodies
        dt_field : ti.field(float, shape=()) dt
        """
        nb = int(nb_field[None])
        if nb < 2:
            return
        dt     = float(dt_field[None])
        inv_dt = 1.0 / dt if dt > 1e-10 else 0.0

        self._detect_all(nb)
        self._pre_step(nb, inv_dt)
        for _ in range(SI_ITERS):
            self._si_iter(nb)
