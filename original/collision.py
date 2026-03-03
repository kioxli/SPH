"""
collision.py — 刚体间碰撞检测与约束求解
═══════════════════════════════════════════════════════════════

SAT (Separating Axis Theorem) 窄相检测 + Sequential Impulse 求解器,
移植自 box2d-lite 并简化为纯 NumPy 实现。

流程:
    CollisionSolver.solve(bodies, dt)
        1. O(n^2) 遍历所有刚体对
        2. SAT 窄相 → 接触点 + 法线 + 穿透深度
        3. Pre-step: 计算有效质量、bias
        4. N 次 SI 迭代: 法向 + 切向冲量钳位
        5. 写回修正后的速度 / 角速度

参考: Erin Catto, "Iterative Dynamics with Temporal Coherence", GDC 2005
"""

import math
import copy
import numpy as np

from config import FRICTION, RESTITUTION, SI_ITERS


# ═══════════════════════════════════════════════════════════
#  辅助
# ═══════════════════════════════════════════════════════════

def _rot2d(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


# ═══════════════════════════════════════════════════════════
#  接触点
# ═══════════════════════════════════════════════════════════

class Contact:
    __slots__ = ('position', 'normal', 'separation',
                 'Pn', 'Pt', 'Pnb',
                 'mass_n', 'mass_t', 'bias',
                 'r1', 'r2', 'feature')

    def __init__(self):
        self.position = np.zeros(2, dtype=np.float64)
        self.normal   = np.zeros(2, dtype=np.float64)
        self.separation = 0.0
        self.Pn  = 0.0
        self.Pt  = 0.0
        self.Pnb = 0.0
        self.mass_n = 0.0
        self.mass_t = 0.0
        self.bias = 0.0
        self.r1 = np.zeros(2, dtype=np.float64)
        self.r2 = np.zeros(2, dtype=np.float64)
        self.feature = (0, 0, 0, 0)


# ═══════════════════════════════════════════════════════════
#  SAT 窄相 (OBB vs OBB)
# ═══════════════════════════════════════════════════════════

_FACE_AX, _FACE_AY, _FACE_BX, _FACE_BY = 0, 1, 2, 3
_EDGE = {0: (3, 1), 1: (2, 4), 2: (3, 1), 3: (2, 4)}


def _clip_segment(v_in, normal, offset, clip_edge):
    """Sutherland-Hodgman 单边裁剪。"""
    out = []
    d0 = np.dot(normal, v_in[0][0]) - offset
    d1 = np.dot(normal, v_in[1][0]) - offset
    if d0 <= 0.0:
        out.append(v_in[0])
    if d1 <= 0.0:
        out.append(v_in[1])
    if d0 * d1 < 0.0:
        t = d0 / (d0 - d1)
        pt = v_in[0][0] + t * (v_in[1][0] - v_in[0][0])
        if d0 > 0.0:
            fp = (clip_edge, v_in[0][1][1], 0, v_in[0][1][3])
        else:
            fp = (v_in[1][1][0], clip_edge, v_in[1][1][2], 0)
        out.append((pt, fp))
    return out


def _incident_edge(hw, hh, pos, rot, normal):
    """计算入射边的两个世界坐标端点 + feature 标记。"""
    n_local = -(rot.T @ normal)
    na = np.abs(n_local)
    if na[0] > na[1]:
        if n_local[0] > 0.0:
            v0 = np.array([ hw, -hh]);  fp0 = (0, 0, 3, 4)
            v1 = np.array([ hw,  hh]);  fp1 = (0, 0, 4, 1)
        else:
            v0 = np.array([-hw,  hh]);  fp0 = (0, 0, 1, 2)
            v1 = np.array([-hw, -hh]);  fp1 = (0, 0, 2, 3)
    else:
        if n_local[1] > 0.0:
            v0 = np.array([ hw,  hh]);  fp0 = (0, 0, 4, 1)
            v1 = np.array([-hw,  hh]);  fp1 = (0, 0, 1, 2)
        else:
            v0 = np.array([-hw, -hh]);  fp0 = (0, 0, 2, 3)
            v1 = np.array([ hw, -hh]);  fp1 = (0, 0, 3, 4)
    return [(pos + rot @ v0, fp0), (pos + rot @ v1, fp1)]


def collide_obb(pos_a, ang_a, hw_a, hh_a,
                pos_b, ang_b, hw_b, hh_b):
    """
    SAT 碰撞检测 (两个 OBB)。

    Returns
    -------
    contacts : list[Contact]    (0, 1, 或 2 个接触点)
    """
    RA = _rot2d(ang_a)
    RB = _rot2d(ang_b)
    dP = pos_b - pos_a
    dA = RA.T @ dP
    dB = RB.T @ dP
    C  = RA.T @ RB
    aC = np.abs(C)
    hA = np.array([hw_a, hh_a])
    hB = np.array([hw_b, hh_b])

    faceA = np.abs(dA) - hA - aC @ hB
    faceB = np.abs(dB) - aC.T @ hA - hB

    if faceA[0] > 0.0 or faceA[1] > 0.0:
        return []
    if faceB[0] > 0.0 or faceB[1] > 0.0:
        return []

    # 选择最佳分离轴
    axis = _FACE_AX
    sep  = faceA[0]
    normal = RA[:, 0] if dA[0] > 0.0 else -RA[:, 0]

    rel_tol = 0.95
    abs_tol = 0.01

    if faceA[1] > rel_tol * sep + abs_tol * hA[1]:
        sep = faceA[1]; axis = _FACE_AY
        normal = RA[:, 1] if dA[1] > 0.0 else -RA[:, 1]

    if faceB[0] > rel_tol * sep + abs_tol * hB[0]:
        sep = faceB[0]; axis = _FACE_BX
        normal = RB[:, 0] if dB[0] > 0.0 else -RB[:, 0]

    if faceB[1] > rel_tol * sep + abs_tol * hB[1]:
        sep = faceB[1]; axis = _FACE_BY
        normal = RB[:, 1] if dB[1] > 0.0 else -RB[:, 1]

    # 构造参考面 / 入射边
    if axis == _FACE_AX:
        fn = normal
        front = np.dot(pos_a, fn) + hw_a
        side_n = RA[:, 1]; side_d = np.dot(pos_a, side_n)
        neg_s, pos_s = -side_d + hh_a, side_d + hh_a
        neg_e, pos_e = 3, 1
        inc = _incident_edge(hw_b, hh_b, pos_b, RB, fn)
    elif axis == _FACE_AY:
        fn = normal
        front = np.dot(pos_a, fn) + hh_a
        side_n = RA[:, 0]; side_d = np.dot(pos_a, side_n)
        neg_s, pos_s = -side_d + hw_a, side_d + hw_a
        neg_e, pos_e = 2, 4
        inc = _incident_edge(hw_b, hh_b, pos_b, RB, fn)
    elif axis == _FACE_BX:
        fn = -normal
        front = np.dot(pos_b, fn) + hw_b
        side_n = RB[:, 1]; side_d = np.dot(pos_b, side_n)
        neg_s, pos_s = -side_d + hh_b, side_d + hh_b
        neg_e, pos_e = 3, 1
        inc = _incident_edge(hw_a, hh_a, pos_a, RA, fn)
    else:
        fn = -normal
        front = np.dot(pos_b, fn) + hh_b
        side_n = RB[:, 0]; side_d = np.dot(pos_b, side_n)
        neg_s, pos_s = -side_d + hw_b, side_d + hw_b
        neg_e, pos_e = 2, 4
        inc = _incident_edge(hw_a, hh_a, pos_a, RA, fn)

    cp1 = _clip_segment(inc, -side_n, neg_s, neg_e)
    if len(cp1) < 2:
        return []
    cp2 = _clip_segment(cp1, side_n, pos_s, pos_e)
    if len(cp2) < 2:
        return []

    contacts = []
    for pt, fp in cp2:
        s = np.dot(fn, pt) - front
        if s <= 0.0:
            c = Contact()
            c.separation = s
            c.normal = normal.copy()
            c.position = pt - s * fn
            if axis in (_FACE_BX, _FACE_BY):
                c.feature = (fp[2], fp[3], fp[0], fp[1])
            else:
                c.feature = fp
            contacts.append(c)
    return contacts


# ═══════════════════════════════════════════════════════════
#  Arbiter (一对刚体的接触管理器)
# ═══════════════════════════════════════════════════════════

class Arbiter:
    """管理一对刚体之间的接触点, 支持 warm-starting。"""

    def __init__(self, contacts):
        self.contacts = contacts
        self.num = len(contacts)

    def update(self, new_contacts):
        """合并新接触, 继承旧接触的累积冲量 (warm-start)。"""
        merged = []
        for cn in new_contacts:
            old_match = None
            for co in self.contacts:
                if cn.feature == co.feature:
                    old_match = co
                    break
            c = copy.copy(cn)
            if old_match is not None:
                c.Pn  = old_match.Pn
                c.Pt  = old_match.Pt
                c.Pnb = old_match.Pnb
            merged.append(c)
        self.contacts = merged
        self.num = len(merged)

    def pre_step(self, bodies, i, j, inv_dt):
        b1, b2 = bodies[i], bodies[j]
        k_allowed = 0.01
        k_bias = 0.2
        rest_thresh = 1.0
        for c in self.contacts:
            c.r1 = c.position - b1['pos']
            c.r2 = c.position - b2['pos']
            rn1 = np.dot(c.r1, c.normal)
            rn2 = np.dot(c.r2, c.normal)

            kN = (b1['inv_m'] + b2['inv_m']
                  + b1['inv_I'] * (np.dot(c.r1, c.r1) - rn1 * rn1)
                  + b2['inv_I'] * (np.dot(c.r2, c.r2) - rn2 * rn2))
            c.mass_n = 1.0 / kN

            tangent = np.array([c.normal[1], -c.normal[0]])
            rt1 = np.dot(c.r1, tangent)
            rt2 = np.dot(c.r2, tangent)
            kT = (b1['inv_m'] + b2['inv_m']
                  + b1['inv_I'] * (np.dot(c.r1, c.r1) - rt1 * rt1)
                  + b2['inv_I'] * (np.dot(c.r2, c.r2) - rt2 * rt2))
            c.mass_t = 1.0 / kT

            pos_bias = -k_bias * inv_dt * min(0.0, c.separation + k_allowed)

            # 恢复速度 bias
            def _cross_w(w, r):
                return np.array([-w * r[1], w * r[0]])
            v1 = b1['vel'] + _cross_w(b1['omg'], c.r1)
            v2 = b2['vel'] + _cross_w(b2['omg'], c.r2)
            vn = np.dot(v2 - v1, c.normal)
            rest_bias = -RESTITUTION * vn if vn < -rest_thresh else 0.0
            c.bias = pos_bias + rest_bias

            # warm-start: 施加上一帧累积冲量
            P = c.Pn * c.normal + c.Pt * tangent
            b1['vel'] -= b1['inv_m'] * P
            b1['omg'] -= b1['inv_I'] * _cross2(c.r1, P)
            b2['vel'] += b2['inv_m'] * P
            b2['omg'] += b2['inv_I'] * _cross2(c.r2, P)

    def apply_impulse(self, bodies, i, j):
        b1, b2 = bodies[i], bodies[j]
        friction = math.sqrt(FRICTION * FRICTION)
        for c in self.contacts:
            c.r1 = c.position - b1['pos']
            c.r2 = c.position - b2['pos']

            def _cross_w(w, r):
                return np.array([-w * r[1], w * r[0]])
            dv = (b2['vel'] + _cross_w(b2['omg'], c.r2)
                  - b1['vel'] - _cross_w(b1['omg'], c.r1))

            # 法向冲量
            vn = np.dot(dv, c.normal)
            dPn = c.mass_n * (-vn + c.bias)
            Pn0 = c.Pn
            c.Pn = max(Pn0 + dPn, 0.0)
            dPn = c.Pn - Pn0
            Pn = dPn * c.normal
            b1['vel'] -= b1['inv_m'] * Pn
            b1['omg'] -= b1['inv_I'] * _cross2(c.r1, Pn)
            b2['vel'] += b2['inv_m'] * Pn
            b2['omg'] += b2['inv_I'] * _cross2(c.r2, Pn)

            # 切向冲量
            dv = (b2['vel'] + _cross_w(b2['omg'], c.r2)
                  - b1['vel'] - _cross_w(b1['omg'], c.r1))
            tangent = np.array([c.normal[1], -c.normal[0]])
            vt = np.dot(dv, tangent)
            dPt = c.mass_t * (-vt)
            maxPt = friction * c.Pn
            Pt0 = c.Pt
            c.Pt = np.clip(Pt0 + dPt, -maxPt, maxPt)
            dPt = c.Pt - Pt0
            Pt = dPt * tangent
            b1['vel'] -= b1['inv_m'] * Pt
            b1['omg'] -= b1['inv_I'] * _cross2(c.r1, Pt)
            b2['vel'] += b2['inv_m'] * Pt
            b2['omg'] += b2['inv_I'] * _cross2(c.r2, Pt)


# ═══════════════════════════════════════════════════════════
#  CollisionSolver (顶层接口)
# ═══════════════════════════════════════════════════════════

class CollisionSolver:
    """
    管理所有刚体对的碰撞检测与 Sequential Impulse 求解。

    用法:
        solver = CollisionSolver(hw, hh)
        solver.solve(bodies, dt)   # bodies 列表会被就地修改 vel / omg
    """

    def __init__(self, hw: float, hh: float):
        self.hw = hw
        self.hh = hh
        self.arbiters: dict[tuple[int, int], Arbiter] = {}

    def solve(self, bodies: list[dict], dt: float):
        """
        Parameters
        ----------
        bodies : list[dict]
            每个 dict 包含: pos(2,), vel(2,), ang, omg, inv_m, inv_I
        dt : float
        """
        n = len(bodies)
        if n < 2:
            return

        inv_dt = 1.0 / dt if dt > 1e-10 else 0.0

        # Broad + narrow phase
        new_keys = set()
        for i in range(n):
            for j in range(i + 1, n):
                cs = collide_obb(
                    bodies[i]['pos'], bodies[i]['ang'],
                    self.hw, self.hh,
                    bodies[j]['pos'], bodies[j]['ang'],
                    self.hw, self.hh)
                key = (i, j)
                if cs:
                    new_keys.add(key)
                    if key in self.arbiters:
                        self.arbiters[key].update(cs)
                    else:
                        self.arbiters[key] = Arbiter(cs)
                else:
                    self.arbiters.pop(key, None)

        # 清理失效 arbiter
        for k in list(self.arbiters.keys()):
            if k not in new_keys:
                del self.arbiters[k]

        # Pre-step
        for (i, j), arb in self.arbiters.items():
            arb.pre_step(bodies, i, j, inv_dt)

        # SI 迭代
        for _ in range(SI_ITERS):
            for (i, j), arb in self.arbiters.items():
                arb.apply_impulse(bodies, i, j)
