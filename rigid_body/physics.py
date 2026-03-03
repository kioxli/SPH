"""
physics.py — 纯 NumPy 刚体物理引擎
═══════════════════════════════════════════════════════════════

基于 box2d-lite 的 OBB SAT + Sequential Impulse，扩展支持:
    · 每个刚体独立尺寸 (hw, hh)
    · 静态刚体 (inv_m = inv_I = 0，用于地面 / 墙壁)
    · World 类统一管理重力积分 + 碰撞求解

参考: Erin Catto, "Iterative Dynamics with Temporal Coherence", GDC 2005
"""

import math
import copy
import numpy as np

from config import GRAVITY, FRICTION, RESTITUTION, SI_ITERS


# ═══════════════════════════════════════════════════════════
#  刚体
# ═══════════════════════════════════════════════════════════

class Body:
    """动态矩形刚体。"""

    def __init__(self, x, y, hw, hh, density=1.0, angle=0.0):
        self.pos   = np.array([x, y], dtype=np.float64)
        self.vel   = np.zeros(2,      dtype=np.float64)
        self.ang   = float(angle)
        self.omg   = 0.0
        self.hw    = float(hw)
        self.hh    = float(hh)
        mass       = density * 4.0 * hw * hh
        I          = mass * (4.0 * hw**2 + 4.0 * hh**2) / 12.0
        self.inv_m = 1.0 / mass
        self.inv_I = 1.0 / I

    def get_corners(self):
        """返回 4 个角点的世界坐标 (逆时针)。"""
        c, s = math.cos(self.ang), math.sin(self.ang)
        offs = [(-self.hw, -self.hh), (self.hw, -self.hh),
                (self.hw,  self.hh), (-self.hw,  self.hh)]
        return [(self.pos[0] + c*lx - s*ly,
                 self.pos[1] + s*lx + c*ly) for lx, ly in offs]


class StaticBody(Body):
    """不可移动刚体 (地面 / 墙壁)。inv_m = inv_I = 0。"""

    def __init__(self, x, y, hw, hh, angle=0.0):
        super().__init__(x, y, hw, hh, density=1.0, angle=angle)
        self.inv_m = 0.0
        self.inv_I = 0.0


# ═══════════════════════════════════════════════════════════
#  接触点
# ═══════════════════════════════════════════════════════════

class Contact:
    __slots__ = ('position', 'normal', 'separation',
                 'Pn', 'Pt',
                 'mass_n', 'mass_t', 'bias',
                 'r1', 'r2', 'feature')

    def __init__(self):
        self.position   = np.zeros(2, dtype=np.float64)
        self.normal     = np.zeros(2, dtype=np.float64)
        self.separation = 0.0
        self.Pn = 0.0
        self.Pt = 0.0
        self.mass_n = 0.0
        self.mass_t = 0.0
        self.bias   = 0.0
        self.r1     = np.zeros(2, dtype=np.float64)
        self.r2     = np.zeros(2, dtype=np.float64)
        self.feature = (0, 0, 0, 0)


# ═══════════════════════════════════════════════════════════
#  SAT 碰撞检测 (OBB vs OBB)
# ═══════════════════════════════════════════════════════════

_FACE_AX, _FACE_AY, _FACE_BX, _FACE_BY = 0, 1, 2, 3


def _rot2d(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _cross2(a, b):
    return a[0]*b[1] - a[1]*b[0]


def _clip_segment(v_in, normal, offset, clip_edge):
    out = []
    d0  = np.dot(normal, v_in[0][0]) - offset
    d1  = np.dot(normal, v_in[1][0]) - offset
    if d0 <= 0.0: out.append(v_in[0])
    if d1 <= 0.0: out.append(v_in[1])
    if d0 * d1 < 0.0:
        t  = d0 / (d0 - d1)
        pt = v_in[0][0] + t * (v_in[1][0] - v_in[0][0])
        fp = ((clip_edge, v_in[0][1][1], 0, v_in[0][1][3])
              if d0 > 0.0
              else (v_in[1][1][0], clip_edge, v_in[1][1][2], 0))
        out.append((pt, fp))
    return out


def _incident_edge(hw, hh, pos, rot, normal):
    n_local = -(rot.T @ normal)
    na      = np.abs(n_local)
    if na[0] > na[1]:
        if n_local[0] > 0.0:
            v0, fp0 = np.array([ hw, -hh]), (0,0,3,4)
            v1, fp1 = np.array([ hw,  hh]), (0,0,4,1)
        else:
            v0, fp0 = np.array([-hw,  hh]), (0,0,1,2)
            v1, fp1 = np.array([-hw, -hh]), (0,0,2,3)
    else:
        if n_local[1] > 0.0:
            v0, fp0 = np.array([ hw,  hh]), (0,0,4,1)
            v1, fp1 = np.array([-hw,  hh]), (0,0,1,2)
        else:
            v0, fp0 = np.array([-hw, -hh]), (0,0,2,3)
            v1, fp1 = np.array([ hw, -hh]), (0,0,3,4)
    return [(pos + rot @ v0, fp0), (pos + rot @ v1, fp1)]


def collide_obb(pos_a, ang_a, hw_a, hh_a,
                pos_b, ang_b, hw_b, hh_b):
    """
    OBB SAT 碰撞检测。每个 OBB 可以有不同尺寸。

    Returns
    -------
    contacts : list[Contact]   (0, 1 或 2 个接触点)
    """
    RA  = _rot2d(ang_a)
    RB  = _rot2d(ang_b)
    dP  = pos_b - pos_a
    dA  = RA.T @ dP
    dB  = RB.T @ dP
    C   = RA.T @ RB
    aC  = np.abs(C)
    hA  = np.array([hw_a, hh_a])
    hB  = np.array([hw_b, hh_b])

    faceA = np.abs(dA) - hA - aC @ hB
    faceB = np.abs(dB) - aC.T @ hA - hB

    if faceA[0] > 0 or faceA[1] > 0: return []
    if faceB[0] > 0 or faceB[1] > 0: return []

    axis   = _FACE_AX
    sep    = faceA[0]
    normal = RA[:, 0] if dA[0] > 0 else -RA[:, 0]

    rel, abs_ = 0.95, 0.01
    if faceA[1] > rel*sep + abs_*hA[1]:
        sep = faceA[1]; axis = _FACE_AY
        normal = RA[:, 1] if dA[1] > 0 else -RA[:, 1]
    if faceB[0] > rel*sep + abs_*hB[0]:
        sep = faceB[0]; axis = _FACE_BX
        normal = RB[:, 0] if dB[0] > 0 else -RB[:, 0]
    if faceB[1] > rel*sep + abs_*hB[1]:
        axis = _FACE_BY
        normal = RB[:, 1] if dB[1] > 0 else -RB[:, 1]

    if axis == _FACE_AX:
        fn = normal; front = np.dot(pos_a, fn) + hw_a
        side_n = RA[:, 1]; side_d = np.dot(pos_a, side_n)
        neg_s, pos_s = -side_d + hh_a, side_d + hh_a
        inc = _incident_edge(hw_b, hh_b, pos_b, RB, fn)
    elif axis == _FACE_AY:
        fn = normal; front = np.dot(pos_a, fn) + hh_a
        side_n = RA[:, 0]; side_d = np.dot(pos_a, side_n)
        neg_s, pos_s = -side_d + hw_a, side_d + hw_a
        inc = _incident_edge(hw_b, hh_b, pos_b, RB, fn)
    elif axis == _FACE_BX:
        fn = -normal; front = np.dot(pos_b, fn) + hw_b
        side_n = RB[:, 1]; side_d = np.dot(pos_b, side_n)
        neg_s, pos_s = -side_d + hh_b, side_d + hh_b
        inc = _incident_edge(hw_a, hh_a, pos_a, RA, fn)
    else:
        fn = -normal; front = np.dot(pos_b, fn) + hh_b
        side_n = RB[:, 0]; side_d = np.dot(pos_b, side_n)
        neg_s, pos_s = -side_d + hw_b, side_d + hw_b
        inc = _incident_edge(hw_a, hh_a, pos_a, RA, fn)

    cp1 = _clip_segment(inc, -side_n, neg_s, 3 if axis in (_FACE_AX,_FACE_BX) else 2)
    if len(cp1) < 2: return []
    cp2 = _clip_segment(cp1,  side_n, pos_s, 1 if axis in (_FACE_AX,_FACE_BX) else 4)
    if len(cp2) < 2: return []

    contacts = []
    for pt, fp in cp2:
        s = np.dot(fn, pt) - front
        if s <= 0.0:
            c             = Contact()
            c.separation  = s
            c.normal      = normal.copy()
            c.position    = pt - s * fn
            c.feature     = (fp[2],fp[3],fp[0],fp[1]) if axis in (_FACE_BX,_FACE_BY) else fp
            contacts.append(c)
    return contacts


# ═══════════════════════════════════════════════════════════
#  Arbiter — 一对刚体的接触管理器 (含 warm-start)
# ═══════════════════════════════════════════════════════════

class Arbiter:

    def __init__(self, contacts):
        self.contacts = contacts
        self.num      = len(contacts)

    def update(self, new_contacts):
        merged = []
        for cn in new_contacts:
            old = next((co for co in self.contacts if cn.feature == co.feature), None)
            c   = copy.copy(cn)
            if old is not None:
                c.Pn = old.Pn
                c.Pt = old.Pt
            merged.append(c)
        self.contacts = merged
        self.num      = len(merged)

    def pre_step(self, bodies, i, j, inv_dt):
        b1, b2 = bodies[i], bodies[j]
        for c in self.contacts:
            c.r1 = c.position - b1['pos']
            c.r2 = c.position - b2['pos']
            rn1  = np.dot(c.r1, c.normal)
            rn2  = np.dot(c.r2, c.normal)

            kN = (b1['inv_m'] + b2['inv_m']
                  + b1['inv_I'] * (np.dot(c.r1,c.r1) - rn1*rn1)
                  + b2['inv_I'] * (np.dot(c.r2,c.r2) - rn2*rn2))
            c.mass_n = 1.0 / kN

            tg  = np.array([c.normal[1], -c.normal[0]])
            rt1 = np.dot(c.r1, tg)
            rt2 = np.dot(c.r2, tg)
            kT  = (b1['inv_m'] + b2['inv_m']
                   + b1['inv_I'] * (np.dot(c.r1,c.r1) - rt1*rt1)
                   + b2['inv_I'] * (np.dot(c.r2,c.r2) - rt2*rt2))
            c.mass_t = 1.0 / kT

            c.bias = -0.2 * inv_dt * min(0.0, c.separation + 0.01)

            def _cw(w, r): return np.array([-w*r[1], w*r[0]])
            v1  = b1['vel'] + _cw(b1['omg'], c.r1)
            v2  = b2['vel'] + _cw(b2['omg'], c.r2)
            vn  = np.dot(v2 - v1, c.normal)
            c.bias += -RESTITUTION * vn if vn < -1.0 else 0.0

            # warm-start
            P = c.Pn * c.normal + c.Pt * tg
            b1['vel'] -= b1['inv_m'] * P
            b1['omg'] -= b1['inv_I'] * _cross2(c.r1, P)
            b2['vel'] += b2['inv_m'] * P
            b2['omg'] += b2['inv_I'] * _cross2(c.r2, P)

    def apply_impulse(self, bodies, i, j):
        b1, b2   = bodies[i], bodies[j]
        friction = FRICTION
        for c in self.contacts:
            c.r1 = c.position - b1['pos']
            c.r2 = c.position - b2['pos']

            def _cw(w, r): return np.array([-w*r[1], w*r[0]])

            # 法向冲量
            dv  = b2['vel'] + _cw(b2['omg'],c.r2) - b1['vel'] - _cw(b1['omg'],c.r1)
            vn  = np.dot(dv, c.normal)
            dPn = c.mass_n * (-vn + c.bias)
            Pn0 = c.Pn
            c.Pn = max(Pn0 + dPn, 0.0)
            dPn  = c.Pn - Pn0
            Pn   = dPn * c.normal
            b1['vel'] -= b1['inv_m'] * Pn
            b1['omg'] -= b1['inv_I'] * _cross2(c.r1, Pn)
            b2['vel'] += b2['inv_m'] * Pn
            b2['omg'] += b2['inv_I'] * _cross2(c.r2, Pn)

            # 切向冲量
            dv  = b2['vel'] + _cw(b2['omg'],c.r2) - b1['vel'] - _cw(b1['omg'],c.r1)
            tg  = np.array([c.normal[1], -c.normal[0]])
            vt  = np.dot(dv, tg)
            dPt = c.mass_t * (-vt)
            Pt0 = c.Pt
            c.Pt = np.clip(Pt0 + dPt, -friction*c.Pn, friction*c.Pn)
            dPt  = c.Pt - Pt0
            Pt   = dPt * tg
            b1['vel'] -= b1['inv_m'] * Pt
            b1['omg'] -= b1['inv_I'] * _cross2(c.r1, Pt)
            b2['vel'] += b2['inv_m'] * Pt
            b2['omg'] += b2['inv_I'] * _cross2(c.r2, Pt)


# ═══════════════════════════════════════════════════════════
#  CollisionSolver — 管理所有刚体对
# ═══════════════════════════════════════════════════════════

class CollisionSolver:

    def __init__(self):
        self.arbiters: dict[tuple[int,int], Arbiter] = {}

    def solve(self, bodies: list[dict], dt: float):
        n      = len(bodies)
        inv_dt = 1.0 / dt if dt > 1e-10 else 0.0

        # 宽相 + 窄相 (每个 body dict 中含 hw, hh 独立尺寸)
        active = set()
        for i in range(n):
            for j in range(i+1, n):
                cs = collide_obb(
                    bodies[i]['pos'], bodies[i]['ang'],
                    bodies[i]['hw'],  bodies[i]['hh'],
                    bodies[j]['pos'], bodies[j]['ang'],
                    bodies[j]['hw'],  bodies[j]['hh'])
                key = (i, j)
                if cs:
                    active.add(key)
                    if key in self.arbiters:
                        self.arbiters[key].update(cs)
                    else:
                        self.arbiters[key] = Arbiter(cs)
                else:
                    self.arbiters.pop(key, None)

        for k in list(self.arbiters):
            if k not in active:
                del self.arbiters[k]

        for (i,j), arb in self.arbiters.items():
            arb.pre_step(bodies, i, j, inv_dt)

        for _ in range(SI_ITERS):
            for (i,j), arb in self.arbiters.items():
                arb.apply_impulse(bodies, i, j)


# ═══════════════════════════════════════════════════════════
#  World — 场景管理 (重力 + 积分 + 碰撞)
# ═══════════════════════════════════════════════════════════

class World:
    """管理所有刚体、重力积分与碰撞求解。"""

    def __init__(self):
        self.bodies: list[Body] = []
        self.solver = CollisionSolver()

    def add(self, body: Body) -> Body:
        self.bodies.append(body)
        return body

    def step(self, dt: float):
        # 1. 重力 → 动态刚体速度
        for b in self.bodies:
            if b.inv_m > 0:
                b.vel[1] += GRAVITY * dt

        # 2. 打包为 solver 所需的 dict 列表
        bd = [{'pos': b.pos.copy(), 'vel': b.vel.copy(),
               'ang': b.ang,        'omg': b.omg,
               'inv_m': b.inv_m,    'inv_I': b.inv_I,
               'hw': b.hw,          'hh': b.hh}
              for b in self.bodies]

        # 3. 碰撞检测 + Sequential Impulse
        self.solver.solve(bd, dt)

        # 4. 写回速度
        for i, b in enumerate(self.bodies):
            b.vel[:] = bd[i]['vel']
            b.omg    = bd[i]['omg']

        # 5. 位置积分 (仅动态刚体)
        for b in self.bodies:
            if b.inv_m > 0:
                b.pos += dt * b.vel
                b.ang += dt * b.omg
