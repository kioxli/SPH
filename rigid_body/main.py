"""
main.py — 纯刚体模拟 Demo (box2d-lite 风格)
═══════════════════════════════════════════════════════════════

操作:
    · 左键点击  → 在鼠标位置投入新矩形刚体
    · 空格      → 暂停/继续
    · R         → 重置场景
    · ESC/关闭  → 退出

颜色:
    浅灰  = 静态边界 (地面/墙壁)
    彩色  = 动态刚体 (每个不同颜色)

运行:
    cd SPH/fsi/rigid_body
    python main.py
"""

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

from config import DOMAIN_W, DOMAIN_H, DT, SUBSTEPS, MAX_BODIES
from physics import Body, StaticBody, World

# ── 颜色调色板 (动态刚体循环使用) ──────────────────────────
_PALETTE = [
    (0.95, 0.35, 0.35),   # 红
    (0.35, 0.85, 0.45),   # 绿
    (0.40, 0.60, 0.95),   # 蓝
    (0.95, 0.80, 0.20),   # 黄
    (0.85, 0.45, 0.90),   # 紫
    (0.30, 0.90, 0.90),   # 青
    (0.95, 0.55, 0.20),   # 橙
    (0.60, 0.95, 0.40),   # 黄绿
]

# ── Taichi 渲染字段 ─────────────────────────────────────────
# 每个刚体: 4 条边 × 2 端点 = 8 个顶点
_N_STATIC     = 3                           # 地面 + 左墙 + 右墙
_MAX_VERTS    = (_N_STATIC + MAX_BODIES) * 8
seg_field     = ti.Vector.field(2, dtype=ti.f32, shape=_MAX_VERTS)
col_field     = ti.Vector.field(3, dtype=ti.f32, shape=_MAX_VERTS)


# ── 工具 ────────────────────────────────────────────────────
def _to_screen(x: float, y: float):
    """世界坐标 → 归一化屏幕坐标 [0,1]×[0,1]。"""
    return x / DOMAIN_W, y / DOMAIN_H


def _make_world():
    """构建初始场景, 返回 (world, n_static)。"""
    world = World()

    # ── 静态边界 ──────────────────────────────────────────
    # 地面: 顶面在 y=0, 中心在 y=-0.5
    world.add(StaticBody(DOMAIN_W / 2,       -0.5,
                         DOMAIN_W / 2 + 0.5, 0.5))
    # 左墙: 右面在 x=0, 中心在 x=-0.5
    world.add(StaticBody(-0.5,
                         DOMAIN_H / 2,
                         0.5,
                         DOMAIN_H / 2 + 0.5))
    # 右墙: 左面在 x=DOMAIN_W, 中心在 x=DOMAIN_W+0.5
    world.add(StaticBody(DOMAIN_W + 0.5,
                         DOMAIN_H / 2,
                         0.5,
                         DOMAIN_H / 2 + 0.5))
    n_static = 3

    # ── 金字塔 (5 行, 共 15 个动态刚体) ─────────────────
    bw      = 0.36          # 半宽
    bh      = 0.36          # 半高
    step    = bw * 2 + 0.04 # 列间距 (含微小间隙)

    for row in range(5):
        n_cols = 5 - row
        y_ctr  = bh + row * step + 0.02    # 底行刚好贴近地面
        for k in range(n_cols):
            x_ctr = DOMAIN_W / 2 + (k - (n_cols - 1) / 2) * step
            world.add(Body(x_ctr, y_ctr, bw, bh, density=1.0))

    return world, n_static


def _build_render_buffers(world, n_static):
    """
    将所有刚体的角点写入 Taichi 渲染字段。

    Returns
    -------
    n_verts : int — 有效顶点数 (每 2 个组成一条线段)
    """
    arr_s = np.zeros((_MAX_VERTS, 2), dtype=np.float32)
    arr_c = np.zeros((_MAX_VERTS, 3), dtype=np.float32)

    vi = 0
    for b_idx, body in enumerate(world.bodies):
        if vi + 8 > _MAX_VERTS:
            break
        corners = body.get_corners()   # 4 个世界坐标角点 (逆时针)
        if b_idx < n_static:
            col = (0.70, 0.72, 0.75)
        else:
            col = _PALETTE[(b_idx - n_static) % len(_PALETTE)]

        for k in range(4):
            x0, y0 = _to_screen(*corners[k])
            x1, y1 = _to_screen(*corners[(k + 1) % 4])
            arr_s[vi]     = [x0, y0]
            arr_s[vi + 1] = [x1, y1]
            arr_c[vi]     = col
            arr_c[vi + 1] = col
            vi += 2

    seg_field.from_numpy(arr_s)
    col_field.from_numpy(arr_c)
    return vi


# ── 主循环 ──────────────────────────────────────────────────
def main():
    world, n_static = _make_world()

    win_h  = int(900 * DOMAIN_H / DOMAIN_W)
    window = ti.ui.Window(
        "Rigid Body Demo  |  左键投入  空格暂停  R重置",
        res=(900, win_h),
        vsync=False,
    )
    canvas = window.get_canvas()
    dt_sub = DT / SUBSTEPS
    paused = False
    frame  = 0

    print("开始模拟 …")
    print("  左键 = 投入新刚体 | 空格 = 暂停/继续 | R = 重置\n")

    while window.running:

        # ── 输入处理 ────────────────────────────────────────
        while window.get_event(ti.ui.PRESS):
            key = window.event.key

            if key == ti.ui.LMB:
                cx, cy = window.get_cursor_pos()
                n_dyn  = len(world.bodies) - n_static
                if n_dyn < MAX_BODIES:
                    wx = cx * DOMAIN_W
                    wy = cy * DOMAIN_H
                    world.add(Body(wx, wy, hw=0.32, hh=0.32,
                                   density=1.0, angle=0.35))
                else:
                    print(f"  已达上限 ({MAX_BODIES} 个动态刚体)")

            elif key == ' ':
                paused = not paused
                print("  [暂停]" if paused else "  [继续]")

            elif key == 'r':
                world, n_static = _make_world()
                print("  场景已重置")

        # ── 物理步进 ────────────────────────────────────────
        if not paused:
            for _ in range(SUBSTEPS):
                world.step(dt_sub)

        # ── 渲染 ────────────────────────────────────────────
        n_verts = _build_render_buffers(world, n_static)

        canvas.set_background_color((0.06, 0.07, 0.12))
        if n_verts >= 2:
            canvas.lines(
                seg_field,
                width=0.004,
                per_vertex_color=col_field,
                vertex_count=n_verts,
            )

        window.show()
        frame += 1

        if frame % 180 == 0:
            n_dyn = len(world.bodies) - n_static
            print(f"  帧 {frame:>5d}  |  动态刚体: {n_dyn:>2d}")


if __name__ == "__main__":
    main()
