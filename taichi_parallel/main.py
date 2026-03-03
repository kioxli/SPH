"""
main.py — FSI 流固耦合 Demo 入口
═══════════════════════════════════════════════════════════════

操作:
    · 启动后自动运行溃坝 + 1 个刚体
    · 左键点击 → 在鼠标位置添加新方块 (最多 10 个)

颜色:
    蓝色 = 流体    灰色 = 墙壁    彩色 = 刚体 (每个不同颜色)

运行:
    cd SPH/fsi
    python main.py
"""

import taichi as ti

ti.init(arch=ti.cuda)

from config import DOMAIN_X, DOMAIN_Y, SUBSTEPS, PARTICLE_DIAMETER
from particles import generate_all
from simulation import FSISimulation

# 渲染半径 = 物理直径 / 域宽 × 0.75, 使粒子轻微重叠 → 更像连续水体
RENDER_RADIUS = PARTICLE_DIAMETER / DOMAIN_X * 0.75


def main():
    # ── 1. 生成粒子数据 ──────────────────────────────
    data = generate_all()

    # ── 2. 构建模拟器 ────────────────────────────────
    sim = FSISimulation(data)

    # ── 3. 创建窗口 ──────────────────────────────────
    window = ti.ui.Window(
        "FSI Demo — 左键点击添加方块",
        res=(800, 800),
        vsync=False,
    )
    canvas = window.get_canvas()
    frame = 0

    print("\n开始模拟 …  左键点击添加新方块")
    print("  蓝 = 流体 | 灰 = 墙壁 | 彩色 = 刚体\n")

    # ── 4. 主循环 ────────────────────────────────────
    while window.running:

        # 事件: 鼠标左键 → 添加刚体
        while window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.LMB:
                cx, cy = window.get_cursor_pos()
                sim.add_body(cx * DOMAIN_X, cy * DOMAIN_Y)

        for _ in range(SUBSTEPS):
            sim.substep()

        # 渲染
        sim.update_vis()

        canvas.set_background_color((0.04, 0.04, 0.08))
        canvas.circles(sim.x_vis,  radius=RENDER_RADIUS, per_vertex_color=sim.c_vis)
        canvas.lines(sim.corners, width=0.004, color=(1.0, 0.9, 0.3))
        window.show()

        frame += 1
        if frame % 200 == 0:
            nb = int(sim.n_bodies[None])
            na = int(sim.n_active[None])
            print(f"  帧 {frame:>5d}  |  刚体 {nb}  |  活跃粒子 {na}")


if __name__ == "__main__":
    main()
