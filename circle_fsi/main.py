"""
main.py — 圆形刚体 FSI Demo
═══════════════════════════════════════════════════════════════

操作:
    · 左键点击 → 在鼠标位置投入新圆球 (最多 10 个)

颜色:
    深蓝/青 = 流体 (速度驱动渐变)
    灰色    = 墙壁
    彩色轮廓 = 圆形刚体 (带旋转指针)

运行:
    cd SPH/fsi/circle_fsi
    python main.py
"""

import taichi as ti

ti.init(arch=ti.cuda)

from config import DOMAIN_X, DOMAIN_Y, SUBSTEPS, PARTICLE_DIAMETER
from particles import generate_all
from simulation import FSISimulation

# 渲染半径: 粒子直径 / 域宽 × 0.75 → 轻微重叠, 呈连续水面感
RENDER_RADIUS = PARTICLE_DIAMETER / DOMAIN_X * 0.75


def main():
    data = generate_all()
    sim  = FSISimulation(data)

    window = ti.ui.Window(
        "FSI Circle Demo — 左键点击投入圆球",
        res=(900, 900),
        vsync=False,
    )
    canvas = window.get_canvas()
    frame  = 0

    print("\n开始模拟 …  左键点击投入圆球")
    print("  深蓝/青 = 流体 | 灰 = 墙壁 | 彩色轮廓 = 圆形刚体\n")

    while window.running:

        while window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.LMB:
                cx, cy = window.get_cursor_pos()
                sim.add_body(cx * DOMAIN_X, cy * DOMAIN_Y)

        for _ in range(SUBSTEPS):
            sim.substep()

        sim.update_vis()

        canvas.set_background_color((0.03, 0.04, 0.10))
        canvas.circles(sim.x_vis, radius=RENDER_RADIUS,
                       per_vertex_color=sim.c_vis)
        canvas.lines(sim.corners, width=0.003, color=(1.0, 0.9, 0.3))
        window.show()

        frame += 1
        if frame % 200 == 0:
            nb = int(sim.n_bodies[None])
            na = int(sim.n_active[None])
            print(f"  帧 {frame:>5d}  |  圆球 {nb}  |  活跃粒子 {na}")


if __name__ == "__main__":
    main()
