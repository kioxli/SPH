"""
record_demo.py — 自动录制 FSI Demo 视频
═══════════════════════════════════════════════════════════════

运行:
    cd SPH/fsi/circle_fsi
    python record_demo.py

输出:
    demo.mp4  (当前目录)
"""

import os
import taichi as ti

ti.init(arch=ti.cuda)

from config import DOMAIN_X, DOMAIN_Y, SUBSTEPS, PARTICLE_DIAMETER
from particles import generate_all
from simulation import FSISimulation

RENDER_RADIUS = PARTICLE_DIAMETER / DOMAIN_X * 0.75

# ── 录制参数 ─────────────────────────────────────────────────
TOTAL_FRAMES = 600          # 总帧数
FPS          = 60           # 输出视频帧率
RES          = (900, 900)   # 窗口分辨率
OUTPUT_FILE  = "demo.mp4"

# ── 自动投入圆球时间表 ───────────────────────────────────────
# (帧号, x坐标, y坐标)
SPAWN_SCHEDULE = [
    (120, 0.5, 1.7),
    (240, 1.5, 1.8),
    (360, 1.0, 1.6),
]


def main():
    data = generate_all()
    sim  = FSISimulation(data)

    window = ti.ui.Window("Recording FSI Demo...", res=RES, vsync=False)
    canvas = window.get_canvas()

    video_mgr = ti.tools.VideoManager(
        output_dir=".",
        framerate=FPS,
        automatic_build=False,
    )

    print(f"\n开始录制 {TOTAL_FRAMES} 帧 → {OUTPUT_FILE}")
    print(f"  分辨率: {RES[0]}x{RES[1]}  |  帧率: {FPS} fps")
    print(f"  预计时长: {TOTAL_FRAMES / FPS:.1f} 秒\n")

    for frame in range(TOTAL_FRAMES):
        # 自动投入圆球
        for t, sx, sy in SPAWN_SCHEDULE:
            if frame == t:
                sim.add_body(sx, sy)

        for _ in range(SUBSTEPS):
            sim.substep()

        sim.update_vis()

        canvas.set_background_color((0.03, 0.04, 0.10))
        canvas.circles(sim.x_vis, radius=RENDER_RADIUS,
                       per_vertex_color=sim.c_vis)
        canvas.lines(sim.corners, width=0.003, color=(1.0, 0.9, 0.3))
        window.show()

        img = window.get_image_buffer_as_numpy()
        video_mgr.write_frame(img)

        if (frame + 1) % 100 == 0:
            nb = int(sim.n_bodies[None])
            print(f"  帧 {frame + 1:>4d}/{TOTAL_FRAMES}  |  圆球 {nb}")

    print("\n正在生成视频...")
    video_mgr.make_video(gif=False, mp4=True)

    # VideoManager 会生成 video.mp4, 重命名为目标文件名
    src = os.path.join(".", "video.mp4")
    if os.path.exists(src):
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        os.rename(src, OUTPUT_FILE)

    print(f"录制完成: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
