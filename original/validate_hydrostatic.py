"""
validate_hydrostatic.py — Hydrostatic pressure validation
═══════════════════════════════════════════════════════════════

Fill domain with a static fluid column, let it settle, then compare
measured SPH pressure against the analytical solution:

    p(y) = ρ₀ · g · (H − y)

where H is the free-surface height.

Usage:
    python validate_hydrostatic.py
"""

import taichi as ti
ti.init(arch=ti.cuda)

import numpy as np

from config import (
    DOMAIN_X, DOMAIN_Y, PARTICLE_DIAMETER, PARTICLE_RADIUS,
    SUPPORT_RADIUS, DENSITY_0, STIFFNESS, EXPONENT,
    WALL_LAYERS, M_V0, BODY0_X, BODY0_Y,
)
from particles import (
    generate_walls, generate_rigid_template, compute_boundary_psi,
)
from simulation import FSISimulation

# ── 1. Generate full-width fluid column ────────────────────

FLUID_HEIGHT = 1.0   # meters
GRAVITY = 9.81

dx = PARTICLE_DIAMETER
margin = dx * (WALL_LAYERS + 0.5)

xs = np.arange(margin, DOMAIN_X - margin + 1e-6, dx)
ys = np.arange(margin, FLUID_HEIGHT, dx)
xx, yy = np.meshgrid(xs, ys, indexing='ij')
fluid_pos = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
n_fluid = len(fluid_pos)

wall_pos, n_wall = generate_walls()
rigid_local, n_rigid_per_body = generate_rigid_template()
psi_wall  = compute_boundary_psi(wall_pos)
psi_rigid = compute_boundary_psi(rigid_local)

data = dict(
    fluid_pos=fluid_pos, n_fluid=n_fluid,
    wall_pos=wall_pos,   n_wall=n_wall,   psi_wall=psi_wall,
    rigid_local=rigid_local,
    n_rigid_per_body=n_rigid_per_body,
    psi_rigid=psi_rigid,
)

print(f"Fluid particles : {n_fluid}")
print(f"Fluid height    : {FLUID_HEIGHT} m")
print(f"Particle radius : {PARTICLE_RADIUS}")
print()

# ── 2. Build simulation, move rigid body out of the way ────

sim = FSISimulation(data)
sim.b_pos[0] = ti.Vector([DOMAIN_X / 2.0, DOMAIN_Y - 0.05])

# ── 3. Settle to equilibrium ──────────────────────────────

N_SETTLE = 4000
print(f"Settling for {N_SETTLE} substeps ...")
for step in range(N_SETTLE):
    sim.substep()
    if (step + 1) % 1000 == 0:
        print(f"  step {step + 1}/{N_SETTLE}")

ti.sync()

# ── 4. Collect and analyse ─────────────────────────────────

na = int(sim.n_active[None])
x_np   = sim.x.to_numpy()[:na]
prs_np = sim.prs.to_numpy()[:na]
rho_np = sim.rho.to_numpy()[:na]
pt_np  = sim.ptype.to_numpy()[:na]

fluid_mask = pt_np == 0
x_fluid = x_np[fluid_mask]
p_fluid = prs_np[fluid_mask]

y_vals   = x_fluid[:, 1]
y_bottom = y_vals.min()
y_top    = y_vals.max()
H = y_top + PARTICLE_RADIUS

N_BINS = 8
bins = np.linspace(y_bottom, y_top, N_BINS + 1)

print()
print(f"  Free surface H ~ {H:.3f} m")
print()
print(f"  {'y (m)':>8s}  {'p_SPH':>10s}  {'p_theory':>10s}  {'error':>8s}")
print(f"  {'--------':>8s}  {'----------':>10s}  {'----------':>10s}  {'--------':>8s}")

errors = []
for k in range(N_BINS):
    mask = (y_vals >= bins[k]) & (y_vals < bins[k + 1])
    if mask.sum() < 3:
        continue
    y_avg  = y_vals[mask].mean()
    p_avg  = p_fluid[mask].mean()
    p_theo = DENSITY_0 * GRAVITY * max(0.0, H - y_avg)
    err = abs(p_avg - p_theo) / max(p_theo, 1.0) * 100.0
    errors.append(err)
    print(f"  {y_avg:>8.3f}  {p_avg:>10.1f}  {p_theo:>10.1f}  {err:>6.1f} %")

avg_err = float(np.mean(errors)) if errors else 999.0
print()
print(f"  Mean error : {avg_err:.1f} %")
print()

if avg_err < 20.0:
    print("  [PASS] Hydrostatic pressure test PASSED")
else:
    print("  [FAIL] Hydrostatic pressure test FAILED (error > 20%)")
    print("         WCSPH inherently has pressure oscillation;")
    print("         check STIFFNESS / EXPONENT / particle spacing.")
