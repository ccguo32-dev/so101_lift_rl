#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_so101.py -- Validate the SO-101 MuJoCo model (SO-ARM100 / Simulation/SO101).

What it does
------------
1. Loads the SO-101 MJCF with MuJoCo.
2. Prints every joint: name, type, qpos address, range (rad + deg), limited flag.
3. Prints the actuator list: name, transmission, joint, ctrlrange, forcerange, gear.
4. Renders one 640x480 frame of the whole robot -> PNG.
5. Sweeps the gripper joint from its minimum to its maximum value, renders the two
   extremes -> PNGs, and measures the jaw aperture at every step from the actual
   mesh vertices of the two jaws.
6. Decides whether the LOW joint value means CLOSED or OPEN and prints the verdict.

Why the aperture metric is what it is
-------------------------------------
The raw minimum distance between the two jaw meshes is dominated by the proximal
hinge-housing geometry (~1 mm) and stays essentially constant over the whole
sweep, so it cannot tell closed from open.  The script therefore measures the
jaw aperture as the distance between the two jaws' distal tip points, where the
distal tip of a jaw is the mesh vertex farthest from the gripper hinge anchor.
That metric is monotonic in the opening angle.  A centroid-to-centroid distance
is reported as an independent cross-check.

Run with the project venv:
    .venv\Scripts\python.exe verify_so101.py
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    sys.exit("mujoco is not installed in this interpreter: " + str(exc))

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE / "SO-ARM100"
DEFAULT_MJCF = DEFAULT_REPO / "Simulation" / "SO101" / "scene.xml"
OUT_DIR = HERE / "renders"

FIXED_JAW_MESH = "wrist_roll_follower_so101_v1"
MOVING_JAW_MESH = "moving_jaw_so101_v1"

RAD2DEG = 180.0 / math.pi


def find_mjcf(explicit):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        DEFAULT_MJCF,
        DEFAULT_REPO / "Simulation" / "SO101" / "so101_new_calib.xml",
        DEFAULT_REPO / "Simulation" / "SO101" / "so101_old_calib.xml",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise SystemExit("Could not find the SO-101 MJCF. Tried:\n  "
                     + "\n  ".join(str(c) for c in candidates))


def name_of(model, objtype, idx):
    raw = mujoco.mj_id2name(model, objtype, idx)
    if raw is None:
        return "<unnamed#%d>" % idx
    if isinstance(raw, bytes):
        raw = raw.decode()
    return raw


def model_name(model):
    try:
        first = model.names.split(b"\0")[0]
        if first:
            return first.decode("utf-8", "replace")
    except Exception:
        pass
    return "<unnamed>"


# --------------------------------------------------------------------------
# Dependency-free PNG writer (fallback when neither imageio nor PIL is present)
# --------------------------------------------------------------------------
def write_png(path, img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        h, w = img.shape
        raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
        ctype = 0
    else:
        h, w, ch = img.shape
        raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
        ctype = 2 if ch == 3 else 6

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def save_image(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
        iio.imwrite(path, img)
        return "imageio"
    except Exception:
        pass
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
        return "PIL"
    except Exception:
        pass
    write_png(path, img)
    return "builtin"


# --------------------------------------------------------------------------
# Introspection
# --------------------------------------------------------------------------
def describe_model(model, mjcf):
    line = "=" * 78
    print(line)
    print("SO-101 MuJoCo model verification")
    print(line)
    print("MJCF file      : %s" % mjcf)
    print("Model name     : %s" % model_name(model))
    print("MuJoCo version : %s" % mujoco.__version__)
    print("nq=%d  nv=%d  nu=%d  nbody=%d  ngeom=%d  nmesh=%d  nsensor=%d"
          % (model.nq, model.nv, model.nu, model.nbody, model.ngeom,
             model.nmesh, model.nsensor))
    print()

    print("-" * 78)
    print("JOINTS  (every joint in the compiled model)")
    print("-" * 78)
    print("%-22s %-7s %5s %5s %4s %13s %13s %9s %9s"
          % ("name", "type", "qpos", "dof", "lim", "range_min", "range_max",
             "min[deg]", "max[deg]"))
    joints = []
    for j in range(model.njnt):
        jname = name_of(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = int(model.jnt_type[j])
        jtype_s = {int(mujoco.mjtJoint.mjJNT_FREE): "free",
                   int(mujoco.mjtJoint.mjJNT_BALL): "ball",
                   int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
                   int(mujoco.mjtJoint.mjJNT_HINGE): "hinge"}.get(jtype, "?%d" % jtype)
        lo, hi = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
        limited = bool(model.jnt_limited[j])
        if limited:
            rng = ("%.6f" % lo, "%.6f" % hi)
            deg = ("%.2f" % (lo * RAD2DEG), "%.2f" % (hi * RAD2DEG))
        else:
            rng = deg = ("n/a", "n/a")
        print("%-22s %-7s %5d %5d %4s %13s %13s %9s %9s"
              % (jname, jtype_s, int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]),
                 "Y" if limited else "N", rng[0], rng[1], deg[0], deg[1]))
        joints.append(dict(index=j, name=jname, type=jtype_s,
                           qposadr=int(model.jnt_qposadr[j]),
                           dofadr=int(model.jnt_dofadr[j]),
                           limited=limited, lo=lo, hi=hi))
    print()

    print("-" * 78)
    print("ACTUATORS")
    print("-" * 78)
    print("%-18s %-10s %-16s %8s %10s %10s %8s %8s %5s %6s"
          % ("name", "trntype", "joint", "gear", "ctrl_min", "ctrl_max",
             "f_min", "f_max", "clim", "flim"))
    actuators = []
    for a in range(model.nu):
        aname = name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        trntype = int(model.actuator_trntype[a])
        trntype_s = {int(mujoco.mjtTrn.mjTRN_JOINT): "joint",
                     int(mujoco.mjtTrn.mjTRN_JOINTINPARENT): "jointinpar",
                     int(mujoco.mjtTrn.mjTRN_SLIDERCRANK): "slidercrank",
                     int(mujoco.mjtTrn.mjTRN_TENDON): "tendon",
                     int(mujoco.mjtTrn.mjTRN_SITE): "site",
                     int(mujoco.mjtTrn.mjTRN_BODY): "body"}.get(trntype, "?%d" % trntype)
        tid = int(model.actuator_trnid[a][0])
        if trntype in (int(mujoco.mjtTrn.mjTRN_JOINT),
                       int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
            tname = name_of(model, mujoco.mjtObj.mjOBJ_JOINT, tid) if tid >= 0 else "-"
        else:
            tname = "-"
        cr = model.actuator_ctrlrange[a]
        fr = model.actuator_forcerange[a]
        print("%-18s %-10s %-16s %8.4f %10.4f %10.4f %8.2f %8.2f %5s %6s"
              % (aname, trntype_s, tname, float(model.actuator_gear[a][0]),
                 cr[0], cr[1], fr[0], fr[1],
                 "Y" if model.actuator_ctrllimited[a] else "N",
                 "Y" if model.actuator_forcelimited[a] else "N"))
        actuators.append(dict(index=a, name=aname, trntype=trntype_s, joint=tname,
                              gear=float(model.actuator_gear[a][0]),
                              ctrl=(float(cr[0]), float(cr[1])),
                              force=(float(fr[0]), float(fr[1]))))
    print()
    print("actuators %d, joints %d; map: %s"
          % (model.nu, model.njnt, ", ".join("%s->%s" % (x["name"], x["joint"])
                                             for x in actuators)))
    print()
    return joints, actuators


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_frame(model, data, out_png, renderer, width=640, height=480,
                 lookat=None, distance=0.5, azimuth=135.0, elevation=-20.0):
    if renderer is None:
        renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
    if lookat is not None:
        cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    how = save_image(out_png, img)
    print("  wrote %s  (%dx%d, %s)" % (out_png, img.shape[1], img.shape[0], how))
    return img, renderer


# --------------------------------------------------------------------------
# Gripper sweep
# --------------------------------------------------------------------------
def world_points(model, data, geom_id, stride=3):
    mid = int(model.geom_dataid[geom_id])
    if mid < 0:
        return np.zeros((0, 3))
    vadr, vnum = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    verts = np.asarray(model.mesh_vert[vadr:vadr + vnum], dtype=np.float64)[::stride]
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    return verts @ mat.T + pos


def min_cloud_distance(points_a, points_b, step=512):
    if len(points_a) == 0 or len(points_b) == 0:
        return float("nan")
    best = float("inf")
    for i in range(0, len(points_a), step):
        chunk = points_a[i:i + step]
        d2 = ((chunk[:, None, :] - points_b[None, :, :]) ** 2).sum(-1)
        best = min(best, float(np.sqrt(d2.min())))
    return best


def geom_by_mesh(model, mesh_stem):
    for g in range(model.ngeom):
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(model.geom_dataid[g])
        if mid < 0:
            continue
        fname = name_of(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        if mesh_stem.lower() in str(fname).lower():
            return g
    return None


def sweep_gripper(model, data, joints, renderer, out_dir):
    gj = next((j for j in joints if j["name"] == "gripper"), None)
    if gj is None:
        print("!! no joint named gripper; cannot sweep")
        return {}, renderer, {}
    qposadr, lo, hi = gj["qposadr"], gj["lo"], gj["hi"]
    gjid = gj["index"]

    fixed_g = geom_by_mesh(model, FIXED_JAW_MESH)
    moving_g = geom_by_mesh(model, MOVING_JAW_MESH)
    print("fixed  jaw mesh geom id = %s  (%s)" % (fixed_g, FIXED_JAW_MESH))
    print("moving jaw mesh geom id = %s  (%s)" % (moving_g, MOVING_JAW_MESH))

    n = 61
    qs = np.linspace(lo, hi, n)
    tip_gap = np.full(n, np.nan)
    cen_gap = np.full(n, np.nan)
    raw_min = np.full(n, np.nan)
    tip_fixed_all = None
    for k, q in enumerate(qs):
        data.qpos[:] = model.qpos0
        data.qpos[qposadr] = q
        mujoco.mj_forward(model, data)
        if fixed_g is None or moving_g is None:
            continue
        pf = world_points(model, data, fixed_g)
        pm = world_points(model, data, moving_g)
        anchor = np.asarray(data.xanchor[gjid], dtype=np.float64)
        tip_f = pf[int(np.argmax(np.linalg.norm(pf - anchor, axis=1)))]
        tip_m = pm[int(np.argmax(np.linalg.norm(pm - anchor, axis=1)))]
        tip_gap[k] = float(np.linalg.norm(tip_f - tip_m))
        cen_gap[k] = float(np.linalg.norm(pf.mean(0) - pm.mean(0)))
        raw_min[k] = min_cloud_distance(pf, pm)
        if tip_fixed_all is None:
            tip_fixed_all = tip_f

    print()
    print("-" * 78)
    print("GRIPPER SWEEP  joint 'gripper'  qmin=%.6f rad (%.2f deg) -> qmax=%.6f rad (%.2f deg)"
          % (lo, lo * RAD2DEG, hi, hi * RAD2DEG))
    print("aperture = distance between the two jaws' distal tip points (mm)")
    print("-" * 78)
    print("%10s %10s %14s %14s %14s" % ("q[rad]", "q[deg]", "tip aperture", "centroid gap", "raw min dist"))
    for k in range(n):
        if k % 5 == 0 or k == n - 1:
            print("%10.4f %10.2f %14.3f %14.3f %14.3f"
                  % (qs[k], qs[k] * RAD2DEG, tip_gap[k] * 1000.0,
                     cen_gap[k] * 1000.0, raw_min[k] * 1000.0))

    d_tip = np.diff(tip_gap)
    up = int(np.sum(d_tip > 0))
    down = int(np.sum(d_tip < 0))
    slope = float(np.polyfit(qs, tip_gap, 1)[0])
    gap_lo = float(np.mean(tip_gap[:3]))
    gap_hi = float(np.mean(tip_gap[-3:]))
    cen_lo = float(np.mean(cen_gap[:3]))
    cen_hi = float(np.mean(cen_gap[-3:]))

    if gap_hi > gap_lo * 1.3 and up >= 0.8 * (n - 1):
        verdict = "LOW q  ==> CLOSED ;  HIGH q ==> OPEN   (jaw aperture grows with q)"
    elif gap_lo > gap_hi * 1.3 and down >= 0.8 * (n - 1):
        verdict = "LOW q  ==> OPEN   ;  HIGH q ==> CLOSED (jaw aperture shrinks with q)"
    else:
        verdict = "AMBIGUOUS (inspect the rendered frames)"

    print()
    print("tip aperture at qmin (mean first 3) = %.3f mm" % (gap_lo * 1000.0))
    print("tip aperture at qmax (mean last  3) = %.3f mm" % (gap_hi * 1000.0))
    print("centroid gap at qmin / qmax         = %.3f / %.3f mm" % (cen_lo * 1000.0, cen_hi * 1000.0))
    print("monotonic steps (increasing q): up=%d down=%d of %d" % (up, down, n - 1))
    print("d(tip aperture)/dq = %+.6f m/rad" % slope)
    print("raw min mesh distance is %.3f-%.3f mm over the whole sweep -> dominated by the"
          " hinge housing, NOT a usable opening proxy"
          % (np.nanmin(raw_min) * 1000.0, np.nanmax(raw_min) * 1000.0))
    print("VERDICT  : %s" % verdict)

    # Close-up camera framing on the jaw tips.
    data.qpos[:] = model.qpos0
    data.qpos[qposadr] = 0.5 * (lo + hi)
    mujoco.mj_forward(model, data)
    pf = world_points(model, data, fixed_g)
    pm = world_points(model, data, moving_g)
    lookat = 0.5 * (pf.mean(0) + pm.mean(0))
    span = float(np.linalg.norm(pf.mean(0) - pm.mean(0)))
    dist = max(0.12, 2.6 * span)

    frames = {}
    for tag, q in (("qmin", lo), ("qmax", hi)):
        data.qpos[:] = model.qpos0
        data.qpos[qposadr] = q
        mujoco.mj_forward(model, data)
        img, renderer = render_frame(model, data, out_dir / ("gripper_%s.png" % tag),
                                     renderer, lookat=lookat, distance=dist,
                                     azimuth=90.0, elevation=-25.0)
        frames[tag] = img

    summary = dict(qmin=lo, qmax=hi,
                   tip_aperture_at_qmin=gap_lo, tip_aperture_at_qmax=gap_hi,
                   centroid_gap_at_qmin=cen_lo, centroid_gap_at_qmax=cen_hi,
                   monotonic_up=up, monotonic_down=down, slope_daperture_dq=slope,
                   verdict=verdict)
    series = dict(q=[float(x) for x in qs],
                  tip_aperture_mm=[float(x * 1000.0) for x in tip_gap],
                  centroid_gap_mm=[float(x * 1000.0) for x in cen_gap],
                  raw_min_mm=[float(x * 1000.0) for x in raw_min])
    return summary, renderer, series


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Verify the SO-101 MuJoCo model.")
    ap.add_argument("--mjcf", default=None,
                    help="path to the MJCF (default: SO-ARM100/Simulation/SO101/scene.xml)")
    ap.add_argument("--outdir", default=str(OUT_DIR), help="directory for PNG output")
    ap.add_argument("--no-render", action="store_true", help="introspection only")
    args = ap.parse_args()

    out_dir = Path(args.outdir).resolve()
    mjcf = find_mjcf(args.mjcf)
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)

    joints, actuators = describe_model(model, mjcf)

    if args.no_render:
        return 0

    print("-" * 78)
    print("RENDERING")
    print("-" * 78)
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception as exc:
        print("!! offscreen MuJoCo renderer unavailable: %r" % exc)
        print("   introspection above is still valid; skipping images.")
        return 0

    mujoco.mj_forward(model, data)
    img, renderer = render_frame(model, data, out_dir / "so101_frame.png", renderer,
                                 lookat=np.array([0.0, 0.0, 0.12]), distance=0.55,
                                 azimuth=135.0, elevation=-18.0)
    print("  frame shape=%s dtype=%s min=%d max=%d"
          % (img.shape, img.dtype, int(img.min()), int(img.max())))

    summary, renderer, series = sweep_gripper(model, data, joints, renderer, out_dir)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("MJCF            : %s" % mjcf)
    print("joints (%d)     : %s" % (len(joints), ", ".join(j["name"] for j in joints)))
    print("actuators (%d)  : %s" % (len(actuators), ", ".join(a["name"] for a in actuators)))
    if summary:
        print("gripper qmin/hi : %.6f / %.6f rad  (%.2f / %.2f deg)"
              % (summary["qmin"], summary["qmax"],
                 summary["qmin"] * RAD2DEG, summary["qmax"] * RAD2DEG))
        print("tip aperture    : %.3f mm (qmin) -> %.3f mm (qmax)"
              % (summary["tip_aperture_at_qmin"] * 1000.0,
                 summary["tip_aperture_at_qmax"] * 1000.0))
        print("verdict         : %s" % summary["verdict"])
    print("images          : %s" % out_dir)
    print("=" * 78)

    with open(out_dir / "verify_so101_result.json", "w", encoding="utf-8") as fh:
        json.dump(dict(mjcf=str(mjcf), model_name=model_name(model),
                       mujoco_version=mujoco.__version__,
                       joints=joints, actuators=actuators,
                       gripper=summary, sweep=series), fh, indent=2)
    print("wrote %s" % (out_dir / "verify_so101_result.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
