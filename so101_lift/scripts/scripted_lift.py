#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripted_lift.py -- prove the SO-101 lift task is solvable with a hand-written
state machine, no learning involved.

State machine
-------------
    1. OPEN    : open the gripper fully
    2. ALIGN   : move the end effector to 5 cm directly above the cube
    3. DESCEND : Cartesian-interpolate the EE straight down to grasp height
    4. CLOSE   : close the gripper onto the cube
    5. LIFT    : Cartesian-interpolate the EE straight up by 8 cm
    6. DONE    : report success

The arm is driven through the environment's real action interface: every step we
solve IK for a desired joint vector and feed clip((q_desired - q_target)/0.05)
as the 5 arm actions, so the scripted controller obeys exactly the same rate
limit a learned policy would.

IK
--
mj_jacSite + damped least squares.  The trick that makes a top-down grasp
reliable is that the fingertip ee_site sits on the wrist-roll axis, so rotating
wrist_roll never moves it.  We therefore fix wrist_flex and wrist_roll, solve the
first three joints for a position, and pick the wrist_flex that leaves the
gripper pointing straight down.

Run:
    .venv\Scripts\python.exe so101_lift\scripts\scripted_lift.py --episodes 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402

from so101_lift.envs.so101_lift_env import SO101LiftEnv  # noqa: E402

RENDER_DIR = PROJECT_ROOT / "so101_lift" / "renders"

WRIST_FLEX_RANGE = (0.70, 1.45)
WRIST_FLEX_SAMPLES = 16


class IkSolver:
    """Position IK for the SO-101 using mj_jacSite + damped least squares."""

    def __init__(self, model, data, ee_site_id, gripper_body_id):
        self.model = model
        self.data = data
        self.ee_site_id = int(ee_site_id)
        self.gripper_body_id = int(gripper_body_id)
        self.lo = np.array([model.jnt_range[i][0] for i in range(6)], dtype=np.float64)
        self.hi = np.array([model.jnt_range[i][1] for i in range(6)], dtype=np.float64)
        self._J = np.zeros((3, model.nv))
        self._Jr = np.zeros((3, model.nv))

    def _forward(self, q5):
        self.data.qpos[:5] = q5
        mujoco.mj_forward(self.model, self.data)

    def ee_pos(self):
        return np.asarray(self.data.site_xpos[self.ee_site_id], dtype=np.float64).copy()

    def ee_axes(self):
        mat = np.asarray(self.data.xmat[self.gripper_body_id], dtype=np.float64)
        return mat.reshape(3, 3).copy()

    def _jac(self, free_joints):
        self._J[:] = 0.0
        self._Jr[:] = 0.0
        mujoco.mj_jacSite(self.model, self.data, self._J, self._Jr, self.ee_site_id)
        return self._J[:, list(free_joints)]

    def solve_fixed_wrist(self, target, q3, q4, seeds, iters=400, damp=1e-2, step=0.35):
        """Solve joints 0..2 for target, with wrist_flex=q3 and wrist_roll=q4 fixed."""
        target = np.asarray(target, dtype=np.float64)
        best = None
        for seed in seeds:
            q = np.array([seed[0], seed[1], seed[2], q3, q4], dtype=np.float64)
            for _ in range(iters):
                self._forward(q)
                err = target - self.ee_pos()
                if float(np.linalg.norm(err)) < 3.0e-4:
                    break
                jac = self._jac(range(3))
                dq = jac.T @ np.linalg.solve(jac @ jac.T + damp * np.eye(3), err)
                q[:3] = np.clip(q[:3] + step * np.clip(dq, -0.35, 0.35), self.lo[:3], self.hi[:3])
            self._forward(q)
            res = float(np.linalg.norm(target - self.ee_pos()))
            if best is None or res < best[0]:
                best = (res, q.copy())
        return best

    def solve_arm(self, target, q_seed, iters=300, damp=1e-2, step=0.5):
        """General 5-joint position IK, seeded so it stays in the right basin."""
        target = np.asarray(target, dtype=np.float64)
        q = np.array(q_seed, dtype=np.float64).copy()
        for _ in range(iters):
            self._forward(q)
            err = target - self.ee_pos()
            if float(np.linalg.norm(err)) < 3.0e-4:
                break
            jac = self._jac(range(5))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damp * np.eye(3), err)
            q[:5] = np.clip(q[:5] + step * np.clip(dq, -0.25, 0.25), self.lo[:5], self.hi[:5])
        self._forward(q)
        return q, float(np.linalg.norm(target - self.ee_pos()))

    def find_grasp(self, target, verbose=False):
        """Best straight-down grasp pose reaching target (the cube centre)."""
        seeds = [(0.0, 0.3, -0.2), (0.0, 0.5, -0.5), (0.0, 0.0, 0.0),
                 (0.0, 0.8, -0.9), (0.0, 0.2, 0.1), (0.0, 1.0, -1.2),
                 (0.4, 0.4, -0.3), (-0.4, 0.4, -0.3)]
        seeds += [(s0, s1, s2)
                  for s0 in np.linspace(self.lo[0], self.hi[0], 3)
                  for s1 in np.linspace(self.lo[1], self.hi[1], 3)
                  for s2 in np.linspace(self.lo[2], self.hi[2], 3)]

        best = None
        for q3 in np.linspace(WRIST_FLEX_RANGE[0], WRIST_FLEX_RANGE[1], WRIST_FLEX_SAMPLES):
            for q4 in (0.0, -0.15, 0.15):
                r = self.solve_fixed_wrist(target, q3, q4, seeds)
                if r is None or r[0] > 1.5e-3:
                    continue
                res, q = r
                self._forward(q)
                rot = self.ee_axes()
                appr = -rot[:, 2]
                appr = appr / max(1e-9, float(np.linalg.norm(appr)))
                sep = rot[:, 1]
                sep = sep / max(1e-9, float(np.linalg.norm(sep)))
                score = float(appr[2]) - 0.5 * abs(float(sep[2])) + res
                if best is None or score < best[0]:
                    best = (score, q.copy(), res, appr.copy(), sep.copy())
        if verbose and best is not None:
            print("  grasp: appr.z=%.4f |sep.z|=%.4f pos_err=%.4f mm q=%s"
                  % (best[3][2], abs(best[4][2]), best[2] * 1000.0, np.round(best[1], 4)))
        return best


class ScriptedLift:
    """The state machine, driven through env.step."""

    def __init__(self, env, verbose=False):
        self.env = env
        self.verbose = verbose
        self.solver = IkSolver(env.model, env.data, env.ee_site_id,
                               mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper"))
        self.lo, self.hi = env.lo, env.hi

    def _clip5(self, q5):
        return np.clip(np.asarray(q5, dtype=np.float64), self.lo[:5], self.hi[:5])

    def _actions_for(self, q_desired5, grip_value):
        delta = self._clip5(q_desired5) - self.env.arm_target[:5]
        a = np.clip(delta / 0.05, -1.0, 1.0)
        lo5, hi5 = float(self.lo[5]), float(self.hi[5])
        a5 = 2.0 * (float(grip_value) - lo5) / (hi5 - lo5) - 1.0
        return np.concatenate([a, [np.clip(a5, -1.0, 1.0)]])

    def _drive(self, q_desired5, grip_value, steps, frames=None, record=False):
        last = None
        for _ in range(steps):
            action = self._actions_for(q_desired5, grip_value)
            last = self.env.step(action)
            if record and frames is not None:
                frames.append(self.env.render())
            if last[2]:
                break
        return last

    def _cartesian(self, targets, q_seed, grip_value, steps_per_segment, frames=None, record=False):
        q_seed = np.asarray(q_seed, dtype=np.float64).copy()
        last = None
        for k in range(1, len(targets)):
            a = np.asarray(targets[k - 1], dtype=np.float64)
            b = np.asarray(targets[k], dtype=np.float64)
            n = max(1, int(steps_per_segment))
            for t in np.linspace(1.0 / n, 1.0, n):
                point = a + (b - a) * t
                q_seed, err = self.solver.solve_arm(point, q_seed)
                last = self._drive(q_seed[:5], grip_value, 1, frames, record)
        return last

    def run(self, record=False):
        env = self.env
        obs, info = env.reset()
        cube = np.asarray(env.data.xpos[env.cube_body_id], dtype=np.float64).copy()

        grasp = self.solver.find_grasp(cube, verbose=self.verbose)
        if grasp is None:
            return False, dict(failure="ik_no_grasp", cube=cube)
        _, q_grasp, _, appr, _ = grasp

        above = cube + np.array([0.0, 0.0, 0.05])
        lift = cube + np.array([0.0, 0.0, 0.08])
        q_above, _ = self.solver.solve_arm(above, q_grasp)
        q_lift, _ = self.solver.solve_arm(lift, q_grasp)

        frames = [] if record else None
        open_grip = float(self.hi[5])
        close_grip = float(self.lo[5])

        self._drive(q_above[:5], open_grip, 45, frames, record)
        self._drive(q_above[:5], open_grip, 25, frames, record)

        self._cartesian([self.solver.ee_pos(), above, cube], q_above, open_grip, 22, frames, record)
        self._drive(q_grasp[:5], open_grip, 15, frames, record)

        self._drive(q_grasp[:5], close_grip, 30, frames, record)
        closed = dict(env.last_info)

        self._cartesian([cube, cube + np.array([0.0, 0.0, 0.04]), lift],
                        q_grasp, close_grip, 25, frames, record)
        self._drive(q_lift[:5], close_grip, 25, frames, record)

        final = dict(env.last_info)
        result = dict(cube=cube, q_grasp=q_grasp, appr_z=float(appr[2]),
                      closed_both=bool(closed.get("both", False)),
                      cube_dz=float(final.get("cube_dz", 0.0)),
                      is_success=bool(final.get("is_success", False)),
                      frames=frames)
        return result["is_success"], result


def main():
    ap = argparse.ArgumentParser(description="Scripted (non-learned) SO-101 lift demo")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--record-episode", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    env = SO101LiftEnv(render_mode=None if args.no_render else "rgb_array",
                       max_episode_steps=args.max_steps)

    print("=" * 78)
    print("SO-101 scripted lift")
    print("=" * 78)
    print("obs  dims :", env.observation_space.shape, env.observation_space.dtype)
    print("act  dims :", env.action_space.shape, env.action_space.dtype)
    print("n_substeps:", env.n_substeps, " control_dt:", env.control_dt,
          " physics_dt:", env.model.opt.timestep)
    print("attempts  : %d   base seed: %d" % (args.episodes, args.seed))
    print("-" * 78)

    results = []
    recorded = None
    t0 = time.time()
    for ep in range(args.episodes):
        record = (args.record_episode == ep) or (args.record_episode is None and recorded is None)
        env.reset(seed=args.seed + ep)
        policy = ScriptedLift(env, verbose=args.verbose)
        ok, res = policy.run(record=bool(record and not args.no_render))
        results.append(bool(ok))
        print("episode %2d: %-7s cube_start=(%.3f,%+.3f,%.4f) appr.z=%+.4f "
              "closed_both=%s cube_dz=%+.4f"
              % (ep, "SUCCESS" if ok else "fail", res["cube"][0], res["cube"][1],
                 res["cube"][2], res.get("appr_z", float("nan")),
                 res.get("closed_both"), res.get("cube_dz", float("nan"))))
        if ok and recorded is None and res.get("frames"):
            recorded = (ep, res["frames"])

    n_ok = int(sum(results))
    print("-" * 78)
    print("successes: %d / %d  (%.0f%%)   wall time %.1f s"
          % (n_ok, len(results), 100.0 * n_ok / max(1, len(results)), time.time() - t0))

    if recorded is not None:
        ep, frames = recorded
        try:
            import imageio.v3 as iio
            frames = [np.asarray(f) for f in frames]
            gif_path = RENDER_DIR / "scripted_lift_success.gif"
            iio.imwrite(gif_path, frames, duration=1000.0 / 25.0, loop=0)
            print("wrote %s  (%d frames, %s)" % (gif_path, len(frames), frames[0].shape))
            picks = np.linspace(0, len(frames) - 1, 4).astype(int)
            for i, k in enumerate(picks):
                p = RENDER_DIR / ("scripted_lift_ep%02d_%02d.png" % (ep, i))
                iio.imwrite(p, frames[k])
                print("wrote %s" % p)
        except Exception as exc:
            print("could not save GIF:", repr(exc))

    env.close()
    print("=" * 78)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
