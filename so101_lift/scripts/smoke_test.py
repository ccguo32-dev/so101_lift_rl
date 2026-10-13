"""Smoke test for :class:`SO101LiftEnv`.

Run from any working directory::

    C:\\Users\\35184\\Desktop\\S101\\.venv\\Scripts\\python.exe \\
        C:\\Users\\35184\\Desktop\\S101\\so101_lift\\scripts\\smoke_test.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]  # .../S101
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v3 as iio  # noqa: E402
import mujoco  # noqa: E402

from so101_lift.envs.so101_lift_env import SO101LiftEnv  # noqa: E402

RENDER_DIR = PROJECT_ROOT / "so101_lift" / "renders"


def main() -> int:
    env = SO101LiftEnv(render_mode="rgb_array")

    obs, info = env.reset(seed=1234)

    print("=== SO-101 lift smoke test ===")
    print(f"obs dims      : {env.observation_space.shape}")
    print(f"action dims   : {env.action_space.shape}")
    print(f"n_substeps    : {env.n_substeps}")
    print(f"control_dt    : {env.control_dt}")
    print(f"timestep      : {env.model.opt.timestep}")
    print("joints (name, lo, hi, ctrlrange):")
    for i, jid in enumerate(env.joint_ids):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, int(jid))
        lo, hi = env.model.jnt_range[jid]
        crange = env.model.actuator_ctrlrange[i]
        print(
            f"  {i} {name:<14} joint=[{lo:+.6f}, {hi:+.6f}] "
            f"ctrl=[{crange[0]:+.6f}, {crange[1]:+.6f}]"
        )

    assert env.observation_space.shape == (21,), env.observation_space.shape
    assert env.action_space.shape == (6,), env.action_space.shape
    assert env.n_substeps == 20, env.n_substeps

    print("\n=== episodes ===")
    spawns = []
    for ep in range(3):
        rng = np.random.default_rng(1000 + ep)
        obs, info = env.reset(seed=1000 + ep)
        spawns.append(np.asarray(info["spawn"], dtype=np.float64))
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        is_success = bool(info["is_success"])
        ever_success = is_success

        for _ in range(250):
            action = rng.uniform(-1.0, 1.0, size=6)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_reward += float(reward)

            obs_arr = np.asarray(obs)
            assert obs_arr.shape == (21,), obs_arr.shape
            assert obs_arr.dtype == np.float32, obs_arr.dtype
            assert np.isfinite(obs_arr).all(), "obs has NaN/Inf"
            assert np.isfinite(reward), "reward is not finite"
            for key in ("is_success", "cube_dz", "cube_z", "both", "dist"):
                assert key in info, f"info missing {key!r}"
            is_success = bool(info["is_success"])
            assert not (ever_success and not is_success), "is_success reverted to False"
            ever_success = ever_success or is_success
            if terminated or truncated:
                break

        print(
            f"episode {ep}: reward={total_reward:+.4f} steps={steps} "
            f"terminated={bool(terminated)} truncated={bool(truncated)} "
            f"is_success={is_success} spawn={spawns[-1].tolist()}"
        )

    print("\n=== spawn variation ===")
    for i, s in enumerate(spawns):
        print(f"  spawn[{i}] = ({s[0]:.6f}, {s[1]:.6f})")
    assert not np.allclose(spawns[0], spawns[1]) or not np.allclose(
        spawns[1], spawns[2]
    ), f"cube spawn did not vary: {spawns}"

    print("\n=== renders ===")
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        env.reset(seed=2000 + i)
        img = env.render()
        assert img is not None
        img = np.asarray(img)
        assert img.shape == (480, 640, 3), img.shape
        assert img.dtype == np.uint8, img.dtype
        mean_px = float(img.mean())
        out = RENDER_DIR / f"smoke_{i:02d}.png"
        iio.imwrite(out, img)
        print(f"  smoke_{i:02d}.png shape={img.shape} dtype={img.dtype} mean={mean_px:.3f} -> {out}")
        assert mean_px > 5.0, f"frame {i} looks black (mean={mean_px})"

    env.close()
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    try:
        _rc = main()
    except Exception:
        traceback.print_exc()
        _rc = 1
    sys.exit(_rc)
