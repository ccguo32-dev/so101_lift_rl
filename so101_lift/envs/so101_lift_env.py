"""SO-101 lift task environment (Gymnasium) built directly on MuJoCo.

The scene is ``so101_lift/scenes/lift_scene.xml``.  Everything the policy needs
(joint limits, site ids, body/geom ids, substep count) is read from the compiled
model, so nothing geometric is hard-coded here.

Rendering is fully headless: ``mujoco.Renderer`` is used instead of a viewer, so
this works on machines with no display.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- make the project importable no matter where this file is executed from ---
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gymnasium  # noqa: E402
import mujoco  # noqa: E402


class SO101LiftEnv(gymnasium.Env):
    """Lift a 3 cm cube off the table and hold it up with both gripper jaws."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 25}

    #: arm joints, in qpos/actuator order -- read from the model, kept only for
    #: reporting (all real lookups go through the model).
    JOINT_NAMES = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )

    _JAW_MESHES = {
        # body name -> mesh name, per the scene
        "gripper": "wrist_roll_follower_so101_v1",
        "moving_jaw_so101_v1": "moving_jaw_so101_v1",
    }

    def __init__(
        self,
        render_mode=None,
        max_episode_steps=250,
        control_freq=25.0,
        image_width=640,
        image_height=480,
        xml_path=None,
        verbose=False,
    ):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode

        if xml_path is None:
            xml_path = Path(__file__).resolve().parents[1] / "scenes" / "lift_scene.xml"
        self.xml_path = Path(xml_path)

        self.max_episode_steps = int(max_episode_steps)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.verbose = bool(verbose)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # --- control decimation: 25 Hz control over a 2 ms physics step ---
        self.n_substeps = max(
            1, round(1.0 / (control_freq * self.model.opt.timestep))
        )
        self.control_dt = self.n_substeps * self.model.opt.timestep
        if self.verbose:
            print(
                f"[SO101LiftEnv] n_substeps={self.n_substeps} "
                f"control_dt={self.control_dt:.6f} "
                f"(control_freq={control_freq}, timestep={self.model.opt.timestep})"
            )

        # --- joint limits / addresses (never hard-coded) ---
        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0):
            raise RuntimeError(f"could not resolve arm joints in {self.xml_path}")
        self.qpos_adr = np.array(
            [self.model.jnt_qposadr[j] for j in self.joint_ids], dtype=np.int32
        )
        self.joint_ranges = np.array(
            [self.model.jnt_range[j] for j in self.joint_ids], dtype=np.float64
        )
        self.lo = self.joint_ranges[:, 0].copy()
        self.hi = self.joint_ranges[:, 1].copy()

        # --- cube ---
        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        if self.cube_body_id < 0:
            raise RuntimeError("body 'cube' not found in scene")
        self.cube_geom_ids = [
            g
            for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == self.cube_body_id
        ]
        self.cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
        )
        if self.cube_joint_id < 0:
            # fall back to the free joint owned by the cube body
            self.cube_joint_id = next(
                j
                for j in range(self.model.njnt)
                if int(self.model.jnt_bodyid[j]) == self.cube_body_id
            )
        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.cube_qvel_adr = int(self.model.jnt_dofadr[self.cube_joint_id])

        # --- sites ---
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        if self.ee_site_id < 0:
            raise RuntimeError("site 'ee_site' not found in scene")

        # --- jaw geoms: the two meshes clamped onto the cube ---
        # NOTE: each jaw has both a visual and a collision mesh geom in the
        # scene; both copies are collected so contact tests never miss a hit.
        jaw_geoms = self._resolve_jaw_geoms()
        self.fixed_jaw_geom_ids = [int(g) for g in jaw_geoms["fixed"]]
        self.moving_jaw_geom_ids = [int(g) for g in jaw_geoms["moving"]]
        self.jaw_geom_ids = self.fixed_jaw_geom_ids + self.moving_jaw_geom_ids
        self._fixed_jaw_geoms = set(self.fixed_jaw_geom_ids)
        self._moving_jaw_geoms = set(self.moving_jaw_geom_ids)
        self._cube_geoms = set(int(g) for g in self.cube_geom_ids)

        # --- spaces ---
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        # --- runtime state ---
        self.arm_target = self.data.qpos[self.qpos_adr].astype(np.float64).copy()
        self.cube_z0 = float(self.data.xpos[self.cube_body_id][2])
        self._elapsed = 0
        self._success = False
        self.ee_contact_prev = False
        self._info = {"is_success": False, "success_step": None}
        self.last_info = dict(self._info)

        # --- rendering (lazy) ---
        self._renderer = None
        self.cam_lookat = np.array([0.20, 0.0, 0.10], dtype=np.float64)
        self.cam_distance = 0.75
        self.cam_azimuth = 160.0
        self.cam_elevation = -15.0

    # ------------------------------------------------------------------ helpers
    def _resolve_jaw_geoms(self):
        """Find the gripper jaw geoms by mesh name (visual + collision copies)."""
        found = {"fixed": [], "moving": []}
        for g in range(self.model.ngeom):
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = int(self.model.geom_dataid[g])
            if mesh_id < 0:
                continue
            mesh_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
            )
            if mesh_name == self._JAW_MESHES["gripper"]:
                found["fixed"].append(g)
            elif mesh_name == self._JAW_MESHES["moving_jaw_so101_v1"]:
                found["moving"].append(g)
        for key, ids in found.items():
            if not ids:
                raise RuntimeError(f"could not find jaw geoms for {key!r} jaw")
        return found

    def _contact_flags(self):
        """Which jaws are currently touching the cube (any cube geom counts)."""
        contact_fixed = False
        contact_moving = False
        for c in range(self.data.ncon):
            g1 = int(self.data.contact[c].geom1)
            g2 = int(self.data.contact[c].geom2)
            pair = {g1, g2}
            if not pair & self._cube_geoms:
                continue
            if g1 in self._fixed_jaw_geoms or g2 in self._fixed_jaw_geoms:
                contact_fixed = True
            if g1 in self._moving_jaw_geoms or g2 in self._moving_jaw_geoms:
                contact_moving = True
        return contact_fixed, contact_moving

    def get_obs(self):
        """Current 21-dim observation."""
        ee_pos = np.asarray(self.data.site_xpos[self.ee_site_id], dtype=np.float64)
        cube_pos = np.asarray(self.data.xpos[self.cube_body_id], dtype=np.float64)
        obs = np.concatenate(
            [
                self.data.qpos[:6],
                self.data.qvel[:6],
                ee_pos,
                cube_pos,
                cube_pos - ee_pos,
            ]
        )
        return np.asarray(obs, dtype=np.float32)

    # ------------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random  # do not create a second RNG

        mujoco.mj_resetData(self.model, self.data)

        # arm: zero pose plus a little joint noise, clipped to the joint limits
        arm_qpos = np.zeros(6, dtype=np.float64) + rng.uniform(-0.02, 0.02, size=6)
        arm_qpos = np.clip(arm_qpos, self.lo, self.hi)
        self.data.qpos[self.qpos_adr] = arm_qpos

        # cube: uniformly sampled position on the table top
        spawn_x = float(rng.uniform(0.15, 0.28))
        spawn_y = float(rng.uniform(-0.10, 0.10))
        yaw = float(rng.uniform(-0.3, 0.3))
        cube_z = 0.015 + 1e-4  # tiny margin, corrected after mj_forward
        self.data.qpos[
            self.cube_qpos_adr : self.cube_qpos_adr + 3
        ] = (spawn_x, spawn_y, cube_z)
        self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = (
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        )

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # the cube half-edge is exactly 0.015 -> sit it exactly on the table
        self.data.qpos[self.cube_qpos_adr + 2] = 0.015
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.arm_target = self.data.qpos[self.qpos_adr].astype(np.float64).copy()
        self.data.ctrl[:6] = self.arm_target
        self.cube_z0 = float(self.data.xpos[self.cube_body_id][2])
        self._elapsed = 0
        self._success = False
        self.ee_contact_prev = False
        self._info = {"is_success": False, "success_step": None}

        cube_xy = np.asarray(self.data.xpos[self.cube_body_id][:2], dtype=np.float64)
        info = {
            "is_success": False,
            "success_step": None,
            "cube_z0": self.cube_z0,
            "cube_xy": cube_xy,
            "spawn": (spawn_x, spawn_y),
        }
        self._info = info
        self.last_info = info
        return self.get_obs(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(6)

        # --- 1. turn the normalised action into position targets ---
        for i in range(5):
            self.arm_target[i] = float(
                np.clip(self.arm_target[i] + action[i] * 0.05, self.lo[i], self.hi[i])
            )
        lo5, hi5 = float(self.lo[5]), float(self.hi[5])
        self.arm_target[5] = lo5 + (action[5] + 1.0) / 2.0 * (hi5 - lo5)

        # --- 2. hold the targets and step the physics ---
        self.data.ctrl[:6] = self.arm_target
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # --- 3. observations / contacts from the post-step state ---
        obs = self.get_obs()
        ee_pos = np.asarray(self.data.site_xpos[self.ee_site_id], dtype=np.float64)
        cube_pos = np.asarray(self.data.xpos[self.cube_body_id], dtype=np.float64)
        contact_fixed, contact_moving = self._contact_flags()
        both = bool(contact_fixed and contact_moving)
        self.ee_contact_prev = both

        dist = float(np.linalg.norm(ee_pos - cube_pos))
        cube_z = float(cube_pos[2])
        cube_dz = float(cube_z - self.cube_z0)

        reward = (
            -dist
            + 1.0 * float(both)
            + 10.0 * max(0.0, cube_dz)
            - 0.01 * float(np.dot(action, action))
        )

        terminated = False
        new_success = False
        if cube_dz >= 0.05 and both:
            terminated = True
            reward += 20.0
            if not self._success:
                new_success = True
                self._success = True
                self._info["success_step"] = self._elapsed + 1
        if abs(cube_pos[0]) > 0.28 + 0.05 or abs(cube_pos[1]) > 0.10 + 0.05:
            terminated = True
        if cube_z < 0.0 - 0.02:
            terminated = True

        self._elapsed += 1
        truncated = bool(self._elapsed >= self.max_episode_steps)

        info = {
            "is_success": bool(self._success),
            "success_step": self._info.get("success_step"),
            "terminated": bool(terminated),
            "truncated": truncated,
            "cube_dz": cube_dz,
            "cube_z": cube_z,
            "both": bool(both),
            "contact_fixed": bool(contact_fixed),
            "contact_moving": bool(contact_moving),
            "dist": dist,
            "ee_pos": ee_pos,
            "cube_pos": cube_pos,
            "cube_z0": float(self.cube_z0),
        }
        if new_success:
            info["success"] = True
        self._info = {"is_success": bool(self._success),
                      "success_step": info["success_step"]}
        self.last_info = info
        return obs, float(reward), bool(terminated), truncated, info

    def render(self):
        """Headless render.  ``human`` falls back to the same offscreen path."""
        if self.render_mode is None:
            return None

        if self._renderer is None:
            # MuJoCo takes height first, then width.
            self._renderer = mujoco.Renderer(
                self.model, height=self.image_height, width=self.image_width
            )

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.lookat[:] = self.cam_lookat
        cam.distance = self.cam_distance
        cam.azimuth = self.cam_azimuth
        cam.elevation = self.cam_elevation

        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = SO101LiftEnv(verbose=True)
    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)
    print("action shape:", env.action_space.shape)
    print("obs shape:", env.observation_space.shape)
    print("n_substeps:", env.n_substeps, "control_dt:", env.control_dt)
    obs, info = env.reset(seed=0)
    print("reset obs shape:", obs.shape, "dtype:", obs.dtype)
    print("reset info:", info)
    for k in range(3):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        print(
            f"step {k}: reward={rew:.4f} terminated={term} truncated={trunc} "
            f"dist={info['dist']:.4f} both={info['both']}"
        )
    env.close()
