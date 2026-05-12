#!/usr/bin/env python3
"""Side-by-side viser visualization of the Booster T1 and Unitree G1 URDFs.

Loads each robot from its URDF (with meshes) and exposes per-joint controls
both as GUI sliders AND as draggable rotation handles in the 3D scene.
Modeled after:
  - https://viser.studio/main/examples/demos/urdf_visualizer/   (sliders)
  - https://viser.studio/main/examples/demos/smpl_visualizer/   (3D gizmos)

Run it with one of the conda environments set up for this repo:

    # Setup 1 (IsaacSim/IsaacLab env — see SETUP_ISAACSIM_NOTES.md).
    # Requires `pip install viser` once into the env; yourdfpy + tyro are
    # already there.
    source scripts/source_isaacsim_setup.sh
    python vis/visualize_robots.py

    # Setup 2 (retargeting env — viser/yourdfpy/tyro ship in its deps).
    # See scripts/setup_retargeting.sh to create the `hsretargeting` env.
    source scripts/source_retargeting_setup.sh
    python vis/visualize_robots.py

Optional flags: `--port 8099`, `--spacing 1.5`,
`--t1-urdf /path/to/t1.urdf`, `--g1-urdf /path/to/g1.urdf`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tyro
import viser
import yourdfpy
from viser.extras import ViserUrdf

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_T1_URDF = REPO_ROOT / "src/holosoma/holosoma/data/robots/t1/t1_29dof.urdf"
DEFAULT_G1_URDF = REPO_ROOT / "src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf"


# ---------------------------------------------------------------------------
# Tiny wxyz-quaternion helpers (avoid coupling to viser.transforms internals).
# ---------------------------------------------------------------------------
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_log(q: np.ndarray) -> np.ndarray:
    """Axis-angle vector for a unit quaternion in wxyz layout."""
    w, v = float(q[0]), q[1:]
    vn = float(np.linalg.norm(v))
    if vn < 1e-12:
        return np.zeros(3)
    theta = 2.0 * np.arctan2(vn, w)
    # Wrap to (-pi, pi] so successive drags don't go the long way round.
    if theta > np.pi:
        theta -= 2.0 * np.pi
    return (v / vn) * theta


# ---------------------------------------------------------------------------
# Robot joint controls: sliders + 3D rotation handles, bidirectionally bound.
# ---------------------------------------------------------------------------
class RobotJointControls:
    """Wires one slider and one rotation-only transform gizmo per actuated joint.

    The gizmo is parented under the robot's root scene node, so its position is
    expressed in robot-local coordinates. On each drag update we:
      1. Compute the delta rotation vs. the last-accepted gizmo orientation.
      2. Project that delta onto the joint's rotation axis (in robot-local
         frame) to get a scalar angle delta.
      3. Push the delta into the slider — the slider's own callback then
         updates the URDF and refreshes every gizmo's position via FK.

    We never overwrite a gizmo's `wxyz` from FK (doing so would cancel an
    active drag); only positions get refreshed. The last-accepted orientation
    is tracked in `_ref_wxyz`.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        robot_root_node: str,
        vurdf: ViserUrdf,
        folder_label: str,
        gizmo_scale: float = 0.1,
    ) -> None:
        self.server = server
        self.robot_root_node = robot_root_node
        self.vurdf = vurdf
        self.urdf: yourdfpy.URDF = vurdf._urdf

        limits = vurdf.get_actuated_joint_limits()
        self.joint_names: list[str] = list(limits.keys())
        self.limits: list[tuple[float, float]] = [
            (
                lo if lo is not None else -np.pi,
                hi if hi is not None else np.pi,
            )
            for lo, hi in limits.values()
        ]
        self.joints: list[yourdfpy.Joint] = [self.urdf.joint_map[n] for n in self.joint_names]
        self.n = len(self.joints)
        # Prefer the neutral/zero pose when 0 is a valid joint value — otherwise
        # joints with one-sided limits (e.g. knees with lo=0) get initialized
        # deep into their range and the robot starts in a squat.
        self.initial_values: list[float] = [
            0.0 if lo <= 0.0 <= hi else (lo + hi) / 2.0
            for lo, hi in self.limits
        ]

        self.sliders: list[viser.GuiInputHandle[float]] = []
        self.controls: list[viser.TransformControlsHandle] = []
        self._ref_wxyz: list[np.ndarray] = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(self.n)]
        self._axes_local: list[np.ndarray] = [np.zeros(3) for _ in range(self.n)]

        self._build_sliders(folder_label)
        self._build_gizmos(gizmo_scale)
        self.apply_cfg()

    # -- construction -------------------------------------------------------
    def _build_sliders(self, folder_label: str) -> None:
        with self.server.gui.add_folder(folder_label):
            for name, (lo, hi), init in zip(
                self.joint_names, self.limits, self.initial_values, strict=True
            ):
                s = self.server.gui.add_slider(
                    label=name,
                    min=lo,
                    max=hi,
                    step=1e-3,
                    initial_value=init,
                )
                s.on_update(lambda _e: self.apply_cfg())
                self.sliders.append(s)

    def _build_gizmos(self, scale: float) -> None:
        for i, name in enumerate(self.joint_names):
            c = self.server.scene.add_transform_controls(
                f"{self.robot_root_node}/gizmos/{name}",
                scale=scale,
                disable_axes=True,
                disable_sliders=True,
                depth_test=False,
            )

            def make_cb(idx: int):
                def cb(_event: object) -> None:
                    self._on_gizmo_update(idx)

                return cb

            c.on_update(make_cb(i))
            self.controls.append(c)

    # -- callbacks ----------------------------------------------------------
    def _on_gizmo_update(self, i: int) -> None:
        q_new = np.asarray(self.controls[i].wxyz, dtype=np.float64)
        q_ref = self._ref_wxyz[i]
        # delta = q_new * q_ref^-1; then take the log to get an axis-angle vec.
        q_delta = _quat_mul(q_new, _quat_conj(q_ref))
        aa = _quat_log(q_delta)
        d_theta = float(aa @ self._axes_local[i])
        lo, hi = self.limits[i]
        new_val = float(np.clip(self.sliders[i].value + d_theta, lo, hi))
        # Accept the new gizmo reference BEFORE touching the slider so the
        # slider callback (which refreshes gizmo positions) doesn't re-enter
        # this logic with stale state.
        self._ref_wxyz[i] = q_new
        # Writing to .value fires the slider's on_update, which calls apply_cfg.
        # If the value is already at the clamp, viser may no-op — in that case
        # we still need FK to be up to date; apply_cfg is cheap so just call it.
        if abs(new_val - self.sliders[i].value) < 1e-12:
            self.apply_cfg()
        else:
            self.sliders[i].value = new_val

    # -- public api ---------------------------------------------------------
    def apply_cfg(self) -> None:
        """Push the current slider values through FK; refresh gizmo positions."""
        cfg = np.array([s.value for s in self.sliders])
        self.vurdf.update_cfg(cfg)
        for i, j in enumerate(self.joints):
            # Child-link frame coincides with the joint frame *after* rotation;
            # its rotation preserves the joint axis, so rot @ axis is fine for
            # projecting gizmo deltas.
            T = self.urdf.get_transform(j.child)
            self.controls[i].position = tuple(T[:3, 3].tolist())
            axis = T[:3, :3] @ np.asarray(j.axis, dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            self._axes_local[i] = axis / norm if norm > 1e-12 else axis

    def reset(self) -> None:
        for s, init in zip(self.sliders, self.initial_values, strict=True):
            s.value = init

    def set_gizmo_visibility(self, visible: bool) -> None:
        for c in self.controls:
            c.visible = visible


# ---------------------------------------------------------------------------
# Scene setup helpers.
# ---------------------------------------------------------------------------
def _load_robot(
    server: viser.ViserServer,
    urdf_path: Path,
    root_node_name: str,
    base_position: tuple[float, float, float],
    root_handle_scale: float = 0.35,
) -> tuple[viser.FrameHandle, viser.TransformControlsHandle, ViserUrdf]:
    """Load a URDF at `root_node_name` with a separate 6-DoF drag handle.

    The frame at `root_node_name` is the parent of the URDF meshes and the
    per-joint gizmos (under `{root_node_name}/gizmos/...`). A sibling
    `TransformControls` at `{root_node_name}_handle` drives the frame via its
    `on_update` callback — this way, hiding the gizmo doesn't also hide the
    meshes and joint gizmos parented under the frame.
    """
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    root_frame = server.scene.add_frame(
        root_node_name,
        position=base_position,
        show_axes=False,
    )
    urdf = yourdfpy.URDF.load(
        str(urdf_path),
        load_meshes=True,
        build_scene_graph=True,
    )
    vurdf = ViserUrdf(server, urdf_or_path=urdf, root_node_name=root_node_name)

    root_handle = server.scene.add_transform_controls(
        f"{root_node_name}_handle",
        scale=root_handle_scale,
        position=base_position,
        depth_test=False,
    )

    @root_handle.on_update
    def _(_event: object) -> None:
        root_frame.position = root_handle.position
        root_frame.wxyz = root_handle.wxyz

    return root_frame, root_handle, vurdf


def _drop_feet_to_floor(
    root_frame: viser.FrameHandle,
    root_handle: viser.TransformControlsHandle,
    vurdf: ViserUrdf,
    floor_z: float,
) -> None:
    scene = vurdf._urdf.scene or vurdf._urdf.collision_scene
    if scene is None:
        return
    min_z = float(scene.bounds[0, 2])
    x, y, _ = root_frame.position
    new_pos = (x, y, floor_z - min_z)
    root_frame.position = new_pos
    root_handle.position = new_pos


def main(
    t1_urdf: Path = DEFAULT_T1_URDF,
    g1_urdf: Path = DEFAULT_G1_URDF,
    spacing: float = 1.2,
    port: int = 8080,
    gizmo_scale: float = 0.08,
) -> None:
    """Launch a viser server rendering both robots with sliders + 3D handles.

    Args:
        t1_urdf: Path to the Booster T1 URDF.
        g1_urdf: Path to the Unitree G1 URDF.
        spacing: Distance (m) between the two robots along the y-axis.
        port: Port to bind the viser server on.
        gizmo_scale: Size of the per-joint rotation handles (meters).
    """
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")

    floor_z = 0.0
    half = spacing / 2.0
    t1_frame, t1_handle, t1 = _load_robot(server, t1_urdf, "/t1", base_position=(0.0, +half, 0.0))
    g1_frame, g1_handle, g1 = _load_robot(server, g1_urdf, "/g1", base_position=(0.0, -half, 0.0))

    t1_ctl = RobotJointControls(server, "/t1", t1, "Booster T1 joints", gizmo_scale=gizmo_scale)
    g1_ctl = RobotJointControls(server, "/g1", g1, "Unitree G1 joints", gizmo_scale=gizmo_scale)

    # Raise each robot so its lowest mesh vertex sits exactly on the floor.
    _drop_feet_to_floor(t1_frame, t1_handle, t1, floor_z)
    _drop_feet_to_floor(g1_frame, g1_handle, g1, floor_z)

    floor_size = max(4.0, spacing + 2.0)
    floor_thickness = 0.02
    server.scene.add_box(
        "/floor",
        dimensions=(floor_size, floor_size, floor_thickness),
        position=(0.0, 0.0, floor_z - floor_thickness / 2.0),
        color=(0.82, 0.82, 0.85),
    )
    server.scene.add_grid(
        "/grid",
        width=floor_size,
        height=floor_size,
        position=(0.0, 0.0, floor_z + 1e-3),
    )

    with server.gui.add_folder("Visibility"):
        show_visual_cb = server.gui.add_checkbox("Show visual meshes", initial_value=True)
        show_collision_cb = server.gui.add_checkbox("Show collision meshes", initial_value=False)
        show_handles_cb = server.gui.add_checkbox("Show joint handles", initial_value=True)
        show_root_cb = server.gui.add_checkbox("Show root handles", initial_value=True)

    @show_visual_cb.on_update
    def _(_event: object) -> None:
        t1.show_visual = show_visual_cb.value
        g1.show_visual = show_visual_cb.value

    @show_collision_cb.on_update
    def _(_event: object) -> None:
        t1.show_collision = show_collision_cb.value
        g1.show_collision = show_collision_cb.value

    @show_handles_cb.on_update
    def _(_event: object) -> None:
        t1_ctl.set_gizmo_visibility(show_handles_cb.value)
        g1_ctl.set_gizmo_visibility(show_handles_cb.value)

    @show_root_cb.on_update
    def _(_event: object) -> None:
        t1_handle.visible = show_root_cb.value
        g1_handle.visible = show_root_cb.value

    reset_button = server.gui.add_button("Reset pose")
    initial_t1_pos = t1_frame.position
    initial_g1_pos = g1_frame.position
    identity_wxyz = (1.0, 0.0, 0.0, 0.0)

    @reset_button.on_click
    def _(_event: object) -> None:
        t1_ctl.reset()
        g1_ctl.reset()
        t1_frame.position = initial_t1_pos
        g1_frame.position = initial_g1_pos
        t1_frame.wxyz = identity_wxyz
        g1_frame.wxyz = identity_wxyz
        t1_handle.position = initial_t1_pos
        g1_handle.position = initial_g1_pos
        t1_handle.wxyz = identity_wxyz
        g1_handle.wxyz = identity_wxyz

    print(f"[vis] T1 joints: {t1_ctl.n} | G1 joints: {g1_ctl.n}")
    print(f"[vis] Open the viser URL printed above (port {port}). Ctrl+C to exit.")
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)
