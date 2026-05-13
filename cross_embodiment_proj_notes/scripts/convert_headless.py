"""Headless wrapper around convert_data_format_mj.py.

The upstream conversion script opens a MuJoCo GLFW viewer via
mujoco.viewer.launch_passive, which fails on headless hosts. This wrapper
monkey-patches launch_passive with a no-op stub so the conversion loop can
run, then delegates to the upstream main().
"""
from __future__ import annotations

import sys
import types
from pathlib import Path


def _install_noop_viewer() -> None:
    """Replace mujoco.viewer.launch_passive with a stub that records no state."""
    import mujoco.viewer as mjv

    class _NoopViewer:
        opt = types.SimpleNamespace(flags={})
        cam = types.SimpleNamespace(distance=0.0, elevation=0.0, azimuth=0.0)

        def sync(self) -> None:
            pass

        def close(self) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            self.close()

    def _launch_passive(*args, **kwargs):
        return _NoopViewer()

    mjv.launch_passive = _launch_passive  # type: ignore[assignment]


def main() -> None:
    _install_noop_viewer()

    retarget_root = Path(
        "/home/ubuntu/sky_workdir/holosoma/src/holosoma_retargeting"
    )
    if str(retarget_root) not in sys.path:
        sys.path.insert(0, str(retarget_root))

    import tyro
    from holosoma_retargeting.config_types.data_conversion import (
        DataConversionConfig,
    )
    from holosoma_retargeting.data_conversion.convert_data_format_mj import (
        main as convert_main,
    )

    cfg = tyro.cli(DataConversionConfig)
    convert_main(cfg)


if __name__ == "__main__":
    main()
