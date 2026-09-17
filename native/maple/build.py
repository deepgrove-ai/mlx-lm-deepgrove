#!/usr/bin/env python3
"""Build Maple's native extension with the active Python and MLX installation."""

import argparse
import importlib.metadata
import importlib.util
import platform
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path

VERSION = "0.1.0"


def run(*args):
    return subprocess.run(args, check=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", action="store_true", help="Package a prebuilt wheel")
    args = parser.parse_args()
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("The optional Maple extension requires Apple Silicon.")
    if importlib.metadata.version("mlx") != "0.32.0":
        raise SystemExit("The native extension requires MLX 0.32.0.")
    source = Path(__file__).resolve().parent
    root = source.parents[1]
    build = root / "build" / "maple-native"
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    tool_env = build / f"tools-{tag}"
    venv.EnvBuilder(with_pip=True, symlinks=True).create(tool_env)
    python = tool_env / "bin" / "python"
    # MLX 0.32.0 wheels use nanobind ABI v20. Newer nanobind ABIs cannot
    # exchange mlx.core.array objects, even when the module imports successfully.
    run(
        str(python),
        "-m",
        "pip",
        "install",
        "cmake>=3.27,<5",
        "nanobind==2.14.0",
        "wheel>=0.45,<1",
    )
    nanobind_dir = subprocess.check_output(
        [str(python), "-m", "nanobind", "--cmake_dir"], text=True
    ).strip()
    mlx_dir = Path(
        next(iter(importlib.util.find_spec("mlx").submodule_search_locations))
    )
    cmake = str(tool_env / "bin" / "cmake")
    cmake_dir = build / (f"cmake-{tag}" if args.wheel else "cmake")
    output = build / "wheel" / tag / "mlx_lm_maple" if args.wheel else root / "mlx_lm"
    run(
        cmake,
        "-S",
        str(source),
        "-B",
        str(cmake_dir),
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dnanobind_DIR={nanobind_dir}",
        f"-DMLX_DIR={mlx_dir / 'share' / 'cmake' / 'MLX'}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output}",
        f"-DCMAKE_BUILD_WITH_INSTALL_RPATH={'ON' if args.wheel else 'OFF'}",
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.0",
    )
    run(cmake, "--build", str(cmake_dir), "--parallel", "6")
    if args.wheel:
        abi = "cp" + sysconfig.get_config_var("SOABI").split("-")[1]
        info = output.parent / f"mlx_lm_maple_kernels-{VERSION}.dist-info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: mlx-lm-maple-kernels\nVersion: {VERSION}\n"
            "Summary: Prebuilt Maple kernels for MLX\nRequires-Python: >=3.12\n"
            "Requires-Dist: mlx==0.32.0\nLicense: MIT AND BSD-3-Clause\n"
        )
        (info / "WHEEL").write_text(
            "Wheel-Version: 1.0\nGenerator: maple-build\nRoot-Is-Purelib: false\n"
            f"Tag: {tag}-{abi}-macosx_26_0_arm64\n"
        )
        (info / "LICENSE").write_bytes((root / "LICENSE").read_bytes())
        license_path = next(
            Path(nanobind_dir).parents[1].glob("nanobind-*.dist-info/licenses/LICENSE")
        )
        (info / "LICENSE.nanobind").write_bytes(license_path.read_bytes())
        (root / "dist").mkdir(exist_ok=True)
        run(
            str(python),
            "-m",
            "wheel",
            "pack",
            str(output.parent),
            "--dest-dir",
            str(root / "dist"),
        )
    else:
        print("Built mlx_lm._maple_native.")


if __name__ == "__main__":
    main()
